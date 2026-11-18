"""Convert fitted GBDT candidates to ONNX with a uniform I/O contract.

Contract: input ``X`` float32 ``[N, n_features]``; output ``probabilities`` float32 ``[N, C]``
(no ZipMap). Export side only; ``predictor.py`` never imports this module.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

from quantautoml.common.exceptions import ModelError, PackagingError
from quantautoml.models.candidates import BaseGBDCandidate

# Highest opset supported by the installed onnxmltools converters.
_CONVERTER_OPSET = 15
_IR_VERSION = 9


def export_candidate_onnx(model: BaseGBDCandidate, n_features: int, path: Path) -> None:
    """Write ``model`` as an ONNX file following the module's I/O contract."""
    if model.estimator_ is None:
        raise ModelError(f"Candidate '{model.name}' has not been fitted yet")
    if model.name == "lightgbm":
        onnx_model = _convert_lightgbm(model, n_features)
    elif model.name == "xgboost":
        onnx_model = _convert_xgboost(model, n_features)
    elif model.name == "catboost":
        onnx_model = _convert_catboost(model, path)
    else:
        raise PackagingError(f"No ONNX converter for model '{model.name}'")
    onnx_model = _normalize_io(onnx_model)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(path))


def _convert_lightgbm(model: BaseGBDCandidate, n_features: int) -> onnx.ModelProto:
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    return convert_lightgbm(
        model.estimator_,
        initial_types=[("X", FloatTensorType([None, n_features]))],
        zipmap=False,
        target_opset=_CONVERTER_OPSET,
    )


def _convert_xgboost(model: BaseGBDCandidate, n_features: int) -> onnx.ModelProto:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    return convert_xgboost(
        model.estimator_,
        initial_types=[("X", FloatTensorType([None, n_features]))],
        target_opset=_CONVERTER_OPSET,
    )


def _convert_catboost(model: BaseGBDCandidate, path: Path) -> onnx.ModelProto:
    tmp = path.with_suffix(".catboost.tmp")
    try:
        model.estimator_.save_model(str(tmp), format="onnx")
        return onnx.load(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


def _normalize_io(m: onnx.ModelProto) -> onnx.ModelProto:
    """Rename the input to ``X``, drop ZipMap/label outputs, expose float ``probabilities``."""
    graph = m.graph
    old_in = graph.input[0].name
    if old_in != "X":
        for node in graph.node:
            for i, name in enumerate(node.input):
                if name == old_in:
                    node.input[i] = "X"
        graph.input[0].name = "X"

    prob_name = _probabilities_tensor(graph)
    keep = [n for n in graph.node if n.op_type != "ZipMap"]
    del graph.node[:]
    graph.node.extend(keep)
    if prob_name != "probabilities":
        for node in graph.node:
            for i, name in enumerate(node.output):
                if name == prob_name:
                    node.output[i] = "probabilities"
            for i, name in enumerate(node.input):
                if name == prob_name:
                    node.input[i] = "probabilities"
    out = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, None])
    del graph.output[:]
    graph.output.append(out)
    del graph.value_info[:]
    m.ir_version = _IR_VERSION
    return m


def _probabilities_tensor(graph: onnx.GraphProto) -> str:
    """Name of the float [N, C] tensor feeding the ZipMap (or the probabilities output)."""
    for node in graph.node:
        if node.op_type == "ZipMap":
            return str(node.input[0])
    for out in graph.output:
        if out.name == "probabilities":
            return "probabilities"
    raise PackagingError("ONNX model has no probabilities output")
