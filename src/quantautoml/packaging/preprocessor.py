"""Build ``preprocessor.onnx`` from a :class:`PreprocessorSpec` (export side only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper

from quantautoml.common.types import PreprocessorSpec

_OPSET = 17
_IR_VERSION = 9


def build_preprocessor_onnx(spec: PreprocessorSpec, path: Path) -> None:
    """Write an ONNX graph ``X[N, n_in] float32 -> Xp[N, len(final_idx)] float32``."""

    def i64(name: str, values: tuple[int, ...]) -> onnx.TensorProto:
        return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)

    fill_mask = np.zeros(spec.n_in, dtype=bool)
    fill_val = np.zeros(spec.n_in, dtype=np.float32)
    for j, v in spec.bool_fill:
        fill_mask[j] = True
        fill_val[j] = v
    inits = [
        numpy_helper.from_array(np.array(np.nan, dtype=np.float32), "nan"),
        numpy_helper.from_array(fill_mask, "fill_mask"),
        numpy_helper.from_array(fill_val, "fill_val"),
        i64("keep_idx", spec.keep_idx),
        i64("ind_idx", spec.indicator_idx),
        i64("final_idx", spec.final_idx),
    ]
    nodes = [
        helper.make_node("IsInf", ["X"], ["is_inf"]),
        helper.make_node("Where", ["is_inf", "nan", "X"], ["Xc"]),
        helper.make_node("IsNaN", ["Xc"], ["is_nan"]),
        helper.make_node("Gather", ["is_nan", "ind_idx"], ["ind_b"], axis=1),
        helper.make_node("Cast", ["ind_b"], ["ind"], to=TensorProto.FLOAT),
        helper.make_node("And", ["is_nan", "fill_mask"], ["do_fill"]),
        helper.make_node("Where", ["do_fill", "fill_val", "Xc"], ["Xf"]),
        helper.make_node("Gather", ["Xf", "keep_idx"], ["Xk"], axis=1),
        helper.make_node("Concat", ["Xk", "ind"], ["T"], axis=1),
        helper.make_node("Gather", ["T", "final_idx"], ["Xp"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "preprocessor",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", spec.n_in])],
        [helper.make_tensor_value_info("Xp", TensorProto.FLOAT, ["N", len(spec.final_idx)])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    model.ir_version = _IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def run_preprocessor_onnx(path: Path, X: NDArray[np.float32]) -> NDArray[np.float32]:
    """Run ``preprocessor.onnx`` (helper for tests and the precision validator)."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {"X": np.ascontiguousarray(X, dtype=np.float32)})
    return np.asarray(out, dtype=np.float32)
