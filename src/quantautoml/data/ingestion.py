"""Load features/labels files, align timestamps to UTC and enforce the float32 contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quantautoml.common.config import FeaturesConfig, PipelineConfig
from quantautoml.common.exceptions import InvalidDataError, TimestampAlignmentError
from quantautoml.common.types import Dataset, IngestionReport
from quantautoml.data.task_inference import infer_task_type

_NAIVE = "naive"
_NULL_MAPPING_KEYS = ("NaN", "nan", "NAN", "", "None", "null", "NULL")


@dataclass(frozen=True)
class _SelectedLabel:
    values: NDArray[np.generic]
    null_mask: NDArray[np.bool_]


class DataIngestion:
    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------ #
    def load(self) -> Dataset:
        cfg = self._cfg
        ts_col = cfg.data.timestamp_col
        features, f_tz, f_unit = self._read_frame(cfg.data.features_path, ts_col)
        n_feature_rows = features.height

        feature_cols = [c for c in features.columns if c != ts_col]
        if not feature_cols:
            raise InvalidDataError("features file contains no feature columns")
        features, bool_cols, string_maps = self._encode_features(features, feature_cols)

        label_cols: list[str] = []
        l_tz: str | None = None
        l_unit: str | None = None
        n_label_rows = 0
        if cfg.data.labels_path is not None:
            labels, l_tz, l_unit = self._read_frame(cfg.data.labels_path, ts_col)
            n_label_rows = labels.height
            label_cols = [c for c in labels.columns if c != ts_col]
            if cfg.labels.ignore_other_columns and cfg.labels.selected in label_cols:
                labels = labels.select([ts_col, cfg.labels.selected])
                label_cols = [cfg.labels.selected]
            clash = sorted(set(label_cols) & set(feature_cols))
            if clash:
                raise InvalidDataError(f"columns present in both files: {clash}")
            selected = cfg.labels.selected
            if selected not in label_cols:
                raise InvalidDataError(
                    f"selected label {selected!r} not found; available: {label_cols}"
                )
            self._require_numeric(
                labels, [c for c in label_cols if c != selected], "label"
            )
            joined = features.join(labels, on=ts_col, how="inner", maintain_order="left")
            n_joined = joined.height
            smaller = min(n_feature_rows, n_label_rows)
            if n_joined == 0 or n_joined / smaller < cfg.data.min_match_ratio:
                raise TimestampAlignmentError(
                    f"only {n_joined} of {smaller} timestamps match between features and "
                    f"labels (min_match_ratio={cfg.data.min_match_ratio})"
                )
        else:
            joined = features
            n_joined = n_feature_rows
            selected = None

        timestamps = joined[ts_col].to_numpy().astype("datetime64[us]")
        x64 = self._to_float64(joined, feature_cols)
        with np.errstate(over="ignore"):  # overflow is detected explicitly below
            x32 = x64.astype(np.float32)
        if np.any(np.isfinite(x64) & ~np.isfinite(x32)):
            raise InvalidDataError("feature values overflow float32 on cast")

        label_arrays = {
            c: self._to_float64(joined, [c])[:, 0] for c in label_cols if c != selected
        }
        if selected is not None:
            selected_values = self._extract_selected_label(joined, selected)
            if cfg.labels.mapping is not None:
                selected_values = self._apply_mapping(selected_values, cfg.labels.mapping, selected)
            # presence mask drives NaN counting / trailing-NaN detection for any label dtype
            label_arrays[selected] = np.where(selected_values.null_mask, np.nan, 0.0)
            if cfg.features is not None:
                label_arrays.update(
                    self._derive_forward_returns(cfg.features, x64, feature_cols, label_arrays, selected)
                )
        label_nan_counts = {c: int(np.isnan(a).sum()) for c, a in label_arrays.items()}
        trailing = {c: self._trailing_nan(a) for c, a in label_arrays.items()}
        max_trailing = max(trailing.values(), default=0)
        horizon = cfg.split.horizon_h
        if max_trailing > horizon:
            worst = max(trailing, key=lambda c: trailing[c])
            raise InvalidDataError(
                f"horizon_h={horizon} is smaller than the {max_trailing} trailing NaN of "
                f"label {worst!r}; set horizon_h >= {max_trailing}"
            )

        n_drop = trailing[selected] if selected is not None else 0
        keep = n_joined - n_drop
        if keep <= 0:
            raise InvalidDataError("no rows left after dropping trailing NaN labels")
        x32, timestamps = x32[:keep], timestamps[:keep]

        y = classes = task = None
        labels_aux: dict[str, NDArray[np.float32]] = {}
        if selected is not None:
            y_raw = selected_values.values[:keep]
            mid_nan_count = int(selected_values.null_mask[:keep].sum())
            inferred = infer_task_type(y_raw)
            y, classes, task = inferred.y, inferred.classes, inferred.task_type
            labels_aux = {
                c: a[:keep].astype(np.float32) for c, a in label_arrays.items() if c != selected
            }
        else:
            task = infer_task_type(None).task_type
            mid_nan_count = 0

        tz_label = f_tz if l_tz in (None, f_tz) else f"features={f_tz}, labels={l_tz}"
        unit_label = f_unit if l_unit in (None, f_unit) else f"features={f_unit}, labels={l_unit}"
        report = IngestionReport(
            n_feature_rows=n_feature_rows,
            n_label_rows=n_label_rows,
            n_joined_rows=n_joined,
            n_unmatched_dropped=(
                n_feature_rows - n_joined if cfg.data.labels_path is not None else 0
            ),
            n_trailing_label_nan_dropped=n_drop,
            source_timezone=tz_label,
            naive_utc_assumed=_NAIVE in (f_tz, l_tz),
            n_features=len(feature_cols),
            n_nan=int(np.isnan(x32).sum()),
            n_inf=int(np.isinf(x32).sum()),
            t_min=str(np.datetime_as_string(timestamps[0])),
            t_max=str(np.datetime_as_string(timestamps[-1])),
            label_nan_counts=label_nan_counts,
            max_trailing_nan=max_trailing,
            horizon_h=horizon,
            timestamp_precision_normalized=any(u not in (None, "us") for u in (f_unit, l_unit)),
            timestamp_original_time_unit=unit_label,
            bool_columns_cast=bool_cols,
            string_columns_encoded=string_maps,
            mid_nan_count=mid_nan_count,
        )
        return Dataset(
            X=x32,
            y=y,
            labels_aux=labels_aux,
            timestamps=timestamps,
            feature_names=tuple(feature_cols),
            selected_label=selected,
            report=report,
            task_type=task,
            classes_=classes,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _derive_forward_returns(
        fcfg: FeaturesConfig,
        x64: NDArray[np.float64],
        feature_cols: list[str],
        existing: dict[str, NDArray[np.float64]],
        selected: str,
    ) -> dict[str, NDArray[np.float64]]:
        """fwd_ret_h = price[t+h]/price[t] - 1 for ic_horizons columns absent from the labels."""
        missing = {
            h: name for h, name in fcfg.ic_horizons.items() if name not in existing and name != selected
        }
        if not missing:
            return {}
        if fcfg.price_col not in feature_cols:
            raise InvalidDataError(
                f"cannot derive forward returns {sorted(missing.values())}: "
                f"price column {fcfg.price_col!r} not found in features"
            )
        price = x64[:, feature_cols.index(fcfg.price_col)]
        n = price.shape[0]
        derived: dict[str, NDArray[np.float64]] = {}
        for h, name in missing.items():
            out = np.full(n, np.nan)
            if h < n:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = price[h:] / price[: n - h] - 1.0
                valid = (price[h:] > 0) & (price[: n - h] > 0)
                out[: n - h] = np.where(valid, ratio, np.nan)
            derived[name] = out
        return derived

    @staticmethod
    def _read_frame(path: Path, ts_col: str) -> tuple[pl.DataFrame, str, str]:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pl.read_parquet(path)
        elif suffix == ".csv":
            df = pl.read_csv(path, try_parse_dates=True)
        else:
            raise InvalidDataError(f"unsupported file type {suffix!r}: {path}")
        if ts_col not in df.columns:
            raise InvalidDataError(f"timestamp column {ts_col!r} not found in {path}")
        if df.height == 0:
            raise InvalidDataError(f"{path} contains no rows")

        col = df[ts_col]
        if col.dtype == pl.Date:
            col = col.cast(pl.Datetime("ns"))
        if not isinstance(col.dtype, pl.Datetime):
            raise InvalidDataError(f"column {ts_col!r} in {path} is not a parseable timestamp")
        tz = col.dtype.time_zone
        col = col.dt.replace_time_zone("UTC") if tz is None else col.dt.convert_time_zone("UTC")
        original_unit = col.dtype.time_unit
        col = col.dt.cast_time_unit("us")
        if col.null_count():
            raise TimestampAlignmentError(f"{path} contains null timestamps")
        df = df.with_columns(col.alias(ts_col))

        values = col.to_numpy()
        if np.any(values[1:] == values[:-1]):
            raise TimestampAlignmentError(f"duplicate timestamps in {path}")
        if np.any(values[1:] < values[:-1]):
            raise TimestampAlignmentError(f"timestamps in {path} are not monotonically increasing")
        return df, (_NAIVE if tz is None else tz), str(original_unit)

    @staticmethod
    def _encode_features(
        df: pl.DataFrame, cols: list[str]
    ) -> tuple[pl.DataFrame, list[str], dict[str, dict[str, int]]]:
        """Bool -> Int8 -> Float32; String/Categorical -> alphabetical ordinal -> Float32."""
        bool_cols: list[str] = []
        string_maps: dict[str, dict[str, int]] = {}
        exprs: list[pl.Expr] = []
        for c in cols:
            dtype = df.schema[c]
            if dtype == pl.Boolean:
                bool_cols.append(c)
                exprs.append(pl.col(c).cast(pl.Int8).cast(pl.Float32))
            elif dtype in (pl.String, pl.Categorical, pl.Enum):
                text = df[c].cast(pl.String)
                uniques = sorted(text.drop_nulls().unique().to_list())
                mapping = {v: i for i, v in enumerate(uniques)}
                string_maps[c] = mapping
                exprs.append(
                    pl.col(c).cast(pl.String).replace_strict(mapping, default=None)
                    .cast(pl.Float32).alias(c)
                )
        if exprs:
            df = df.with_columns(exprs)
        DataIngestion._require_numeric(df, cols, "feature")
        return df, bool_cols, string_maps

    @staticmethod
    def _extract_selected_label(df: pl.DataFrame, selected: str) -> _SelectedLabel:
        """Extract the selected label (numeric, bool or string) as raw values + null mask."""
        col = df[selected]
        null_mask = col.is_null().to_numpy()
        if col.dtype == pl.Boolean:
            col = col.cast(pl.Float64)
        if col.dtype in (pl.String, pl.Categorical, pl.Enum):
            values = np.array(col.cast(pl.String).to_list(), dtype=object)
        elif col.dtype.is_numeric():
            values = col.cast(pl.Float64).to_numpy()
            null_mask = null_mask | np.isnan(values)
        else:
            raise InvalidDataError(f"selected label {selected!r} has unsupported dtype {col.dtype}")
        return _SelectedLabel(values, null_mask)

    @staticmethod
    def _apply_mapping(
        sel: _SelectedLabel, mapping: dict[str, float], name: str
    ) -> _SelectedLabel:
        """Text label -> mapped number; mid-series nulls use any null key (NaN/nan/''/None/null),
        while trailing nulls stay null so trailing-NaN horizon drop remains intact.
        For numeric labels, a null-only mapping (e.g. ``NaN = 0``) fills mid-series NaN while
        rejecting non-null keys on numeric columns.
        """
        n_trailing = DataIngestion._trailing_nan(np.where(sel.null_mask, np.nan, 0.0))
        keep_end = sel.values.size - n_trailing

        if sel.values.dtype.kind != "O":
            non_null_keys = [k for k in mapping if k not in _NULL_MAPPING_KEYS]
            if non_null_keys:
                raise InvalidDataError(
                    f"labels.mapping needs a text label but {name!r} is numeric; remove the mapping"
                )
            null_val: float | None = None
            for nk in _NULL_MAPPING_KEYS:
                if nk in mapping:
                    null_val = float(mapping[nk])
                    break
            out_num = sel.values.astype(np.float64, copy=True)
            new_mask_num = sel.null_mask.copy()
            if null_val is not None and keep_end > 0:
                mid_nulls = new_mask_num[:keep_end]
                out_num[:keep_end] = np.where(mid_nulls, null_val, out_num[:keep_end])
                new_mask_num[:keep_end] = False
            return _SelectedLabel(out_num, new_mask_num)

        present = {v for v, isnull in zip(sel.values.tolist(), sel.null_mask.tolist()) if not isnull}
        unmapped = sorted(present - mapping.keys())
        if unmapped:
            raise InvalidDataError(f"labels.mapping lacks values of {name!r}: {unmapped}")

        null_mapped_val: float | None = None
        for nk in _NULL_MAPPING_KEYS:
            if nk in mapping and nk not in present:
                null_mapped_val = float(mapping[nk])
                break

        out = np.empty(sel.values.size, dtype=np.float64)
        new_mask = sel.null_mask.copy()
        for i, (v, isnull) in enumerate(zip(sel.values.tolist(), sel.null_mask.tolist())):
            if isnull:
                if null_mapped_val is not None and i < keep_end:
                    out[i] = null_mapped_val
                    new_mask[i] = False
                else:
                    out[i] = np.nan
            else:
                out[i] = mapping[v]
        return _SelectedLabel(out, new_mask)

    @staticmethod
    def _require_numeric(df: pl.DataFrame, cols: list[str], kind: str) -> None:
        bad = [c for c in cols if not df.schema[c].is_numeric()]
        if bad:
            raise InvalidDataError(f"non-numeric {kind} columns: {bad}")

    @staticmethod
    def _to_float64(df: pl.DataFrame, cols: list[str]) -> NDArray[np.float64]:
        arr = df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
        return np.ascontiguousarray(arr, dtype=np.float64)

    @staticmethod
    def _trailing_nan(a: NDArray[np.float64]) -> int:
        valid = np.flatnonzero(~np.isnan(a))
        return int(a.size if valid.size == 0 else a.size - 1 - valid[-1])
