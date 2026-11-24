"""CLI: ``quantautoml run --config X --out DIR``."""

from pathlib import Path

import pytest

from quantautoml.cli import main


def _toml(cfg_dir: Path, feats: Path, labels: Path) -> Path:
    p = cfg_dir / "c.toml"
    p.write_text(
        f'[data]\nfeatures_path = "{feats}"\nlabels_path = "{labels}"\n'
        '[labels]\nselected = "y_dir"\n[split]\nhorizon_h = 21\n',
        encoding="utf-8",
    )
    return p


def test_run_command_writes_artifacts(tmp_path: Path, make_frames, capsys) -> None:
    feats, labels = make_frames(n=3000)
    fp, lp = tmp_path / "f.parquet", tmp_path / "l.parquet"
    feats.write_parquet(fp)
    labels.write_parquet(lp)
    out = tmp_path / "out"

    cfg_path = _toml(tmp_path, fp, lp)
    code = main(["run", "--config", str(cfg_path), "--out", str(out)])

    assert code == 0
    assert (out / "config.toml").is_file()
    assert (out / "config.toml").read_text(encoding="utf-8") == cfg_path.read_text(encoding="utf-8")
    assert (out / "01_data" / "segments" / "train.npz").is_file()
    stdout = capsys.readouterr().out
    assert "mode=oof_assisted" in stdout
    assert "start=" in stdout
    assert "end=" in stdout
    assert "duration=" in stdout


def test_run_command_returns_nonzero_on_pipeline_error(tmp_path: Path, make_frames, capsys) -> None:
    feats, labels = make_frames(n=100)  # below minimum rows
    fp, lp = tmp_path / "f.parquet", tmp_path / "l.parquet"
    feats.write_parquet(fp)
    labels.write_parquet(lp)

    code = main(["run", "--config", str(_toml(tmp_path, fp, lp)), "--out", str(tmp_path / "o")])

    assert code == 1
    assert "error" in capsys.readouterr().err.lower()


def test_missing_subcommand_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_run_command_uses_default_config_and_out(
    tmp_path: Path, make_frames, monkeypatch: pytest.MonkeyPatch
) -> None:
    feats, labels = make_frames(n=3000)
    fp, lp = tmp_path / "f.parquet", tmp_path / "l.parquet"
    feats.write_parquet(fp)
    labels.write_parquet(lp)
    (tmp_path / "config.toml").write_text(
        f'[data]\nfeatures_path = "{fp}"\nlabels_path = "{lp}"\n'
        '[labels]\nselected = "y_dir"\n[split]\nhorizon_h = 21\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    code = main(["run"])

    assert code == 0
    assert (tmp_path / "runs" / "default" / "01_data" / "segments" / "train.npz").is_file()


def test_diagnose_command_writes_report_and_returns_zero(tmp_path: Path, capsys) -> None:
    import json

    import numpy as np
    from bundle_helper import make_bundle

    names = ["x0", "x1", "x2", "x3", "Open", "Close"]
    run_dir = tmp_path / "run_diag"
    make_bundle(run_dir / "05_packaging", names=names)
    (run_dir / "01_data" / "segments").mkdir(parents=True, exist_ok=True)
    (run_dir / "01_data" / "02_task.json").write_text(
        json.dumps({"task_type": "binary_classification", "classes": [0, 1], "feature_names": names}),
        encoding="utf-8",
    )
    (run_dir / "config.toml").write_text(
        '[data]\nfeatures_path = "f.parquet"\n[split]\nhorizon_h = 5\n'
        "[backtest]\nrank_window = 50\nic_window = 100\n",
        encoding="utf-8",
    )
    rng = np.random.default_rng(3)
    n = 300
    X = rng.normal(size=(n, 6)).astype(np.float32)
    close = 100.0 * np.cumprod(1.0 + 0.001 * rng.normal(size=n))
    X[:, 4] = close.astype(np.float32)
    X[:, 5] = close.astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    ts = np.datetime64("2024-01-01T00:00:00", "us") + np.arange(n) * np.timedelta64(5, "m")
    np.savez(
        run_dir / "01_data" / "segments" / "oos.npz",
        X=X,
        y=y,
        timestamps=ts,
        index=np.arange(n, dtype=np.int64),
    )

    code = main(["diagnose", "--run-dir", str(run_dir)])
    assert code == 0
    out_file = run_dir / "06_backtest" / "03_regime_diagnostics.json"
    assert out_file.is_file()
    stdout = capsys.readouterr().out
    assert "oos" in stdout


def test_diagnose_command_returns_nonzero_on_missing_run_dir(tmp_path: Path, capsys) -> None:
    code = main(["diagnose", "--run-dir", str(tmp_path / "does_not_exist")])
    assert code == 1
    assert "error" in capsys.readouterr().err.lower()
