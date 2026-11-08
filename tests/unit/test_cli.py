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

    code = main(["run", "--config", str(_toml(tmp_path, fp, lp)), "--out", str(out)])

    assert code == 0
    assert (out / "train.npz").is_file()
    assert "mode=oof_assisted" in capsys.readouterr().out


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
