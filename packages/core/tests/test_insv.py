"""Lesen von Insta360-Dateien (.insv) mit nachgebauten Testdateien."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import needs_ffmpeg
from splatforge.cli import main
from splatforge.frames import export_frames, inspect_input
from splatforge.insv import decode_metadata, parse_protobuf, partner_file, read_metadata
from synthetic import insv_metadata, insv_trailer, make_insv


def test_decode_metadata() -> None:
    meta = decode_metadata(insv_metadata())
    assert meta.camera_type == "Insta360 X4"
    assert meta.serial_number == "IXSE00TEST"
    assert meta.fw_version == "v1.2.3"
    assert meta.dimension == (2880, 2880)
    assert meta.frame_rate == 30
    assert (meta.file_group_index, meta.file_group_total) == (1, 2)
    assert len(meta.offset_v3) == 24
    assert meta.calibration == meta.offset_v3


def test_protobuf_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_protobuf(b"\x0a\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff")


@pytest.mark.parametrize("with_offsets", [False, True])
def test_read_trailer(tmp_path: Path, with_offsets: bool) -> None:
    path = tmp_path / "VID ä.insv"
    trailer = insv_trailer({1: (1, insv_metadata()), 3: (0, b"\x01" * 64)}, with_offsets=with_offsets)
    path.write_bytes(b"\x00" * 1000 + trailer)
    meta = read_metadata(path)
    assert meta is not None
    assert meta.camera_type == "Insta360 X4"
    assert meta.records == [1, 3]
    assert meta.trailer_version == 3


def test_no_trailer(tmp_path: Path) -> None:
    path = tmp_path / "plain.insv"
    path.write_bytes(b"\x00" * 500)
    assert read_metadata(path) is None


def test_broken_trailer_does_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "broken.insv"
    path.write_bytes(
        b"\x00" * 100
        + bytes(32)
        + (99999).to_bytes(4, "little")
        + bytes(4)
        + b"8db42d694ccc418790edff439fe026bf"
    )
    assert read_metadata(path) is None


def test_partner_file(tmp_path: Path) -> None:
    a = tmp_path / "VID_20260101_120000_00_001.insv"
    b = tmp_path / "VID_20260101_120000_10_001.insv"
    a.write_bytes(b"")
    assert partner_file(a) is None
    b.write_bytes(b"")
    assert partner_file(a) == b
    assert partner_file(b) == a
    assert partner_file(tmp_path / "anders.insv") is None


@needs_ffmpeg
@pytest.mark.parametrize(
    ("layout", "expected"), [("two_streams", "two_streams"), ("side_by_side", "side_by_side")]
)
def test_inspect_insv(tmp_path: Path, layout: str, expected: str) -> None:
    path = make_insv(tmp_path / "Aufnahme ü.insv", layout=layout)
    info = inspect_input(path)
    assert info.camera_type == "dual_fisheye"
    assert info.lens_layout == expected
    assert len(info.lenses) == 2
    assert info.insv is not None and info.insv["camera_type"] == "Insta360 X4"
    assert info.insv["has_calibration"]
    assert (info.width, info.height) == (320, 320)


@needs_ffmpeg
def test_split_files_with_partner(tmp_path: Path) -> None:
    a = make_insv(tmp_path / "VID_20260101_120000_00_001.insv", layout="single")
    make_insv(tmp_path / "VID_20260101_120000_10_001.insv", layout="single")
    info = inspect_input(a)
    assert info.lens_layout == "split_files"
    assert [Path(lens["path"]).name for lens in info.lenses] == [
        "VID_20260101_120000_00_001.insv",
        "VID_20260101_120000_10_001.insv",
    ]


@needs_ffmpeg
def test_insv_without_trailer_still_readable(tmp_path: Path) -> None:
    path = make_insv(tmp_path / "exportiert.insv", trailer=b"")
    info = inspect_input(path)
    assert info.camera_type == "dual_fisheye"
    assert info.insv is None
    assert any("Keine Insta360-Metadaten" in n for n in info.notes)


@needs_ffmpeg
@pytest.mark.parametrize("layout", ["two_streams", "side_by_side"])
def test_export_frames_per_lens(tmp_path: Path, layout: str) -> None:
    path = make_insv(tmp_path / "x.insv", layout=layout)
    folders = export_frames(path, tmp_path / "Bilder", count=6, max_edge=640)
    assert [f.name for f in folders] == ["objektiv_1", "objektiv_2"]
    from splatforge.imageio import read_image

    for folder in folders:
        images = sorted(folder.glob("*.jpg"))
        assert 4 <= len(images) <= 8
        first = read_image(images[0])
        assert first is not None and first.shape[:2] == (320, 320)
    # Die beiden Objektive zeigen unterschiedliche Bilder
    a = read_image(sorted(folders[0].glob("*.jpg"))[0])
    b = read_image(sorted(folders[1].glob("*.jpg"))[0])
    assert a is not None and b is not None and abs(a.astype(int) - b.astype(int)).mean() > 10


@needs_ffmpeg
def test_export_frames_refuses_non_empty_folder(tmp_path: Path) -> None:
    path = make_insv(tmp_path / "x.insv")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "wichtig.txt").write_text("x", encoding="utf-8")
    assert main(["frames", str(path), "--out", str(tmp_path / "out")]) == 1


@needs_ffmpeg
def test_analyze_cli_and_run_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = make_insv(tmp_path / "x.insv")
    assert main(["analyze", str(path)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["inputs"][0]["lens_layout"] == "two_streams"


def test_preview_file_and_originals(tmp_path: Path) -> None:
    from splatforge.insv import is_preview_file, original_files

    lrv = tmp_path / "LRV_20220625_140410_11_008.insv"
    vid0 = tmp_path / "VID_20220625_140410_00_008.insv"
    vid1 = tmp_path / "VID_20220625_140410_10_008.insv"
    other = tmp_path / "VID_20220625_140410_00_009.insv"
    for p in (lrv, vid0, vid1, other):
        p.write_bytes(b"")
    assert is_preview_file(lrv)
    assert not is_preview_file(vid0)
    assert original_files(lrv) == [vid0, vid1]
    assert original_files(vid0) == []


@needs_ffmpeg
def test_analyze_warns_about_preview_file(tmp_path: Path) -> None:
    lrv = make_insv(tmp_path / "LRV_20220625_140410_11_008.insv", layout="side_by_side")
    info = inspect_input(lrv)
    assert info.lens_layout == "side_by_side"
    assert any("Vorschaudatei (LRV)" in n for n in info.notes)
    assert any("nicht im selben Ordner" in n for n in info.notes)


@needs_ffmpeg
def test_split_files_metadata_from_partner_and_fixed_order(tmp_path: Path) -> None:
    """Wie bei der ONE RS: Metadaten nur in _00_, angegeben wird die _10_-Datei."""
    a = make_insv(tmp_path / "VID_20220625_140410_00_008.insv", layout="single")
    b = make_insv(tmp_path / "VID_20220625_140410_10_008.insv", layout="single", trailer=b"")
    info = inspect_input(b)
    assert info.insv is not None and info.insv["camera_type"] == "Insta360 X4"
    assert any("Partnerdatei" in n for n in info.notes)
    assert [Path(lens["path"]) for lens in info.lenses] == [a, b]
