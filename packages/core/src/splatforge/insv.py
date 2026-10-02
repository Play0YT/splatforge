"""Metadaten aus Insta360-Dateien (.insv) lesen.

Eine .insv-Datei ist ein normaler MP4-Container (lesbar mit FFmpeg). Zusätzlich hängt Insta360 am Ende
der Datei einen eigenen Block an („Trailer“), unter anderem mit Kameramodell, Seriennummer und der
Objektiv-Kalibrierung („offset“). Aufbau des Trailers (Little Endian):

    ... [Datensatz][Format u8][ID u8][Grösse u32] ...
    [32 Byte Füllung][Grösse u32][Version u32][Magic 32 Byte]

Die Datensätze liegen von hinten nach vorne. Datensatz 1 (Metadaten) ist ein Protobuf. Das Format ist
nicht offiziell dokumentiert; die Beschreibung stammt aus dem Projekt telemetry-parser
(https://github.com/AdrianEddy/telemetry-parser, MIT/Apache-2.0). Fehlt der Trailer oder ist er
unbekannt, liefert der Parser ``None`` statt eines Fehlers, denn das Video selbst bleibt lesbar.
"""

from __future__ import annotations

import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MAGIC = b"8db42d694ccc418790edff439fe026bf"
PADDING_SIZE = 32
TRAILER_SIZE = PADDING_SIZE + 4 + 4 + len(MAGIC)
RECORD_HEADER_SIZE = 1 + 1 + 4
RECORD_OFFSETS = 0
RECORD_METADATA = 1
FORMAT_PROTOBUF = 1
# Schutz gegen kaputte Dateien: so viele Datensätze werden höchstens gelesen
MAX_RECORDS = 256

# Dateinamen wie VID_20260101_120000_00_001.insv: _00_ und _10_ sind die beiden Objektive
_SPLIT_NAME = re.compile(r"^(?P<prefix>.+_)(?P<lens>00|10)(?P<suffix>_\d+\.insv)$", re.IGNORECASE)


@dataclass
class InsvMetadata:
    camera_type: str = ""
    serial_number: str = ""
    fw_version: str = ""
    file_type: str = ""
    creation_time: int = 0
    total_time: int = 0
    frame_rate: int = 0
    total_frames: int = 0
    dimension: tuple[int, int] | None = None
    offset: list[float] = field(default_factory=list)
    offset_v2: list[float] = field(default_factory=list)
    offset_v3: list[float] = field(default_factory=list)
    rolling_shutter_time: float = 0.0
    file_group_index: int | None = None
    file_group_total: int | None = None
    is_dewarp: bool = False
    has_gyro: bool = False
    trailer_version: int = 0
    records: list[int] = field(default_factory=list)

    @property
    def calibration(self) -> list[float]:
        """Die neueste vorhandene Objektiv-Kalibrierung."""
        return self.offset_v3 or self.offset_v2 or self.offset

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["has_calibration"] = bool(self.calibration)
        return data


# Minimaler Protobuf-Leser (nur Wire-Typen 0, 1, 2, 5), damit keine zusätzliche Abhängigkeit nötig ist


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("Varint über das Datenende hinaus")
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("Varint zu lang")


def parse_protobuf(data: bytes) -> dict[int, list[Any]]:
    """Zerlegt eine Protobuf-Nachricht in {Feldnummer: [Rohwerte]}."""
    fields: dict[int, list[Any]] = {}
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        number, wire = key >> 3, key & 0x07
        value: Any
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire == 1:
            value = data[pos : pos + 8]
            pos += 8
        elif wire == 2:
            length, pos = _varint(data, pos)
            value = data[pos : pos + length]
            pos += length
        elif wire == 5:
            value = data[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unbekannter Protobuf-Wire-Typ {wire}")
        if pos > len(data):
            raise ValueError("Protobuf-Feld über das Datenende hinaus")
        fields.setdefault(number, []).append(value)
    return fields


def _str(fields: dict[int, list[Any]], number: int) -> str:
    values = fields.get(number)
    return values[-1].decode("utf-8", errors="replace") if values else ""


def _int(fields: dict[int, list[Any]], number: int) -> int:
    values = fields.get(number)
    return int(values[-1]) if values and isinstance(values[-1], int) else 0


def _double(fields: dict[int, list[Any]], number: int) -> float:
    values = fields.get(number)
    if values and isinstance(values[-1], bytes) and len(values[-1]) == 8:
        return float(struct.unpack("<d", values[-1])[0])
    return 0.0


def _offset(text: str) -> list[float]:
    try:
        return [float(v) for v in text.split("_")] if text else []
    except ValueError:
        return []


def decode_metadata(data: bytes) -> InsvMetadata:
    f = parse_protobuf(data)
    meta = InsvMetadata(
        serial_number=_str(f, 1),
        camera_type=_str(f, 2),
        fw_version=_str(f, 3),
        file_type=_str(f, 4),
        offset=_offset(_str(f, 5)),
        creation_time=_int(f, 7),
        total_time=_int(f, 10),
        has_gyro=bool(f.get(14)),
        frame_rate=_int(f, 20),
        rolling_shutter_time=_double(f, 25),
        total_frames=_int(f, 40),
        is_dewarp=bool(_int(f, 43)),
        offset_v2=_offset(_str(f, 53)),
        offset_v3=_offset(_str(f, 54)),
    )
    if 19 in f:
        dim = parse_protobuf(f[19][-1])
        meta.dimension = (_int(dim, 1), _int(dim, 2))
    if 26 in f:
        group = parse_protobuf(f[26][-1])
        meta.file_group_index = _int(group, 2)
        meta.file_group_total = _int(group, 4)
    return meta


def _read_records(fh: Any, file_size: int, extra_size: int) -> dict[int, tuple[int, bytes]]:
    """Liest die Datensätze von hinten nach vorne. Gibt {ID: (Format, Daten)} zurück."""
    records: dict[int, tuple[int, bytes]] = {}
    extra_start = file_size - extra_size
    pos = file_size - TRAILER_SIZE  # Ende des letzten Datensatzes
    for _ in range(MAX_RECORDS):
        header_start = pos - RECORD_HEADER_SIZE
        if header_start <= extra_start:
            break
        fh.seek(header_start)
        fmt, rec_id, size = struct.unpack("<BBI", fh.read(RECORD_HEADER_SIZE))
        data_start = header_start - size
        if data_start < extra_start:
            break
        fh.seek(data_start)
        data = fh.read(size)
        if rec_id == RECORD_OFFSETS:
            # Neuere Firmware: Inhaltsverzeichnis mit Position und Grösse jedes Datensatzes
            entries = {}
            for i in range(0, len(data) - 9, 10):
                eid, _efmt, esize, eoffset = struct.unpack("<BBII", data[i : i + 10])
                if eid > 0:
                    entries[eid] = (eoffset, esize)
            for eid, (eoffset, esize) in entries.items():
                fh.seek(extra_start + eoffset)
                edata = fh.read(esize)
                efmt, eid2, esize2 = struct.unpack("<BBI", fh.read(RECORD_HEADER_SIZE))
                if eid2 == eid and esize2 == esize:
                    records[eid] = (efmt, edata)
            if records:
                return records
        else:
            records.setdefault(rec_id, (fmt, data))
        pos = data_start
    return records


def read_metadata(path: Path) -> InsvMetadata | None:
    """Liest den Insta360-Trailer. ``None``, wenn die Datei keinen (bekannten) Trailer hat."""
    try:
        file_size = path.stat().st_size
        if file_size < TRAILER_SIZE:
            return None
        with path.open("rb") as fh:
            fh.seek(file_size - TRAILER_SIZE)
            trailer = fh.read(TRAILER_SIZE)
            if trailer[-len(MAGIC) :] != MAGIC:
                return None
            extra_size, version = struct.unpack("<II", trailer[PADDING_SIZE : PADDING_SIZE + 8])
            if extra_size <= TRAILER_SIZE or extra_size > file_size:
                return None
            records = _read_records(fh, file_size, extra_size)
    except (OSError, struct.error):
        return None
    meta = InsvMetadata()
    if RECORD_METADATA in records:
        fmt, data = records[RECORD_METADATA]
        if fmt == FORMAT_PROTOBUF:
            try:
                meta = decode_metadata(data)
            except ValueError:
                meta = InsvMetadata()
    meta.trailer_version = version
    meta.records = sorted(records)
    return meta


def partner_file(path: Path) -> Path | None:
    """Bei Aufnahmen, die pro Objektiv eine Datei schreiben (…_00_….insv und …_10_….insv),
    die Datei des anderen Objektivs, falls sie im selben Ordner liegt."""
    match = _SPLIT_NAME.match(path.name)
    if match is None:
        return None
    other = "10" if match["lens"] == "00" else "00"
    candidate = path.with_name(f"{match['prefix']}{other}{match['suffix']}")
    return candidate if candidate.is_file() else None


# Insta360 legt neben den Originalen (VID_…) eine Vorschau in niedriger Auflösung an (LRV_…), z. B.
# LRV_20220625_140410_11_008.insv zu VID_20220625_140410_00_008.insv und VID_20220625_140410_10_008.insv.
_PREVIEW_NAME = re.compile(r"^LRV_(?P<stamp>\d{8}_\d{6})_\d{2}_(?P<number>\d+)\.insv$", re.IGNORECASE)


def is_preview_file(path: Path) -> bool:
    """Ob es sich um eine Vorschaudatei (LRV) in niedriger Auflösung handelt."""
    return _PREVIEW_NAME.match(path.name) is not None


def original_files(path: Path) -> list[Path]:
    """Zu einer Vorschaudatei die Originaldateien (VID_…) im selben Ordner."""
    match = _PREVIEW_NAME.match(path.name)
    if match is None:
        return []
    original = re.compile(
        rf"^VID_{match['stamp']}_\d{{2}}_{match['number']}\.insv$",
        re.IGNORECASE,
    )
    return sorted(p for p in path.parent.iterdir() if original.match(p.name))
