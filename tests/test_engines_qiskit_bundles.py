"""Quantum artifacts survive relocation and fail closed on incomplete outputs."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from chemrefine.engines.qiskit import bundles
from chemrefine.errors import ConfigError, OutputParseError


def _bundle(tmp_path):
    """Write a minimal complex scientific output."""
    return bundles.write_bundle(
        tmp_path / "state.json",
        kind="determinant_state",
        arrays={"amplitudes": np.array([1, 1j]) / np.sqrt(2)},
        metadata={"units": "hartree", "ordering": "alpha_then_beta"},
    )


def _edit(path, **updates):
    """Modify descriptor facts to exercise the public reader's integrity boundary."""
    record = json.loads(path.read_text())
    record.update(updates)
    path.write_text(json.dumps(record))


def test_complex_arrays_roundtrip_relocate_and_metadata_detaches(tmp_path):
    """Relative references and exact complex values survive copied run directories."""
    path = _bundle(tmp_path / "original")
    shutil.copytree(path.parent, tmp_path / "moved")
    loaded = bundles.read_bundle(tmp_path / "moved" / path.name)
    np.testing.assert_array_equal(loaded.arrays["amplitudes"], np.array([1, 1j]) / np.sqrt(2))
    assert not loaded.arrays["amplitudes"].flags.writeable
    metadata = loaded.metadata
    metadata["units"] = "other"
    assert loaded.metadata["units"] == "hartree"
    assert bundles.bundle_dependencies(path)["payload"].parent == path.parent


@pytest.mark.parametrize(
    "arrays,metadata,kind,limit",
    [
        ({"bad/name": np.ones(1)}, {}, "state", 100),
        ({"object": np.array([object()])}, {}, "state", 100),
        ({"amplitudes": np.ones(20)}, {}, "state", 10),
        ({"amplitudes": np.array([np.nan])}, {}, "state", 100),
        ({}, {"value": np.inf}, "state", 100),
        ({}, {"value": object()}, "state", 100),
        ({}, {}, "", 100),
        ({}, {}, "state", 0),
    ],
)
def test_writer_rejects_invalid_outputs_before_publishing(tmp_path, arrays, metadata, kind, limit):
    """Bad scientific values cannot become plausible successful output."""
    path = tmp_path / "invalid.json"
    with pytest.raises(ConfigError):
        bundles.write_bundle(path, kind=kind, arrays=arrays, metadata=metadata, max_bytes=limit)
    assert not path.exists()


def test_interrupted_write_keeps_previous_complete_bundle(tmp_path, monkeypatch):
    """An interrupted second write does not invalidate the first committed result."""
    path = _bundle(tmp_path)
    previous = path.read_bytes()

    def fail(*args, **kwargs):
        """Simulate failure between payload creation and manifest publication."""
        raise OSError("interrupted")

    monkeypatch.setattr(bundles, "_digest", fail)
    with pytest.raises(OSError, match="interrupted"):
        _bundle(tmp_path)
    assert path.read_bytes() == previous
    assert len(list(tmp_path.glob("*.npz"))) == 1


def _sized_metadata(path, size):
    """Fit metadata to an exact descriptor byte count, including its newline."""
    metadata = {"label": "\u03c0\U0001f331", "padding": ""}
    description = bundles.BundleDescription(
        kind="boundary",
        payload=f"{path.stem}.{'0' * 32}.npz",
        sha256="0" * 64,
        arrays={},
        metadata=metadata,
    )
    metadata["padding"] = "x" * (size - len(bundles.encode_bundle_descriptor(description)))
    return metadata


@pytest.mark.parametrize("offset", [-1, 0])
def test_exact_published_descriptor_limit_roundtrips(tmp_path, offset):
    """Every accepted boundary descriptor is immediately readable at the same limit."""
    path = tmp_path / "boundary.json"
    size = bundles.MAX_DESCRIPTOR_BYTES + offset
    metadata = _sized_metadata(path, size)
    bundles.write_bundle(path, kind="boundary", arrays={}, metadata=metadata)
    document = path.read_bytes()
    assert len(document) == size
    assert document.endswith(b"\n")
    result = bundles.read_bundle(path)
    assert result.metadata == metadata
    assert bundles.encode_bundle_descriptor(result.description) == document
    assert bundles.bundle_dependencies(path)["payload"].is_file()


@pytest.mark.parametrize("previous", [False, True])
def test_oversized_published_descriptor_cleans_up_and_preserves_output(tmp_path, previous):
    """One extra published byte fails atomically, including when replacing valid output."""
    path = tmp_path / "boundary.json"
    if previous:
        bundles.write_bundle(path, kind="boundary", arrays={}, metadata={"previous": True})
    before = {entry.name: entry.read_bytes() for entry in tmp_path.iterdir()}
    metadata = _sized_metadata(path, bundles.MAX_DESCRIPTOR_BYTES + 1)
    with pytest.raises(ConfigError, match="descriptor exceeds its size limit"):
        bundles.write_bundle(path, kind="boundary", arrays={}, metadata=metadata)
    assert {entry.name: entry.read_bytes() for entry in tmp_path.iterdir()} == before
    if previous:
        assert bundles.read_bundle(path).metadata == {"previous": True}


@pytest.mark.parametrize("name", ["../outside.npz", "/outside.npz", "bad\\file.npz", "", ".", ".."])
def test_payload_must_stay_beside_descriptor(tmp_path, name):
    """Malformed output paths never permit reading an unrelated file."""
    path = _bundle(tmp_path)
    _edit(path, payload=name)
    with pytest.raises(OutputParseError, match="filename"):
        bundles.read_bundle(path)
    with pytest.raises(ConfigError):
        bundles.bundle_dependencies(path)


def test_external_symlink_is_refused(tmp_path):
    """Even a local-looking filename cannot escape through a symlink."""
    path = _bundle(tmp_path / "run")
    external = tmp_path / "outside.npz"
    external.write_bytes(b"outside")
    (path.parent / "linked.npz").symlink_to(external)
    _edit(path, payload="linked.npz")
    with pytest.raises(OutputParseError, match="escapes"):
        bundles.read_bundle(path)


def test_missing_corrupt_and_oversized_descriptors(tmp_path, monkeypatch):
    """Output parsing has a stable public exception for IO and schema failures."""
    path = tmp_path / "missing.json"
    with pytest.raises(OutputParseError):
        bundles.read_bundle(path)
    path.write_text("not json")
    with pytest.raises(OutputParseError):
        bundles.read_bundle(path)
    path = _bundle(tmp_path)
    _edit(path, format_version=99)
    with pytest.raises(OutputParseError):
        bundles.read_bundle(path)
    monkeypatch.setattr(bundles, "MAX_DESCRIPTOR_BYTES", 2)
    with pytest.raises(OutputParseError, match="size limit"):
        bundles.read_bundle(path)
    with pytest.raises(ConfigError, match="size limit"):
        _bundle(tmp_path)


def test_changed_payload_or_allocation_is_refused(tmp_path):
    """Digest, allocation and dimensions are independent integrity checks."""
    path = _bundle(tmp_path)
    with pytest.raises(OutputParseError, match="max_bytes"):
        bundles.read_bundle(path, max_bytes=1)
    _edit(path, arrays={"amplitudes": {"shape": [-1], "dtype": "<c16"}})
    with pytest.raises(OutputParseError, match="nonnegative"):
        bundles.read_bundle(path)
    path = _bundle(tmp_path)
    payload = bundles.bundle_dependencies(path)["payload"]
    payload.write_bytes(b"tampered")
    with pytest.raises(OutputParseError, match="digest"):
        bundles.read_bundle(path)
    payload.write_bytes(b"x" * 100000)
    with pytest.raises(OutputParseError, match="allocation"):
        bundles.read_bundle(path)


@pytest.mark.parametrize("mode", ["members", "shape", "nonfinite", "expansion", "version", "v2"])
def test_archive_header_checked_before_array_allocation(tmp_path, mode):
    """A forged huge NPY shape is refused before numpy can allocate its claim."""
    path = _bundle(tmp_path)
    payload = bundles.bundle_dependencies(path)["payload"]
    buffer = io.BytesIO()
    if mode in {"shape", "version", "v2"}:
        header = {
            "descr": "<c16",
            "fortran_order": False,
            "shape": (10**12,) if mode == "shape" else (2,),
        }
        if mode == "v2":
            np.lib.format.write_array_header_2_0(buffer, header)
            buffer.write(np.array([1, 1j]).astype("<c16").tobytes())
        elif mode == "version":
            buffer.write(b"\x93NUMPY\x03\x00")
        else:
            np.lib.format.write_array_header_1_0(buffer, header)
    else:
        np.save(buffer, np.array([np.nan, 0], dtype="<c16"))
    with ZipFile(payload, "w", compression=ZIP_DEFLATED) as archive:
        name = "other.npy" if mode == "members" else "amplitudes.npy"
        archive.writestr(name, b"x" * 100000 if mode == "expansion" else buffer.getvalue())
    _edit(path, sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
    if mode == "v2":
        assert bundles.read_bundle(path).arrays["amplitudes"][1] == 1j
    else:
        with pytest.raises(OutputParseError):
            bundles.read_bundle(path)


def test_payload_changed_between_header_check_and_load_is_refused(tmp_path, monkeypatch):
    """Re-check decoded array dimensions even if a concurrent writer replaces the file."""
    path = _bundle(tmp_path)
    payload = bundles.bundle_dependencies(path)["payload"]
    original_load = np.load

    def replace_before_load(*args, **kwargs):
        """Simulate replacement after the header and digest checks."""
        np.savez(payload, amplitudes=np.ones(3))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", replace_before_load)
    with pytest.raises(OutputParseError, match="disagrees"):
        bundles.read_bundle(path)
