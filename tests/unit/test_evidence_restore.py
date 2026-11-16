"""Packaging must preserve exact records and never overwrite local work."""

import csv
import hashlib
import io
import tarfile

import pytest

from evidence import restore as recovery


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    source = tmp_path / "release"
    evidence = source / "evidence"
    evidence.mkdir(parents=True)
    data = b"case,score\npatient001,0.8\n"
    member = "results/example/per_case.csv"
    with tarfile.open(evidence / "supporting-records.tar.gz", "w:gz") as archive:
        item = tarfile.TarInfo(member)
        item.size = len(data)
        archive.addfile(item, io.BytesIO(data))
    manifest = evidence / "archived_paths.csv"
    row = dict(
        path=member,
        archive="evidence/supporting-records.tar.gz",
        member=member,
        sha256=hashlib.sha256(data).hexdigest(),
    )

    def write(**changes):
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row | changes)

    write()
    monkeypatch.setattr(recovery, "ROOT", source)
    return tmp_path / "restored", data, member, write


def test_restore_exact_bytes_check_and_repeat(snapshot):
    destination, data, member, _ = snapshot
    assert recovery.restore(destination, check=True) == (1, 0)
    assert not destination.exists()
    assert recovery.restore(destination) == (1, 1)
    assert (destination / member).read_bytes() == data
    assert recovery.restore(destination) == (1, 0)
    assert recovery.restore(destination, ("protocols",)) == (0, 0)


def test_restore_preserves_local_changes(snapshot):
    destination, _, member, _ = snapshot
    recovery.restore(destination)
    (destination / member).write_bytes(b"local work")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        recovery.restore(destination)
    assert (destination / member).read_bytes() == b"local work"


def test_restore_rejects_bad_hash(snapshot):
    destination, _, _, write = snapshot
    write(sha256="0" * 64)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        recovery.restore(destination)
    assert not destination.exists()


def test_archive_member_name_is_independent_of_local_symlinks(snapshot):
    destination, data, member, _ = snapshot
    alias = recovery.ROOT / member
    alias.parent.mkdir(parents=True)
    alias.symlink_to(destination / member)
    assert recovery.restore(destination) == (1, 1)
    assert (destination / member).read_bytes() == data


def test_restore_rejects_escape_and_symlink(snapshot, tmp_path):
    destination, _, _, write = snapshot
    write(path="../outside.csv")
    with pytest.raises(ValueError, match="Invalid archived path"):
        recovery.restore(destination)
    write()
    destination.mkdir()
    (destination / "results").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes destination"):
        recovery.restore(destination)
