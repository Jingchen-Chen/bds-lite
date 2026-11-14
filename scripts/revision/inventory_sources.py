import collections
import csv
import hashlib
import json
import os
from pathlib import Path

out = Path("/tmp/bds_audit_20261102")
out.mkdir(exist_ok=True)
roots = [Path("/home/jc") / n for n in ["bds-lite", "bds-lite-new", "bds-followup", "bds-latex"]]
rows = []
summary = {}
skip = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "node_modules"}
for root in roots:
    count = collections.Counter()
    size = 0
    for d, dirs, files in os.walk(root, followlinks=False):
        for n in list(dirs):
            p = Path(d) / n
            if n in skip or p.is_symlink():
                rows.append(
                    [
                        str(root),
                        str(p.relative_to(root)),
                        "excluded_environment_or_link",
                        0,
                        "",
                        os.readlink(p) if p.is_symlink() else "",
                    ]
                )
                dirs.remove(n)
        for n in sorted(files):
            p = Path(d) / n
            r = str(p.relative_to(root))
            s = p.lstat()
            if p.is_symlink():
                rows.append([str(root), r, "symlink", s.st_size, "", os.readlink(p)])
                continue
            h = hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
            rows.append([str(root), r, "file", s.st_size, h, ""])
            count[r.split("/")[0]] += 1
            size += s.st_size
    summary[str(root)] = {
        "files": sum(count.values()),
        "bytes": size,
        "top_level_files": dict(count),
    }
with (out / "inventory.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(["root", "relative_path", "kind", "bytes", "sha256", "link_target"])
    w.writerows(rows)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
index = {(r[0], r[1]): r for r in rows if r[2] == "file"}
with (out / "corresponding_files.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(
        ["source", "relative_path", "comparison_with_primary", "source_sha256", "primary_sha256"]
    )
    for r in rows:
        if r[0] == str(roots[0]) or r[2] != "file":
            continue
        p = index.get((str(roots[0]), r[1]))
        w.writerow(
            [
                r[0],
                r[1],
                "absent" if not p else "identical" if p[4] == r[4] else "conflict",
                r[4],
                p[4] if p else "",
            ]
        )
print(json.dumps(summary, indent=2))
