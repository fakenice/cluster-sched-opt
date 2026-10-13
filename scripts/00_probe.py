#!/usr/bin/env python3
"""Peek inside the downloaded Alibaba PAI GPU trace archives.

Reports member names/sizes and the first rows + status-code distributions so the
ingest step can be written against observed structure instead of assumptions.
"""
import json
import tarfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    for arch in sorted(RAW.glob("*.tar.gz")):
        print("=" * 72)
        print(arch.name, f"{arch.stat().st_size/1e6:.1f} MB")
        try:
            with tarfile.open(arch, "r:gz") as tf:
                members = [m for m in tf.getmembers() if m.isfile()]
                for m in members[:12]:
                    print(f"  member {m.name}  {m.size/1e6:.2f} MB")
                if not members:
                    print("  (no file members)")
                    continue
                first = members[0]
                fh = tf.extractfile(first)
                head = [fh.readline().decode("utf-8", "replace").rstrip() for _ in range(6)]
        except Exception as exc:  # corrupted / partial download
            print(f"  ERROR {type(exc).__name__}: {exc}")
            continue
        for line in head:
            print("   |", line[:160])
    print("=" * 72)
    print("extracted:", json.dumps({p.name: str(p) for p in sorted(RAW.iterdir())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
