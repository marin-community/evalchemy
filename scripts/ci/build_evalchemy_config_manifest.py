"""Build the digest-bearing release manifest for one evalchemy-config wheel."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from evalchemy_config import fingerprint

def package_version() -> str:
    """Read the installed standalone package version."""
    return version("evalchemy-config")


def release_manifest(*, wheel: Path, revision: str, repository: str) -> dict[str, object]:
    """Describe one immutable release asset pair and its provenance."""
    schema_fingerprint = fingerprint()
    tag = f"evalchemy-config-{schema_fingerprint}"
    manifest_name = "evalchemy-config-manifest.json"
    base_url = f"https://github.com/{repository}/releases/download/{tag}"
    return {
        "evalchemy_revision": revision,
        "package": "evalchemy-config",
        "package_version": package_version(),
        "schema_fingerprint": schema_fingerprint,
        "release_tag": tag,
        "wheel": {
            "filename": wheel.name,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "url": f"{base_url}/{wheel.name}",
        },
        "manifest_url": f"{base_url}/{manifest_name}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = release_manifest(wheel=args.wheel, revision=args.revision, repository=args.repository)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
