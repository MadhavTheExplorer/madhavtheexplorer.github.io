#!/usr/bin/env python3
"""Generate Jekyll expedition data from public GitHub repository manifests."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "schema" / "project.schema.json"
DEFAULT_OVERRIDES = ROOT / "_data" / "repositories" / "overrides.yml"
DEFAULT_OUTPUT = ROOT / "_data" / "repositories" / "generated.yml"


def read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def validate_manifest(manifest: dict[str, Any], schema: dict[str, Any], source: str) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(manifest), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in errors
        )
        raise ValueError(f"Invalid manifest in {source}: {details}")


def build_catalog(
    manifests: list[tuple[str, str, dict[str, Any]]],
    schema: dict[str, Any],
    overrides: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    overrides = overrides or {}
    catalog = []
    for repository, url, manifest in manifests:
        validate_manifest(manifest, schema, repository)
        record = {"repository": repository, "url": url, **manifest}
        record.update(overrides.get(repository, {}))
        catalog.append(record)
    return sorted(catalog, key=lambda item: (not item["featured"], item["started"], item["title"]))


def github_request(path: str, token: str | None) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "madhavtheexplorer-catalog",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(f"https://api.github.com{path}", headers=headers)
    with urlopen(request, timeout=30) as response:
        content_type = response.headers.get_content_type()
        body = response.read().decode("utf-8")
        return json.loads(body) if content_type == "application/json" else body


def decode_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and response.get("encoding") == "base64":
        return base64.b64decode(response["content"]).decode("utf-8")
    raise ValueError("GitHub contents response is neither raw text nor base64 content")


def fetch_manifests(owner: str, token: str | None) -> list[tuple[str, str, dict[str, Any]]]:
    repositories = github_request(f"/users/{owner}/repos?type=owner&sort=full_name&per_page=100", token)
    manifests = []
    for repository in repositories:
        if repository["private"] or repository["archived"]:
            continue
        full_name = repository["full_name"]
        try:
            content = github_request(f"/repos/{full_name}/contents/.explorer/project.yml", token)
        except HTTPError as error:
            if error.code == 404:
                continue
            raise
        manifest = yaml.safe_load(decode_content(content))
        if not isinstance(manifest, dict):
            raise ValueError(f"Manifest in {full_name} must be a YAML object")
        manifests.append((full_name, repository["html_url"], manifest))
    return manifests


def render_catalog(catalog: list[dict[str, Any]]) -> str:
    header = "# Generated from public repository manifests. Do not edit by hand.\n"
    return header + yaml.safe_dump(catalog, sort_keys=False, allow_unicode=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="MadhavTheExplorer")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    overrides = read_yaml(args.overrides) if args.overrides.exists() else {}
    manifests = fetch_manifests(args.owner, os.getenv("GITHUB_TOKEN"))
    rendered = render_catalog(build_catalog(manifests, schema, overrides))

    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != rendered:
            raise SystemExit(f"Catalog is stale: run {Path(__file__).name}")
        print(f"Catalog is current with {len(manifests)} expedition(s).")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Generated {len(manifests)} expedition(s) in {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())