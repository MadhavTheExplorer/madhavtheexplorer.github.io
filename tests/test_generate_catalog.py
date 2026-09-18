import json
import base64
import sys
import unittest
from pathlib import Path

import yaml
from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_catalog import build_catalog, decode_content, render_catalog, validate_manifest  # noqa: E402


class CatalogGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads((ROOT / "schema" / "project.schema.json").read_text())
        cls.manifest = yaml.safe_load((ROOT / "tests" / "fixtures" / "robot-arm.yml").read_text())

    def test_fixture_builds_deterministic_catalog_with_override(self):
        catalog = build_catalog(
            [("MadhavTheExplorer/Robot-Arm", "https://github.com/MadhavTheExplorer/Robot-Arm", self.manifest)],
            self.schema,
            {"MadhavTheExplorer/Robot-Arm": {"series": ["controls"]}},
        )

        self.assertEqual(catalog[0]["series"], ["controls"])
        self.assertEqual(catalog[0]["family"], "robotics-and-autonomy")
        self.assertEqual(render_catalog(catalog), render_catalog(catalog))

    def test_invalid_status_is_rejected(self):
        invalid = {**self.manifest, "status": "finished"}

        with self.assertRaises(ValueError):
            validate_manifest(invalid, self.schema, "fixture")

    def test_decodes_github_content_envelope(self):
        content = base64.b64encode(b"title: Example\n").decode("ascii")

        self.assertEqual(decode_content({"encoding": "base64", "content": content}), "title: Example\n")


if __name__ == "__main__":
    unittest.main()