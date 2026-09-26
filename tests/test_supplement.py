"""Anonymity and withdrawn-data gates for the TMLR supplementary archive."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from paper.make_supplement import _write_records, anonymize_real_record, check_text


class SupplementTests(unittest.TestCase):
    def test_private_identifiers_and_credential_shapes_are_rejected(self):
        for private in ("jaxblack", "jiapengli@microsoft.com",
                        "gho_" + "A" * 32):
            with self.subTest(private=private[:6]):
                with self.assertRaisesRegex(ValueError, "private identity"):
                    check_text(f"Some public data {private}", "test-record")

    def test_real_case_owner_and_private_revision_are_removed(self):
        source = {"repo": "jaxblack/limbo-api-validation-test", "source_commit": "private-sha",
                  "spec": {"model": "gpt-6-sol"}, "grade": {"EOS": True}}
        clean = anonymize_real_record(source)
        self.assertNotIn("repo", clean)
        self.assertNotIn("source_commit", clean)
        self.assertEqual(clean["grade"]["EOS"], source["grade"]["EOS"])
        self.assertIn("repo", source)

    def test_unreported_model_never_enters_zip_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "episodes.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in (
                {"episode_id": "include", "spec": {"model": "gpt-6-sol"}},
                {"episode_id": "exclude", "spec": {"model": "unreported-model"}},
            )) + "\n", encoding="utf-8")
            archive_path = root / "supplement.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                self.assertEqual(_write_records(archive, source, "results/e1/episodes.jsonl"), 1)
            with zipfile.ZipFile(archive_path) as archive:
                data = archive.read("results/e1/episodes.jsonl").decode("utf-8")
            self.assertIn("include", data)
            self.assertNotIn("exclude", data)

    def test_real_record_redaction_preserves_grade_in_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "episodes.jsonl"
            source.write_text(json.dumps({"episode_id": "case1", "repo": "jaxblack/private",
                                          "source_commit": "sha", "spec": {"model": "gpt-6-sol"},
                                          "grade": {"n_committed": 2}}) + "\n", encoding="utf-8")
            archive_path = root / "supplement.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                self.assertEqual(_write_records(archive, source, "results/real_github/episodes.jsonl", real=True), 1)
            with zipfile.ZipFile(archive_path) as archive:
                data = json.loads(archive.read("results/real_github/episodes.jsonl"))
            self.assertEqual(data["grade"]["n_committed"], 2)
            self.assertNotIn("repo", data)

    def test_escaped_surrogates_from_harness_output_remain_valid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "episodes.jsonl"
            source.write_text(json.dumps({"episode_id": "case1", "spec": {"model": "gpt-6-sol"},
                                          "assistant_texts": ["\udc94"]}) + "\n", encoding="utf-8")
            with zipfile.ZipFile(root / "supplement.zip", "w") as archive:
                self.assertEqual(_write_records(archive, source, "results/e1/episodes.jsonl"), 1)
            with zipfile.ZipFile(root / "supplement.zip") as archive:
                contents = archive.read("results/e1/episodes.jsonl").decode("utf-8")
            self.assertEqual(json.loads(contents)["assistant_texts"], ["\udc94"])


if __name__ == "__main__":
    unittest.main()
