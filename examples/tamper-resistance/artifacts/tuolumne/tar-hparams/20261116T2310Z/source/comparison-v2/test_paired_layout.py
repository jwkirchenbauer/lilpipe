"""Synthetic paired titles and atomic publication tests; no real figures/data."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paired_layout


class PairedLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "misc/tar_compute_compare.py"
        spec = importlib.util.spec_from_file_location(
            "paired_canonical_fixture", source
        )
        cls.compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.compare)

    def tearDown(self):
        plt.close("all")

    def pair(self):
        frames = [0, 1, 8, 32, 128, 512, 1000]
        records = []
        for name, label, offset in (
            ("reproduced-1000", "B: retain2.5, inner1e-6, outer5e-6", 11),
            ("hparam-rio-lr1e-6", "RIO: retain4, inner1e-05, outer1e-05", 14),
        ):
            record = {
                "name": name,
                "label": label,
                "settings": {"broad_points": 3, "local_surface_points": 3},
                "actual": [
                    {
                        "phase": "attack",
                        "step": step,
                        "probe_loss": offset - step / 1000,
                    }
                    for step in frames
                ],
            }
            for field, radius in (("local", self.compare.LOCAL_RADIUS), ("broad", 125)):
                coordinates = np.linspace(-radius, radius, 3)
                record[field] = [
                    {
                        "phase": "attack",
                        "step": step,
                        "c0": x,
                        "c1": y,
                        "probe_loss": offset
                        - step / 1000
                        + 0.1 * ((x / radius) ** 2 + (y / radius) ** 2),
                    }
                    for step in frames
                    for x in coordinates
                    for y in coordinates
                ]
            records.append(record)
        return {
            "name": "fixture-pair",
            "records": records,
            "schedule_steps": 1000,
            "local_frames": frames,
            "broad_frames": [0, 128, 1000],
        }

    def test_wrapping_preserves_all_nonlabel_data_and_exact_surface_values(self):
        pair = self.pair()
        validations = {
            record["name"]: {"physical_comparable": True} for record in pair["records"]
        }
        wrapped = {
            **pair,
            "records": [
                paired_layout.wrapped_record(record) for record in pair["records"]
            ],
        }
        for first, second in zip(pair["records"], wrapped["records"]):
            self.assertEqual(
                paired_layout.record_identity(first),
                paired_layout.record_identity(second),
            )
            self.assertIn("\n", second["label"])
        for kind in paired_layout.SURFACE_KINDS:
            self.assertEqual(
                paired_layout.surface_identity(
                    self.compare.checkpoint_surface_data(pair, validations, kind)
                ),
                paired_layout.surface_identity(
                    self.compare.checkpoint_surface_data(wrapped, validations, kind)
                ),
            )

    def test_three_and_seven_row_surface_titles_do_not_overlap(self):
        pair = self.pair()
        wrapped = {
            **pair,
            "records": [
                paired_layout.wrapped_record(record) for record in pair["records"]
            ],
        }
        validations = {
            record["name"]: {"physical_comparable": True} for record in pair["records"]
        }
        for kind in paired_layout.SURFACE_KINDS:
            with self.subTest(kind=kind):
                audits = []

                def capture(figure, output):
                    audits.append(paired_layout.present_titles(figure))
                    expected = self.compare.checkpoint_surface_data(
                        pair, validations, kind
                    )
                    for axis in figure.axes[:-1]:
                        self.assertEqual(
                            axis.get_xlim(), (-expected["radius"], expected["radius"])
                        )
                        if not expected["delta"]:
                            self.assertEqual((axis.elev, axis.azim), (27, -58))
                            self.assertEqual(axis.get_zlim(), expected["limits"])

                with (
                    matplotlib.rc_context({"axes.titlesize": 9}),
                    patch.object(self.compare, "save_figure", capture),
                ):
                    self.compare.checkpoint_surface_figure(
                        wrapped, validations, kind, Path("fixture-not-written")
                    )
                self.assertEqual(len(audits), 1)
                self.assertFalse(audits[0]["title_overlap"])

    def test_archive_is_preserved_and_second_publication_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final, stage, archive = (
                root / "final",
                root / "stage",
                root / "layout-v1-original",
            )
            final.mkdir()
            stage.mkdir()
            (final / "image.png").write_bytes(b"original")
            (stage / "image.png").write_bytes(b"readable")
            paired_layout.publish_directories(final, stage, archive)
            self.assertEqual((archive / "image.png").read_bytes(), b"original")
            self.assertEqual((final / "image.png").read_bytes(), b"readable")
            with self.assertRaisesRegex(AssertionError, "overwrite"):
                paired_layout.publish_directories(final, stage, archive)

    def test_failed_staging_publish_rolls_original_directory_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final, stage, archive = (
                root / "final",
                root / "missing-stage",
                root / "archive",
            )
            final.mkdir()
            (final / "image.png").write_bytes(b"original")
            with self.assertRaises(FileNotFoundError):
                paired_layout.publish_directories(final, stage, archive)
            self.assertEqual((final / "image.png").read_bytes(), b"original")
            self.assertFalse(archive.exists())


if __name__ == "__main__":
    unittest.main()
