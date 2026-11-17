"""In-memory synthetic canvas checks; no real-study figures are rendered."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


SPEC = importlib.util.spec_from_file_location(
    "layout_aggregate", Path(__file__).with_name("aggregate.py")
)
aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate)


class LayoutTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def check_bounds(self, figure, panels, columns):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        panel_boxes = [axis.get_tightbbox(renderer) for axis in panels]
        color_box = figure.axes[-1].get_tightbbox(renderer)
        title_box = figure._suptitle.get_window_extent(renderer)
        self.assertLess(max(box.x1 for box in panel_boxes) + 3, color_box.x0)
        self.assertGreater(
            title_box.y0, max(box.y1 for box in panel_boxes[:columns]) + 3
        )
        for index, first in enumerate(panel_boxes):
            for second in panel_boxes[index + 1 :]:
                self.assertFalse(first.overlaps(second), "Panel labels/titles overlap")
        for box in [*panel_boxes, color_box, title_box]:
            self.assertGreaterEqual(box.x0, figure.bbox.x0)
            self.assertLessEqual(box.x1, figure.bbox.x1)
            self.assertGreaterEqual(box.y0, figure.bbox.y0)
            self.assertLessEqual(box.y1, figure.bbox.y1)

    def test_fixed_five_panels_keep_colorbar_and_two_line_header_separate(self):
        figure, axes = plt.subplots(1, 5, figsize=(16.5, 3.5), squeeze=False)
        low, high = 1.114, 22.955
        for index, (axis, cell) in enumerate(
            zip(axes.flat, ("B", "R", "I", "O", "RIO"))
        ):
            radius = (1.1743, 1.231, 1.4436, 1.5585, 2.4944)[index]
            coordinates = np.linspace(-radius, radius, 3)
            image = axis.contourf(
                coordinates,
                coordinates,
                np.arange(9).reshape(3, 3) + index,
                levels=np.linspace(low, high, 18),
                vmin=low,
                vmax=high,
                extend="both",
            )
            axis.set_aspect("equal")
            axis.set_title(f"{cell} · ±{radius:.5g} L2", fontsize=9)
            axis.set_xlabel("c0 (global L2)")
            if index == 0:
                axis.set_ylabel("c1 (global L2)")
        aggregate.finish_figure_layout(
            figure,
            axes,
            image,
            "End-defense fixed attack plane; shared absolute CE scale [1.114,22.955]\nAttack-plane directions are fitted separately per condition",
        )
        self.check_bounds(figure, list(axes.flat), 5)
        self.assertEqual((image.norm.vmin, image.norm.vmax), (low, high))

    def test_three_four_seven_row_families_have_no_drawn_overlap(self):
        for count, radius in (
            (3, 125),
            (4, 0.15713339880469646),
            (7, 0.15713339880469646),
        ):
            with self.subTest(rows=count):
                records = [{"condition": cell} for cell in ("B", "R", "I", "O", "RIO")]
                frames = [0, 1, 8, 32, 128, 512, 1000][:count]
                coordinates = np.linspace(-radius, radius, 3)
                specs = {
                    (record["condition"], step): {
                        "coordinates": coordinates.copy(),
                        "values": np.arange(9, dtype=float).reshape(3, 3)
                        + index
                        + step / 1000,
                    }
                    for index, record in enumerate(records)
                    for step in frames
                }
                with patch.object(Figure, "savefig"), patch.object(plt, "close"):
                    aggregate.draw_grid_matrix(
                        plt,
                        records,
                        specs,
                        Path("fixture-not-written"),
                        title="Shared-reference local slices; matched physical coordinates",
                    )
                    figure = plt.gcf()
                self.check_bounds(figure, figure.axes[:-1], 5)
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
