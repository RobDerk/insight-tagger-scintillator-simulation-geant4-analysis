import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import numpy as np
import pandas as pd

import compare_materials_energies as comparison


class CompareMaterialsEnergiesTest(unittest.TestCase):
    def test_analysis_keeps_absolute_and_relative_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "4mm.csv"
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                for event in (1, 2):
                    for hit, offset in (("top", 0.1), ("bottom", 0.09)):
                        for photon in range(160):
                            row = [0] * len(comparison.COLUMN_NAMES)
                            row[1] = event
                            row[3] = hit
                            row[5] = offset * event + photon * 0.01
                            row[10] = "Scintillation"
                            row[14] = "BC422"
                            row[15] = 1
                            writer.writerow(row)
                for event, hit in ((3, "top"), (4, "other")):
                    row = [0] * len(comparison.COLUMN_NAMES)
                    row[1], row[3], row[5] = event, hit, -1
                    row[10], row[14], row[15] = "Scintillation", "BC422", 1
                    writer.writerow(row)

            with mock.patch.object(comparison, "fit_gauss", return_value=(0, 0.02, 0, 0.001)) as fit:
                result = comparison.analyse_file(path)

        fits = result["fits"].set_index("Threshold")
        self.assertEqual(result["n_events"], 4)
        self.assertAlmostEqual(result["average_detected_photons_over_all_events"], 641 / 3)
        self.assertEqual(fits.loc[150, "Events"], 2)
        self.assertEqual(fits.loc[150, "Sigma_ps"], 20)
        self.assertAlmostEqual(fits.loc[150, "RelativeBrightnessToAverage"], 150 / (641 / 3) * 100)
        self.assertAlmostEqual(fits.loc[150, "RelativeBrightnessToMax"], 150 / 641 * 100)
        self.assertEqual(fits.loc[300, "Events"], 0)
        self.assertTrue(np.isnan(fits.loc[300, "Sigma_ps"]))
        np.testing.assert_allclose(fit.call_args_list[0].args[0], [0.01, 0.02])

    def test_main_saves_both_plot_types_after_one_analysis_per_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            out_dir = root / "plots_comparison"
            files = [
                data_root / material / f"{energy}MeV" / f"{length}mm.csv"
                for material, energy, length in (
                    ("BC422", 1, 4), ("BC422", 2, 4), ("EJ228", 1, 4),
                    ("BC422", 1, 70), ("EJ228", 1, 70),
                    ("Extra", 1, 4), ("Extra", 2, 4),
                )
            ]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            fits = pd.DataFrame({
                "Threshold": [1, 140, 150, 300],
                "Sigma_ps": [50, 51, 52, 53],
                "SigmaErr_ps": [1, 1, 1, 1],
                "RelativeBrightnessToAverage": [0.1, 14, 15, 30],
            })

            def analyse(path):
                return {
                    "material": path.parent.parent.name,
                    "energy_mev": float(path.parent.name.removesuffix("MeV")),
                    "n_events": 2,
                    "average_detected_photons_over_all_events": 1000,
                    "fits": fits,
                }

            saved_plots = {}
            savefig = matplotlib.figure.Figure.savefig

            def capture_plot(figure, path, *args, **kwargs):
                axis = figure.axes[0]
                saved_plots[path] = {
                    "xlabel": axis.get_xlabel(),
                    "xlim": axis.get_xlim(),
                    "ylim": axis.get_ylim(),
                    "curves": [container.lines[0].get_xdata().tolist() for container in axis.containers],
                }
                return savefig(figure, path, *args, **kwargs)

            with (
                mock.patch.object(comparison, "DATA_ROOT", data_root),
                mock.patch.object(comparison, "OUT_DIR", out_dir),
                mock.patch.object(comparison, "MATERIAL_FOLDERS", ["BC422", "EJ228"]),
                mock.patch.object(comparison, "analyse_file", side_effect=analyse) as analyse_mock,
                mock.patch.object(matplotlib.figure.Figure, "savefig", autospec=True, side_effect=capture_plot),
            ):
                comparison.main()

            self.assertCountEqual([call.args[0] for call in analyse_mock.call_args_list], files[:5])
            self.assertEqual(analyse_mock.call_count, 5)
            absolute_materials = out_dir / "4mm" / "materials_1MeV_abs.png"
            relative_materials = out_dir / "4mm" / "materials_1MeV_rel.png"
            self.assertEqual(set(saved_plots), {
                absolute_materials, relative_materials,
                out_dir / "4mm" / "energies_BC422_abs.png",
                out_dir / "4mm" / "energies_BC422_rel.png",
                out_dir / "70mm" / "materials_1MeV_abs.png",
                out_dir / "70mm" / "materials_1MeV_rel.png",
            })
            self.assertEqual(saved_plots[absolute_materials]["xlabel"], "photon threshold N")
            self.assertEqual(saved_plots[absolute_materials]["xlim"], (0, 125))
            self.assertEqual(saved_plots[absolute_materials]["curves"], [[1, 140]] * 2)
            self.assertEqual(saved_plots[relative_materials]["xlabel"], "threshold brightness / %")
            self.assertEqual(saved_plots[relative_materials]["xlim"], (0, 30))
            self.assertEqual(saved_plots[relative_materials]["ylim"], (20, 250))
            self.assertEqual(saved_plots[relative_materials]["curves"], [[0.1, 14, 15, 30]] * 2)
            for path in saved_plots:
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_missing_selected_material_is_reported_before_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            path = data_root / "BC422" / "1MeV" / "4mm.csv"
            path.parent.mkdir(parents=True)
            path.touch()
            with (
                mock.patch.object(comparison, "DATA_ROOT", data_root),
                mock.patch.object(comparison, "MATERIAL_FOLDERS", ["BC422", "EJ228"]),
                mock.patch.object(comparison, "analyse_file") as analyse,
            ):
                with self.assertRaisesRegex(SystemExit, "EJ228"):
                    comparison.main()
                analyse.assert_not_called()

    def test_empty_material_selection_is_reported(self):
        with mock.patch.object(comparison, "MATERIAL_FOLDERS", []):
            with self.assertRaisesRegex(SystemExit, "No materials selected"):
                comparison.main()


if __name__ == "__main__":
    unittest.main()
