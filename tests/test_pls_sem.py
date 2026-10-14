import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pls_sem import (
    ModelValidationError,
    calculate_htmt,
    cronbach_alpha,
    estimate_measurement_model,
    find_out_of_range_items,
    get_measurement_items,
)


def sample_data(rows=180, seed=13):
    rng = np.random.default_rng(seed)
    pq = rng.normal(size=rows)
    bt = rng.normal(size=rows)
    latent_scores = {
        "PQ": pq,
        "BT": bt,
        "CS": 0.5 * pq + 0.4 * bt + rng.normal(scale=0.7, size=rows),
        "BI": 0.3 * pq + 0.4 * bt + rng.normal(scale=0.7, size=rows),
    }
    latent_scores["RI"] = (
        0.5 * latent_scores["CS"]
        + 0.3 * latent_scores["BI"]
        + rng.normal(scale=0.7, size=rows)
    )
    latent_scores["WOM"] = (
        0.4 * latent_scores["CS"]
        + 0.4 * latent_scores["BI"]
        + rng.normal(scale=0.7, size=rows)
    )
    data = {"Participant ID": np.arange(rows)}
    for scale, latent in latent_scores.items():
        for item in range(1, 4):
            data[f"{scale}{item}"] = latent + rng.normal(scale=0.25, size=rows)
    return pd.DataFrame(data)


class MeasurementModelTests(unittest.TestCase):
    def test_scale_detection_excludes_participant_id(self):
        data = sample_data()
        items = get_measurement_items(data)

        self.assertEqual(tuple(items), ("PQ", "BT", "CS", "BI", "RI", "WOM"))
        self.assertNotIn("Participant ID", sum(items.values(), []))

    def test_cronbach_alpha_uses_standard_covariance_formula(self):
        data = pd.DataFrame(
            {"item1": [1, 2, 3, 4], "item2": [1, 3, 2, 5], "item3": [2, 3, 4, 6]}
        )
        expected = 3 / 2 * (
            1 - data.var(ddof=1).sum() / data.sum(axis=1).var(ddof=1)
        )

        self.assertAlmostEqual(cronbach_alpha(data), expected)

    def test_htmt_uses_heterotrait_to_monotrait_ratio(self):
        data = pd.DataFrame(
            {
                "A1": [1, 2, 3, 5, 6],
                "A2": [2, 3, 4, 6, 7],
                "B1": [1, 3, 2, 6, 5],
                "B2": [2, 1, 4, 5, 7],
            }
        )
        correlations = data.corr().abs()
        expected = correlations.loc[["A1", "A2"], ["B1", "B2"]].to_numpy().mean()
        expected /= np.sqrt(
            correlations.loc["A1", "A2"] * correlations.loc["B1", "B2"]
        )

        result = calculate_htmt(data, {"A": ["A1", "A2"], "B": ["B1", "B2"]})
        self.assertAlmostEqual(result.loc[0, "HTMT (Klasik)"], expected)

    def test_htmt_is_undefined_when_monotrait_correlation_is_zero(self):
        data = pd.DataFrame(
            {
                "A1": [1, 1, -1, -1],
                "A2": [1, -1, 1, -1],
                "B1": [1, 2, 3, 4],
                "B2": [2, 4, 1, 3],
            }
        )

        result = calculate_htmt(data, {"A": ["A1", "A2"], "B": ["B1", "B2"]})

        self.assertTrue(np.isnan(result.loc[0, "HTMT (Klasik)"]))

    def test_likert_range_validation_ignores_missing_and_non_item_columns(self):
        data = pd.DataFrame(
            {
                "PQ1": [1, 3, 5, np.nan],
                "PQ2": [1, 2, 3, 4],
                "Participant ID": [0, 8, 9, 10],
            }
        )
        data.loc[0, "PQ2"] = 0

        self.assertEqual(find_out_of_range_items(data), ["PQ2"])

    def test_model_returns_loadings_reliability_validity_and_paths(self):
        result = estimate_measurement_model(sample_data())

        self.assertEqual(len(result.loadings), 18)
        self.assertTrue(result.loadings["Dış yük"].between(-1, 1).all())
        self.assertEqual(len(result.reliability), 6)
        self.assertIn("Cronbach's Alpha", result.reliability)
        self.assertIn("Composite Reliability (rho_c)", result.reliability)
        self.assertIn("AVE", result.reliability)
        self.assertEqual(len(result.validity), 15)
        self.assertEqual(len(result.paths), 8)
        self.assertEqual(result.observations, 180)
        self.assertTrue(result.converged)
        for _, reliability in result.reliability.iterrows():
            loadings = result.loadings.loc[
                result.loadings["Ölçek"] == reliability["Ölçek"], "Dış yük"
            ].to_numpy()
            if loadings.sum() < 0:
                loadings = -loadings
            expected_rho_c = loadings.sum() ** 2 / (
                loadings.sum() ** 2 + (1 - loadings**2).sum()
            )
            self.assertAlmostEqual(reliability["Composite Reliability (rho_c)"], expected_rho_c)
            self.assertAlmostEqual(reliability["AVE"], np.mean(loadings**2))

    def test_loading_orientation_and_paths_are_transformed_consistently(self):
        data = sample_data()
        scale_items = get_measurement_items(data)
        loadings = {
            item: -0.8 if scale == "PQ" else 0.8
            for scale, items in scale_items.items()
            for item in items
        }
        path_coefficients = pd.DataFrame(
            0.0, index=scale_items, columns=scale_items
        )
        path_coefficients.loc["CS", "PQ"] = 0.25

        class StubModel:
            def outer_model(self):
                return pd.DataFrame(
                    {"loading": pd.Series(loadings, dtype=float)}
                )

            def path_coefficients(self):
                return path_coefficients

        with patch("pls_sem.Plspm", return_value=StubModel()):
            result = estimate_measurement_model(data)

        pq_loadings = result.loadings.loc[
            result.loadings["Ölçek"] == "PQ", "Dış yük"
        ]
        self.assertTrue((pq_loadings > 0).all())
        self.assertAlmostEqual(
            result.paths.query("Kaynak == 'PQ' and Hedef == 'CS'")[
                "Yol katsayısı"
            ].iloc[0],
            -0.25,
        )
        reliability = result.reliability.set_index("Ölçek").loc["PQ"]
        oriented = pq_loadings.to_numpy()
        self.assertAlmostEqual(
            reliability["Composite Reliability (rho_c)"],
            oriented.sum() ** 2 / (oriented.sum() ** 2 + (1 - oriented**2).sum()),
        )

    def test_nonconvergence_is_reported_as_an_analysis_error(self):
        data = sample_data()
        with patch("pls_sem.Plspm", side_effect=Exception("Could not converge")):
            with self.assertRaisesRegex(ModelValidationError, "Could not converge"):
                estimate_measurement_model(data)

    def test_missing_rows_are_dropped_and_id_is_not_analyzed(self):
        data = sample_data()
        data.loc[0, "PQ1"] = np.nan
        data["Participant ID"] = data["Participant ID"].astype(object)
        data.loc[1, "Participant ID"] = "not numeric"

        result = estimate_measurement_model(data)

        self.assertEqual(result.observations, 179)
        self.assertEqual(result.dropped_rows, 1)
        self.assertNotIn("Participant ID", result.item_columns)

    def test_cycles_are_rejected(self):
        data = sample_data()

        with self.assertRaisesRegex(ModelValidationError, "döngü"):
            estimate_measurement_model(data, [("PQ", "BT"), ("BT", "PQ")])

    def test_non_numeric_item_is_reported(self):
        data = sample_data()
        data["PQ1"] = data["PQ1"].astype(object)
        data.loc[0, "PQ1"] = "yanıt yok"

        with self.assertRaisesRegex(ModelValidationError, "PQ1"):
            estimate_measurement_model(data)


if __name__ == "__main__":
    unittest.main()
