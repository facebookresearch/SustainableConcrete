#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for boxcrete.utils."""

import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
from parameterized import parameterized

from boxcrete.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_X_COLUMNS,
    DEFAULT_Y_COLUMNS,
    DEFAULT_YSTD_COLUMNS,
    SustainableConcreteDataset,
    get_aggregate_constraint,
    get_binder_constraints,
    get_bounds,
    get_cement_replacement_constraints,
    get_concrete_constraints,
    get_day_zero_data,
    get_hrwr_constraints,
    get_mass_constraints,
    get_mortar_bounds,
    get_mortar_constraints,
    get_paste_constraints,
    get_proportional_sum_constraints,
    get_reference_point,
    get_subset_sum_tensors,
    get_sum_constraints,
    get_sum_equality_constraint,
    get_total_water_reducer_constraints,
    get_water_constraints,
    load_concrete_strength,
    predict_pareto,
    unique_elements,
)

# Column lists for concrete mixes (include Coarse Aggregates, exclude Curing Temp)
CONCRETE_COLUMNS = [
    "Cement",
    "Fly Ash",
    "Slag",
    "Water",
    "HRWR",
    "Coarse Aggregates",
    "Fine Aggregate",
    "Time",
]


def _create_test_dataframe(n=10, inject_nan_at=None):
    """Shared helper to create a test DataFrame matching DEFAULT_X_COLUMNS."""
    data = {
        "Mix ID": [f"Test_{i}" for i in range(n)],
        "Name": [f"Name_{i}" for i in range(n)],
        "Description": [f"Desc_{i}" for i in range(n)],
        "Cement": [400 + i * 10 for i in range(n)],
        "Fly Ash": [50 + i for i in range(n)],
        "Slag": [30 + i for i in range(n)],
        "Water": [200 + i * 5 for i in range(n)],
        "HRWR": [5 + i * 0.1 for i in range(n)],
        "Fine Aggregate": [1000 + i * 10 for i in range(n)],
        "Curing Temp (Cel.)": [20 + i for i in range(n)],
        "Time": [1, 7, 28] * 3 + [28],
        "GWP": [100 + i * 5 for i in range(n)],
        "Strength (Mean)": [3000 + i * 100 for i in range(n)],
        "Strength (Std)": [100 + i * 10 for i in range(n)],
        "# of measurements": [3.0] * n,
    }
    df = pd.DataFrame(data)
    if inject_nan_at:
        for row, col in inject_nan_at:
            df.loc[row, col] = float("nan")
    return df


def _create_test_dataset(n_samples=10, with_bounds=True, with_batch_names=True):
    """Shared helper to create a SustainableConcreteDataset."""
    X = torch.rand(n_samples, len(DEFAULT_X_COLUMNS))
    X[:, -1] = torch.tensor([1, 7, 28] * (n_samples // 3) + [1] * (n_samples % 3))
    Y = torch.rand(n_samples, len(DEFAULT_Y_COLUMNS))
    Ystd = torch.rand(n_samples, len(DEFAULT_Y_COLUMNS)) * 0.1
    bounds = (
        torch.stack([X.min(dim=0).values, X.max(dim=0).values]) if with_bounds else None
    )
    batch_names = (
        {
            "batch_1": list(range(n_samples // 2)),
            "batch_2": list(range(n_samples // 2, n_samples)),
        }
        if with_batch_names
        else None
    )
    return SustainableConcreteDataset(
        X=X,
        Y=Y,
        Ystd=Ystd,
        X_columns=list(DEFAULT_X_COLUMNS),
        Y_columns=list(DEFAULT_Y_COLUMNS),
        Ystd_columns=list(DEFAULT_YSTD_COLUMNS),
        bounds=bounds,
        batch_name_to_indices=batch_names,
    )


class TestSustainableConcreteDataset(unittest.TestCase):
    """Tests for the SustainableConcreteDataset class."""

    def test_initialization_and_properties(self):
        dataset = _create_test_dataset()
        self.assertEqual(dataset.X.shape, (10, len(DEFAULT_X_COLUMNS)))
        self.assertIsNotNone(dataset.Y)
        self.assertIsNotNone(dataset.Ystd)
        self.assertEqual(dataset.X_columns, list(DEFAULT_X_COLUMNS))
        self.assertEqual(dataset.Y_columns, list(DEFAULT_Y_COLUMNS))
        self.assertEqual(dataset.Ystd_columns, list(DEFAULT_YSTD_COLUMNS))
        torch.testing.assert_close(dataset.Yvar, dataset.Ystd.square())

    def test_initialization_invalid_time_column(self):
        with self.assertRaises(ValueError):
            SustainableConcreteDataset(
                X=torch.rand(5, 3),
                Y=torch.rand(5, 2),
                Ystd=torch.rand(5, 2),
                X_columns=["A", "B", "NotTime"],
                Y_columns=["Y1", "Y2"],
                Ystd_columns=["Ystd1", "Ystd2"],
            )

    def test_strength_data(self):
        dataset = _create_test_dataset()
        X, Y, Yvar, bounds = dataset.strength_data
        self.assertEqual(X.shape[0], dataset.X.shape[0])
        self.assertEqual(Y.shape[-1], 1)

    def test_gwp_data(self):
        dataset = _create_test_dataset()
        X, Y, Yvar, bounds = dataset.gwp_data
        self.assertLessEqual(X.shape[0], dataset.X.shape[0])
        self.assertEqual(X.shape[-1], len(DEFAULT_X_COLUMNS) - 1)

    @parameterized.expand([(1,), (7,), (28,)])
    def test_strength_data_by_time(self, time):
        dataset = _create_test_dataset(n_samples=12)
        X, Y, Yvar = dataset.strength_data_by_time(float(time))
        if X.shape[0] > 0:
            torch.testing.assert_close(X[:, -1], torch.full((X.shape[0],), float(time)))

    def test_unique_compositions(self):
        dataset = _create_test_dataset()
        unique, rev = dataset.unique_compositions
        self.assertEqual(unique[rev].shape, dataset.X[:, :-1].shape)
        indices = dataset.unique_composition_indices
        self.assertTrue(all(0 <= i < dataset.X.shape[0] for i in indices))

    def test_subselect_batch_names(self):
        dataset = _create_test_dataset()
        subset = dataset.subselect_batch_names(["batch_1"])
        self.assertLess(subset.X.shape[0], dataset.X.shape[0])

    def test_subselect_batch_names_no_batch_info_raises(self):
        dataset = _create_test_dataset(with_batch_names=False)
        with self.assertRaises(ValueError):
            dataset.subselect_batch_names(["batch_1"])


class TestConstraintFunction(unittest.TestCase):
    """Tests for all constraint-related functions."""

    def _verify(self, constraints, expected_len=2):
        self.assertEqual(len(constraints), expected_len)
        for indices, coeffs, value in constraints:
            self.assertIsInstance(indices, torch.Tensor)
            self.assertEqual(indices.shape, coeffs.shape)

    @parameterized.expand(
        [
            ("binder", get_binder_constraints, DEFAULT_X_COLUMNS, 100.0, 950.0),
            ("water", get_water_constraints, DEFAULT_X_COLUMNS, 0.35, 0.5),
            ("hrwr", get_hrwr_constraints, DEFAULT_X_COLUMNS, 0.0, 0.1),
            (
                "cement_repl",
                get_cement_replacement_constraints,
                DEFAULT_X_COLUMNS,
                0.0,
                0.5,
            ),
            ("aggregate", get_aggregate_constraint, CONCRETE_COLUMNS, 0.5, 2.0),
            ("mass", get_mass_constraints, CONCRETE_COLUMNS, 3600.0, 4400.0),
            ("paste", get_paste_constraints, CONCRETE_COLUMNS, 0.16, 0.35),
            (
                "total_wr",
                get_total_water_reducer_constraints,
                DEFAULT_X_COLUMNS,
                0.0,
                0.1,
            ),
            (
                "total_wr_mrwr",
                get_total_water_reducer_constraints,
                list(DEFAULT_X_COLUMNS[:-1]) + ["MRWR", "Time"],
                0.0,
                0.1,
            ),
        ]
    )
    def test_constraint_functions(self, name, func, columns, lower, upper):
        self._verify(func(columns, lower, upper))

    def test_get_mortar_constraints(self):
        eq, ineq = get_mortar_constraints(DEFAULT_X_COLUMNS, min_wb=0.35)
        self.assertGreater(len(eq), 0)
        self.assertGreater(len(ineq), 0)

    def test_get_concrete_constraints(self):
        self._verify(get_concrete_constraints(CONCRETE_COLUMNS), expected_len=10)

    def test_get_sum_equality_constraint(self):
        indices, coeffs, value = get_sum_equality_constraint(
            DEFAULT_X_COLUMNS, ["Cement", "Fly Ash", "Slag"], 500.0
        )
        self.assertEqual(value, 500.0)
        self.assertTrue(torch.all(coeffs != 0))

    def test_get_sum_constraints(self):
        self._verify(
            get_sum_constraints(
                DEFAULT_X_COLUMNS, ["Cement", "Fly Ash", "Slag"], 100.0, 900.0
            )
        )

    @parameterized.expand(
        [
            (["Cement"], ["Cement", "Fly Ash", "Slag"], 0.0, 1.0),
            (["Water"], ["Cement", "Fly Ash", "Slag"], 0.35, 0.5),
        ]
    )
    def test_get_proportional_sum_constraints(self, num, den, lo, hi):
        self._verify(
            get_proportional_sum_constraints(DEFAULT_X_COLUMNS, num, den, lo, hi)
        )

    def test_get_subset_sum_tensors(self):
        indices, coeffs = get_subset_sum_tensors(DEFAULT_X_COLUMNS, ["Cement", "Water"])
        self.assertEqual(len(indices), 2)
        self.assertEqual(coeffs.sum().item(), 2)


class TestBoundsFunction(unittest.TestCase):
    """Tests for bounds-related functions."""

    @parameterized.expand([(DEFAULT_X_COLUMNS,), (DEFAULT_X_COLUMNS[:-1],)])
    def test_get_mortar_bounds(self, columns):
        bounds = get_mortar_bounds(columns)
        self.assertEqual(bounds.shape, (2, len(columns)))
        self.assertTrue(torch.all(bounds[0] <= bounds[1]))

    def test_get_bounds(self):
        bounds = get_bounds(CONCRETE_COLUMNS)
        self.assertEqual(bounds.shape, (2, len(CONCRETE_COLUMNS)))
        self.assertTrue(torch.all(bounds[0] <= bounds[1]))


class TestUtilityFunction(unittest.TestCase):
    """Tests for utility helper functions."""

    @parameterized.expand(
        [
            ([1, 2, 3, 2, 1], [1, 2, 3]),
            (["a", "b", "a", "c"], ["a", "b", "c"]),
            ([], []),
        ]
    )
    def test_unique_elements(self, input_list, expected):
        self.assertEqual(unique_elements(input_list), expected)

    def test_get_reference_point(self):
        ref = get_reference_point()
        self.assertEqual(ref.shape, (3,))

    @parameterized.expand(
        [(32, None), (64, torch.tensor([[0, 0, 0], [100, 100, 28]]).float())]
    )
    def test_get_day_zero_data(self, n, bounds):
        X = torch.rand(10, 3)
        X_0, Y_0, Yvar_0 = get_day_zero_data(X, bounds=bounds, n=n)
        self.assertEqual(X_0.shape[0], n)
        expected_time = (
            bounds[0, -1].expand(n)
            if bounds is not None
            else X.amin(dim=0)[-1].expand(n)
        )
        torch.testing.assert_close(X_0[:, -1], expected_time)
        torch.testing.assert_close(Y_0, torch.zeros(n, 1))


class TestDataLoading(unittest.TestCase):
    """Tests for data loading functionality."""

    def test_load_from_dataframe(self):
        df = _create_test_dataframe()
        dataset = load_concrete_strength(data_path=df)
        self.assertIsInstance(dataset, SustainableConcreteDataset)
        self.assertEqual(dataset.X.shape[0], len(df))
        self.assertTrue(torch.all(dataset.Y[:, 0] <= 0))  # GWP negated

    def test_handles_missing_data(self):
        df = _create_test_dataframe(
            inject_nan_at=[(0, "Cement"), (3, "Strength (Mean)")]
        )
        dataset = load_concrete_strength(data_path=df)
        self.assertEqual(dataset.X.shape[0], 8)  # 10 - 2 dropped

    def test_batch_filter(self):
        df = _create_test_dataframe()
        df["Mix ID"] = [
            f"BatchA_{i}" if i < 5 else f"BatchB_{i}" for i in range(len(df))
        ]
        dataset = load_concrete_strength(data_path=df, batch_names=["BatchA"])
        self.assertLessEqual(dataset.X.shape[0], len(df))

    @unittest.skipUnless(
        os.path.exists(DEFAULT_DATA_PATH), "Real CSV data not available"
    )
    def test_load_from_csv_path(self):
        dataset = load_concrete_strength(data_path=DEFAULT_DATA_PATH)
        self.assertIsInstance(dataset, SustainableConcreteDataset)
        self.assertGreater(dataset.X.shape[0], 0)


class TestPredictPareto(unittest.TestCase):
    """Tests for predict_pareto function."""

    @patch("boxcrete.utils.sample_q_batches_from_polytope")
    def test_predict_pareto(self, mock_sampler):
        d_in, n = 3, 8
        mock_sampler.return_value = torch.rand(n, 1, d_in)
        mock_model = MagicMock()
        mock_post = MagicMock()
        mock_post.mean = (
            torch.tensor(
                [
                    [-100, 3000, 4000],
                    [-200, 3500, 4500],
                    [-150, 2000, 3000],
                    [-80, 4000, 5000],
                    [-300, 5000, 6000],
                    [-50, 1000, 2000],
                    [-120, 3200, 4200],
                    [-250, 4500, 5500],
                ]
            )
            .unsqueeze(-2)
            .float()
        )
        mock_post.variance = torch.ones_like(mock_post.mean) * 100
        mock_model.posterior.return_value = mock_post

        X, Y, Ystd = predict_pareto(
            model_list=mock_model,
            pareto_dims=[0, 1],
            ref_point=torch.tensor([-400.0, 1000.0, 2000.0]),
            bounds=torch.stack([torch.zeros(d_in), torch.ones(d_in) * 1000]),
            equality_constraints=[],
            inequality_constraints=[],
            num_candidates=n,
        )
        self.assertEqual(Y.shape[-1], 2)
        self.assertGreater(Y.shape[0], 0)
        if Y.shape[0] > 1:
            self.assertTrue(torch.all(Y[1:, 0] >= Y[:-1, 0]))


@unittest.skipUnless(os.path.exists(DEFAULT_DATA_PATH), "Real CSV data not available")
class TestRealDataIntegration(unittest.TestCase):
    """Integration tests using the real dataset for regression testing."""

    def test_load_with_batch_names(self):
        dataset = load_concrete_strength(
            data_path=DEFAULT_DATA_PATH,
            process_batch_names_from_mix_name=True,
            mix_name_column="Mix Name",
        )
        self.assertGreater(dataset.X.shape[0], 0)
        self.assertIsNotNone(dataset._batch_name_to_indices)
        self.assertGreater(len(dataset._batch_name_to_indices), 0)

    def test_gwp_data_property(self):
        dataset = load_concrete_strength(data_path=DEFAULT_DATA_PATH)
        dataset.bounds = get_mortar_bounds(dataset.X_columns)
        X, Y, Yvar, X_bounds = dataset.gwp_data
        self.assertGreater(X.shape[0], 0)
        self.assertEqual(Y.shape[-1], 1)
        self.assertEqual(X_bounds.shape[-1], X.shape[-1])

    def test_full_workflow(self):
        dataset = load_concrete_strength(data_path=DEFAULT_DATA_PATH)
        self.assertEqual(dataset.X.shape[-1], len(DEFAULT_X_COLUMNS))
        self.assertTrue(torch.all(dataset.Y[:, 0] <= 0))

        bounds = get_mortar_bounds(dataset.X_columns[:-1])
        eq, ineq = get_mortar_constraints(dataset.X_columns[:-1])
        self.assertEqual(bounds.shape[0], 2)
        self.assertGreater(len(eq), 0)

        X_s, Y_s, Yvar_s, _ = dataset.strength_data
        X_g, Y_g, Yvar_g, _ = dataset.gwp_data
        self.assertGreater(X_s.shape[0], X_g.shape[0])


if __name__ == "__main__":
    unittest.main()
