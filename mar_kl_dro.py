"""
This module extracts the paper method used in the ``MAR-KL DRO`` for Newsvendor Problem
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import product
from typing import Any, Hashable, Mapping, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy import sparse


_OPTIMAL_STATUSES = frozenset({cp.OPTIMAL, cp.OPTIMAL_INACCURATE})


@dataclass(frozen=True)
class NewsvendorDROResult:
    """Solution returned by :meth:`MissingCovariateDRO.solve_newsvendor`."""

    order_quantity: float
    worst_case_cost: float
    likelihood_threshold: float
    likelihood_radius: float
    maximum_log_likelihood: float
    status: str
    solver: str | None


class MissingCovariateDRO:
    """
    Parameters
    ----------
    covariate_supports:
        Finite support of each covariate, in column order.
    outcome_support:
        Finite numerical support of the uncertain outcome.
    covariate_columns:
        DataFrame columns corresponding to ``covariate_supports``.  Defaults
        to ``X0``, ``X1``, ... .
    outcome_column:
        DataFrame column containing the fully observed outcome.
    mask_column:
        DataFrame column containing a binary missingness mask.  A value of 1
        means the corresponding covariate is missing.

    """

    def __init__(
        self,
        covariate_supports: Sequence[Sequence[Hashable]],
        outcome_support: Sequence[float],
        *,
        covariate_columns: Sequence[str] | None = None,
        outcome_column: str = "Y",
        mask_column: str = "missing_mask",
    ) -> None:
        supports = tuple(tuple(values) for values in covariate_supports)
        if not supports or any(not values for values in supports):
            raise ValueError("Each covariate must have a non-empty finite support.")
        if any(len(set(values)) != len(values) for values in supports):
            raise ValueError("Covariate supports must not contain duplicates.")

        outcomes = tuple(outcome_support)
        if not outcomes or len(set(outcomes)) != len(outcomes):
            raise ValueError("outcome_support must be non-empty and unique.")
        try:
            outcome_values = np.asarray(outcomes, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("Newsvendor outcomes must be numerical.") from exc
        if not np.all(np.isfinite(outcome_values)):
            raise ValueError("outcome_support must contain only finite values.")

        dimension = len(supports)
        columns = (
            tuple(covariate_columns)
            if covariate_columns is not None
            else tuple(f"X{j}" for j in range(dimension))
        )
        if len(columns) != dimension or len(set(columns)) != dimension:
            raise ValueError(
                "covariate_columns must be unique and match covariate_supports."
            )

        self.covariate_supports = supports
        self.outcome_support = outcomes
        self.covariate_columns = columns
        self.outcome_column = outcome_column
        self.mask_column = mask_column

        self.support_: tuple[tuple[Hashable, ...], ...] = tuple(
            product(*supports, outcomes)
        )
        self.support_index_: dict[tuple[Hashable, ...], int] = {
            state: index for index, state in enumerate(self.support_)
        }
        self._outcome_vector = np.asarray(
            [state[-1] for state in self.support_], dtype=float
        )

        self.compatibility_matrix_: sparse.csr_matrix | None = None
        self.mle_distribution_: np.ndarray | None = None
        self.maximum_log_likelihood_: float | None = None
        self.n_samples_: int | None = None

    @property
    def dimension(self) -> int:
        """Number of covariates."""

        return len(self.covariate_supports)

    @property
    def n_states(self) -> int:
        """Number of full ``(x, y)`` states in the Cartesian support."""

        return len(self.support_)

    def build_compatibility_matrix(self, data: pd.DataFrame) -> sparse.csr_matrix:
        """Construct the sparse partial-observation compatibility matrix.
        """

        self._validate_columns(data)
        if len(data) == 0:
            raise ValueError("data must contain at least one observation.")

        row_indices: list[int] = []
        column_indices: list[int] = []

        for row_position, (_, row) in enumerate(data.iterrows()):
            mask = self._parse_mask(row[self.mask_column])
            outcome = row[self.outcome_column]
            if pd.isna(outcome) or outcome not in self.outcome_support:
                raise ValueError(
                    f"Row {row_position}: outcome {outcome!r} is outside "
                    "outcome_support."
                )

            compatible_values: list[Sequence[Hashable]] = []
            for j, (column, support) in enumerate(
                zip(self.covariate_columns, self.covariate_supports)
            ):
                if mask[j] == 1:
                    compatible_values.append(support)
                    continue

                value = row[column]
                if pd.isna(value):
                    raise ValueError(
                        f"Row {row_position}: {column} is missing but mask[{j}] is 0."
                    )
                if value not in support:
                    raise ValueError(
                        f"Row {row_position}: {column}={value!r} is outside its support."
                    )
                compatible_values.append((value,))

            for covariates in product(*compatible_values):
                state = (*covariates, outcome)
                column_indices.append(self.support_index_[state])
                row_indices.append(row_position)

        values = np.ones(len(row_indices), dtype=float)
        return sparse.csr_matrix(
            (values, (row_indices, column_indices)),
            shape=(len(data), self.n_states),
        )

    def fit(
        self,
        data: pd.DataFrame,
        *,
        solver: str | None = "MOSEK",
        solver_options: Mapping[str, Any] | None = None,
        verbose: bool = False,
    ) -> "MissingCovariateDRO":
        

        compatibility = self.build_compatibility_matrix(data)
        n_samples = compatibility.shape[0]

        probabilities = cp.Variable(self.n_states, nonneg=True)
        observed_masses = compatibility @ probabilities
        problem = cp.Problem(
            cp.Maximize(cp.sum(cp.log(observed_masses)) / n_samples),
            [cp.sum(probabilities) == 1],
        )
        self._solve_problem(
            problem,
            solver=solver,
            solver_options=solver_options,
            verbose=verbose,
        )
        self._require_optimal(problem, "Maximum-likelihood fit")

        self.compatibility_matrix_ = compatibility
        self.n_samples_ = n_samples
        self.mle_distribution_ = np.asarray(probabilities.value, dtype=float)
        self.maximum_log_likelihood_ = float(problem.value)
        return self

    def likelihood_threshold(self, radius: float) -> float:

        self._require_fitted()
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be a finite nonnegative number.")
        assert self.maximum_log_likelihood_ is not None
        return self.maximum_log_likelihood_ - float(radius)

    def solve_newsvendor(
        self,
        context: Sequence[Hashable],
        *,
        radius: float | None = None,
        log_likelihood_threshold: float | None = None,
        overage_cost: float = 3.0,
        underage_cost: float = 6.0,
        decision_bounds: tuple[float, float] | None = None,
        solver: str | None = "MOSEK",
        solver_options: Mapping[str, Any] | None = None,
        verbose: bool = False,
    ) -> NewsvendorDROResult:
        """Solve the robust conditional newsvendor problem.

        Specify exactly one of ``radius`` or ``log_likelihood_threshold``.
        ``radius`` is usually preferable.
        """

        self._require_fitted()
        if (radius is None) == (log_likelihood_threshold is None):
            raise ValueError(
                "Specify exactly one of radius or log_likelihood_threshold."
            )
        if overage_cost <= 0 or underage_cost <= 0:
            raise ValueError("overage_cost and underage_cost must be positive.")

        context_tuple = tuple(context)
        self._validate_context(context_tuple)
        assert self.maximum_log_likelihood_ is not None
        assert self.compatibility_matrix_ is not None
        assert self.n_samples_ is not None

        if radius is not None:
            threshold = self.likelihood_threshold(radius)
            radius_value = float(radius)
        else:
            threshold = float(log_likelihood_threshold)
            if not np.isfinite(threshold):
                raise ValueError("log_likelihood_threshold must be finite.")
            radius_value = self.maximum_log_likelihood_ - threshold
            if radius_value < -1e-7:
                raise ValueError(
                    "log_likelihood_threshold exceeds the fitted maximum."
                )
            radius_value = max(0.0, radius_value)

        lower, upper = self._decision_bounds(decision_bounds)
        n_samples = self.n_samples_
        compatibility = self.compatibility_matrix_

        context_indicator = np.asarray(
            [state[:-1] == context_tuple for state in self.support_], dtype=float
        )

        order_quantity = cp.Variable(name="order_quantity")
        scale = cp.Variable(pos=True, name="likelihood_scale")
        row_multipliers = cp.Variable(
            n_samples, pos=True, name="row_multipliers"
        )
        normalization_multiplier = cp.Variable(name="normalization_multiplier")
        conditional_value = cp.Variable(name="negative_worst_case_cost")

        losses = (
            overage_cost * cp.pos(order_quantity - self._outcome_vector)
            + underage_cost * cp.pos(self._outcome_vector - order_quantity)
        )
        support_multipliers = (
            compatibility.T @ row_multipliers + normalization_multiplier
        )

        constraints = [
            support_multipliers
            + cp.multiply(
                context_indicator,
                losses + conditional_value,
            )
            <= 0,
            scale * n_samples * (1.0 + threshold)
            - cp.sum(cp.rel_entr(scale, row_multipliers))
            + normalization_multiplier
            >= 0,
            order_quantity >= lower,
            order_quantity <= upper,
        ]
        problem = cp.Problem(cp.Minimize(-conditional_value), constraints)
        self._solve_problem(
            problem,
            solver=solver,
            solver_options=solver_options,
            verbose=verbose,
        )
        self._require_optimal(problem, "Robust newsvendor solve")

        solver_name = getattr(problem.solver_stats, "solver_name", None)
        return NewsvendorDROResult(
            order_quantity=float(order_quantity.value),
            worst_case_cost=float(-conditional_value.value),
            likelihood_threshold=threshold,
            likelihood_radius=radius_value,
            maximum_log_likelihood=self.maximum_log_likelihood_,
            status=problem.status,
            solver=solver_name,
        )

    def _parse_mask(self, raw_mask: Any) -> tuple[int, ...]:
        if isinstance(raw_mask, str):
            try:
                raw_mask = ast.literal_eval(raw_mask)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Invalid missingness mask: {raw_mask!r}") from exc

        try:
            mask = tuple(int(value) for value in raw_mask)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid missingness mask: {raw_mask!r}") from exc
        if len(mask) != self.dimension or any(value not in (0, 1) for value in mask):
            raise ValueError(
                f"Missingness mask must contain {self.dimension} binary entries."
            )
        return mask

    def _validate_columns(self, data: pd.DataFrame) -> None:
        required = {*self.covariate_columns, self.outcome_column, self.mask_column}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

    def _validate_context(self, context: tuple[Hashable, ...]) -> None:
        if len(context) != self.dimension:
            raise ValueError(f"context must contain {self.dimension} values.")
        for j, (value, support) in enumerate(zip(context, self.covariate_supports)):
            if value not in support:
                raise ValueError(f"context[{j}]={value!r} is outside its support.")

    def _decision_bounds(
        self, bounds: tuple[float, float] | None
    ) -> tuple[float, float]:
        if bounds is None:
            values = np.asarray(self.outcome_support, dtype=float)
            return float(values.min()), float(values.max())
        lower, upper = map(float, bounds)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("decision_bounds must be finite with lower < upper.")
        return lower, upper

    def _require_fitted(self) -> None:
        if self.maximum_log_likelihood_ is None:
            raise RuntimeError("Call fit(data) before solving the robust problem.")

    @staticmethod
    def _solve_problem(
        problem: cp.Problem,
        *,
        solver: str | None,
        solver_options: Mapping[str, Any] | None,
        verbose: bool,
    ) -> None:
        options = dict(solver_options or {})
        if solver is None:
            problem.solve(verbose=verbose, **options)
        else:
            problem.solve(solver=solver, verbose=verbose, **options)

    @staticmethod
    def _require_optimal(problem: cp.Problem, label: str) -> None:
        if problem.status not in _OPTIMAL_STATUSES:
            raise RuntimeError(f"{label} failed with status {problem.status!r}.")


def solve_mar_likelihood_newsvendor(
    data: pd.DataFrame,
    context: Sequence[Hashable],
    *,
    covariate_supports: Sequence[Sequence[Hashable]],
    outcome_support: Sequence[float],
    radius: float,
    covariate_columns: Sequence[str] | None = None,
    outcome_column: str = "Y",
    mask_column: str = "missing_mask",
    overage_cost: float = 3.0,
    underage_cost: float = 6.0,
    decision_bounds: tuple[float, float] | None = None,
    solver: str | None = "MOSEK",
    solver_options: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> NewsvendorDROResult:
    """Fit and solve the paper's discrete missing-covariate newsvendor model."""

    model = MissingCovariateDRO(
        covariate_supports,
        outcome_support,
        covariate_columns=covariate_columns,
        outcome_column=outcome_column,
        mask_column=mask_column,
    )
    model.fit(
        data,
        solver=solver,
        solver_options=solver_options,
        verbose=verbose,
    )
    return model.solve_newsvendor(
        context,
        radius=radius,
        overage_cost=overage_cost,
        underage_cost=underage_cost,
        decision_bounds=decision_bounds,
        solver=solver,
        solver_options=solver_options,
        verbose=verbose,
    )


__all__ = [
    "MissingCovariateDRO",
    "NewsvendorDROResult",
    "solve_mar_likelihood_newsvendor",
]
