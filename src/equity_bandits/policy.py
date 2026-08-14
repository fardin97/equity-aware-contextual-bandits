"""Pure policy and reward components extracted from the research notebook.

The functions in this module implement the notebook's transparent simulation
assumptions. They do not estimate causal effects or encode a deployment policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

ACTIONS = (
    "HOME_DELIVERY",
    "HOME_ZONE_ADL",
    "WORK_SCHOOL_ADL",
    "ERRAND_SHOP_ADL",
    "TRANSIT_ADL",
    "MIN_BURDEN_VISITED_ADL",
)
ACTION_TO_ID = {action: index for index, action in enumerate(ACTIONS)}

ACTION_BASE_BURDEN = {
    "HOME_DELIVERY": 0.00,
    "HOME_ZONE_ADL": 0.32,
    "WORK_SCHOOL_ADL": 0.14,
    "ERRAND_SHOP_ADL": 0.10,
    "TRANSIT_ADL": 0.18,
    "MIN_BURDEN_VISITED_ADL": 0.08,
}

ACTION_SERVICE_VALUE = {
    "HOME_DELIVERY": 0.00,
    "HOME_ZONE_ADL": 0.12,
    "WORK_SCHOOL_ADL": 0.22,
    "ERRAND_SHOP_ADL": 0.24,
    "TRANSIT_ADL": 0.20,
    "MIN_BURDEN_VISITED_ADL": 0.26,
}

ENV_PARAMS_DEFAULT = {
    "beta_opp": 0.90,
    "beta_burden": 0.85,
    "beta_chain": 0.55,
    "beta_available": 0.45,
    "beta_extra_trip": 0.45,
    "success_value": 1.00,
    "service_value_weight": 0.60,
    "opportunity_reward": 0.35,
    "consolidation_reward": 0.20,
    "behavior_targeting_reward": 0.15,
    "burden_penalty": 0.22,
    "equity_bonus": 0.18,
    "equity_burden_penalty": 0.10,
    "recommendation_cost": 0.015,
    "ineligible_penalty": 0.20,
}


def sigmoid(values: object) -> np.ndarray:
    """Numerically stable logistic transform."""
    array = np.asarray(values, dtype=float)
    clipped = np.clip(array, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logit(probabilities: object, epsilon: float = 1e-6) -> np.ndarray:
    """Log-odds transform with finite clipping at both endpoints."""
    array = np.asarray(probabilities, dtype=float)
    clipped = np.clip(array, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def _zone_sequence(zones: object) -> Sequence[object]:
    if isinstance(zones, (list, tuple, np.ndarray, pd.Series)):
        return zones
    return ()


def best_zone(
    zones: object, zone_score_lookup: Mapping[int, float]
) -> tuple[float, float]:
    """Return the candidate zone with the largest opportunity score."""
    valid: list[tuple[int, float]] = []
    for zone in _zone_sequence(zones):
        if pd.isna(zone):
            continue
        try:
            zone_id = int(zone)
            score = float(zone_score_lookup.get(zone_id, 0.0))
        except (TypeError, ValueError):
            continue
        valid.append((zone_id, score))
    if not valid:
        return np.nan, 0.0
    zone_id, score = max(valid, key=lambda item: item[1])
    return float(zone_id), score


def make_action_features_for_row(
    row: Mapping[str, object] | pd.Series,
    zone_score_lookup: Mapping[int, float],
) -> list[dict[str, object]]:
    """Expand one context into the six candidate recommendation actions."""
    home_zone = row.get("home_zone_num", np.nan)
    visited = _zone_sequence(row.get("visited_zones", ()))
    work_school = _zone_sequence(row.get("work_school_zones", ()))
    errand = _zone_sequence(row.get("errand_shop_zones", ()))
    transit = _zone_sequence(row.get("transit_compatible_zones", ()))

    actions: list[dict[str, object]] = [
        {
            "action": "HOME_DELIVERY",
            "action_id": ACTION_TO_ID["HOME_DELIVERY"],
            "candidate_zone": home_zone,
            "opp_score": 0.0,
            "available": 1,
            "eligible": 1,
            "chain_compatible": 1,
            "requires_extra_trip": 0,
            "burden": 0.0,
            "is_adl_action": 0,
            "service_value": ACTION_SERVICE_VALUE["HOME_DELIVERY"],
        }
    ]

    def add(
        action: str,
        candidate_zones: Sequence[object],
        *,
        extra_trip: bool,
        chain_compatible: bool,
    ) -> None:
        zone, opportunity = best_zone(candidate_zones, zone_score_lookup)
        available = int(opportunity > 0.0)
        eligible = int(bool(available) and chain_compatible)
        burden = ACTION_BASE_BURDEN[action]
        burden += 0.25 * (1 - available)
        burden += 0.18 * int(extra_trip)
        burden -= 0.10 * opportunity
        actions.append(
            {
                "action": action,
                "action_id": ACTION_TO_ID[action],
                "candidate_zone": zone,
                "opp_score": opportunity,
                "available": available,
                "eligible": eligible,
                "chain_compatible": int(chain_compatible),
                "requires_extra_trip": int(extra_trip),
                "burden": float(np.clip(burden, 0.01, 2.0)),
                "is_adl_action": 1,
                "service_value": ACTION_SERVICE_VALUE[action],
            }
        )

    home_zones = [int(home_zone)] if pd.notna(home_zone) else []
    add("HOME_ZONE_ADL", home_zones, extra_trip=True, chain_compatible=True)
    add(
        "WORK_SCHOOL_ADL",
        work_school,
        extra_trip=False,
        chain_compatible=bool(len(work_school)),
    )
    add(
        "ERRAND_SHOP_ADL",
        errand,
        extra_trip=False,
        chain_compatible=bool(len(errand)),
    )
    add(
        "TRANSIT_ADL",
        transit,
        extra_trip=False,
        chain_compatible=bool(len(transit)),
    )
    add(
        "MIN_BURDEN_VISITED_ADL",
        visited,
        extra_trip=False,
        chain_compatible=bool(len(visited)),
    )
    return actions


def build_action_table(
    contexts: pd.DataFrame,
    base_probabilities: Sequence[float],
    zone_score_lookup: Mapping[int, float],
) -> pd.DataFrame:
    """Expand a context table into one row per observation and action."""
    probabilities = np.asarray(base_probabilities, dtype=float)
    if len(contexts) != len(probabilities):
        raise ValueError("contexts and base_probabilities must have equal length")
    rows: list[dict[str, object]] = []
    for obs_id, (_, context) in enumerate(contexts.reset_index(drop=True).iterrows()):
        for action in make_action_features_for_row(context, zone_score_lookup):
            row = dict(action)
            row["obs_id"] = obs_id
            row["p_base"] = probabilities[obs_id]
            row["equity_priority"] = int(context.get("equity_priority", 0))
            rows.append(row)
    return pd.DataFrame(rows)


def calibrate_action_intercept(
    action_table: pd.DataFrame,
    params: Mapping[str, float],
    target_mean: float,
    baseline_action: str = "MIN_BURDEN_VISITED_ADL",
) -> float:
    """Match pre-eligibility mean simulated acceptance for a baseline action."""
    if not 0.0 < target_mean < 1.0:
        raise ValueError("target_mean must be strictly between zero and one")
    subset = action_table.loc[action_table["action"] == baseline_action]
    if subset.empty:
        raise ValueError(f"baseline action is absent: {baseline_action}")
    utility_without_intercept = (
        logit(subset["p_base"])
        + params["beta_opp"] * subset["opp_score"].to_numpy()
        - params["beta_burden"] * subset["burden"].to_numpy()
        + params["beta_chain"] * subset["chain_compatible"].to_numpy()
        + params["beta_available"] * subset["available"].to_numpy()
        - params["beta_extra_trip"] * subset["requires_extra_trip"].to_numpy()
    )
    lower, upper = -10.0, 10.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if float(sigmoid(utility_without_intercept + midpoint).mean()) < target_mean:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def add_environment_columns(
    action_table: pd.DataFrame,
    target_mean: float,
    params: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, float]:
    """Apply the notebook's assumed action-response and reward environment."""
    configured = dict(ENV_PARAMS_DEFAULT)
    if params is not None:
        configured.update(params)
    table = action_table.copy()
    intercept = calibrate_action_intercept(table, configured, target_mean)
    table["env_intercept"] = intercept

    utility = (
        logit(table["p_base"])
        + intercept
        + configured["beta_opp"] * table["opp_score"].to_numpy()
        - configured["beta_burden"] * table["burden"].to_numpy()
        + configured["beta_chain"] * table["chain_compatible"].to_numpy()
        + configured["beta_available"] * table["available"].to_numpy()
        - configured["beta_extra_trip"] * table["requires_extra_trip"].to_numpy()
    )
    table["p_accept"] = np.clip(
        sigmoid(utility)
        * table["is_adl_action"].to_numpy()
        * table["eligible"].to_numpy(),
        0.0,
        1.0,
    )
    table["accept_reward"] = (
        configured["success_value"]
        + configured["service_value_weight"] * table["service_value"]
        + configured["opportunity_reward"] * table["opp_score"]
        + configured["consolidation_reward"]
        * table["opp_score"]
        * table["chain_compatible"]
        + configured["behavior_targeting_reward"] * table["p_base"]
        + configured["equity_bonus"] * table["equity_priority"]
    )
    table["deterministic_cost"] = (
        -configured["burden_penalty"] * table["burden"] * table["is_adl_action"]
        - configured["equity_burden_penalty"]
        * table["burden"]
        * table["equity_priority"]
        * table["is_adl_action"]
        - configured["recommendation_cost"] * table["is_adl_action"]
        - configured["ineligible_penalty"]
        * ((table["is_adl_action"] == 1) & (table["eligible"] == 0)).astype(int)
    )
    table["expected_reward"] = (
        table["p_accept"] * table["accept_reward"] + table["deterministic_cost"]
    )
    home = table["action"] == "HOME_DELIVERY"
    table.loc[
        home,
        ["p_accept", "accept_reward", "deterministic_cost", "expected_reward"],
    ] = 0.0
    return table, intercept


def fallback_home_if_bad(action_rows: pd.DataFrame, selected_action_id: int) -> int:
    """Return home when a selected action is infeasible or has negative reward."""
    selected = action_rows.loc[action_rows["action_id"] == selected_action_id]
    if len(selected) != 1:
        raise ValueError("selected action must occur exactly once")
    row = selected.iloc[0]
    if int(row["eligible"]) < 1 or float(row["expected_reward"]) < 0.0:
        return ACTION_TO_ID["HOME_DELIVERY"]
    return int(selected_action_id)


def _validate_action_group(group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.sort_values("action_id")
    expected = list(range(len(ACTIONS)))
    if ordered["action_id"].astype(int).tolist() != expected:
        raise ValueError("each observation must contain each action exactly once")
    return ordered


def choose_static_policy(
    action_environment: pd.DataFrame,
    policy: str,
    *,
    propensity_threshold: float | None = None,
) -> np.ndarray:
    """Choose actions for one of the notebook's static baseline policies."""
    choices: list[int] = []
    forced_actions = set(ACTIONS) - {"HOME_DELIVERY"}
    if policy == "HIGH_PROPENSITY_MIN_BURDEN" and propensity_threshold is None:
        home_rows = action_environment.loc[
            action_environment["action"] == "HOME_DELIVERY", "p_base"
        ]
        propensity_threshold = float(home_rows.quantile(0.75))
    for _, raw_group in action_environment.groupby("obs_id", sort=True):
        group = _validate_action_group(raw_group)
        eligible = group.loc[group["eligible"] == 1]
        home_id = ACTION_TO_ID["HOME_DELIVERY"]

        if policy == "HOME_ONLY":
            selected = home_id
        elif policy in forced_actions:
            selected = ACTION_TO_ID[policy]
        elif policy == "HIGHEST_OPPORTUNITY":
            candidates = group.loc[group["action"] != "HOME_DELIVERY"]
            selected = int(
                candidates.iloc[candidates["opp_score"].to_numpy().argmax()][
                    "action_id"
                ]
            )
        elif policy == "HIGHEST_PREDICTED_ACCEPTANCE":
            candidates = group.loc[group["action"] != "HOME_DELIVERY"]
            selected = int(
                candidates.iloc[candidates["p_accept"].to_numpy().argmax()]["action_id"]
            )
        elif policy == "LOWEST_BURDEN_ELIGIBLE":
            adl = eligible.loc[eligible["is_adl_action"] == 1]
            selected = (
                home_id
                if adl.empty
                else int(adl.iloc[adl["burden"].to_numpy().argmin()]["action_id"])
            )
        elif policy == "HIGH_PROPENSITY_MIN_BURDEN":
            assert propensity_threshold is not None
            if float(group["p_base"].iloc[0]) < propensity_threshold:
                selected = home_id
            else:
                adl = eligible.loc[eligible["is_adl_action"] == 1]
                selected = (
                    home_id
                    if adl.empty
                    else int(adl.iloc[adl["burden"].to_numpy().argmin()]["action_id"])
                )
        elif policy == "EQUITY_PRIORITY_MIN_BURDEN":
            if int(group["equity_priority"].iloc[0]) != 1:
                selected = home_id
            else:
                adl = eligible.loc[eligible["is_adl_action"] == 1]
                selected = (
                    home_id
                    if adl.empty
                    else int(adl.iloc[adl["burden"].to_numpy().argmin()]["action_id"])
                )
        elif policy == "ACTIVITY_PRIORITY_RULE":
            selected = home_id
            for action in (
                "ERRAND_SHOP_ADL",
                "WORK_SCHOOL_ADL",
                "TRANSIT_ADL",
                "MIN_BURDEN_VISITED_ADL",
                "HOME_ZONE_ADL",
            ):
                row = group.loc[group["action"] == action].iloc[0]
                if int(row["eligible"]) == 1 and float(row["expected_reward"]) >= 0.0:
                    selected = int(row["action_id"])
                    break
        elif policy == "SCORE_HEURISTIC":
            scored = group.copy()
            scored["score"] = (
                0.50 * scored["p_accept"]
                + 0.25 * scored["opp_score"]
                + 0.15 * scored["chain_compatible"]
                + 0.10 * scored["equity_priority"]
                - 0.25 * scored["burden"]
            )
            scored.loc[scored["action"] == "HOME_DELIVERY", "score"] = 0.0
            scored.loc[scored["eligible"] != 1, "score"] = -np.inf
            winner = scored.iloc[scored["score"].to_numpy().argmax()]
            selected = (
                int(winner["action_id"]) if float(winner["score"]) > 0.0 else home_id
            )
        elif policy == "ORACLE_EXPECTED_REWARD":
            selected = int(group["expected_reward"].to_numpy().argmax())
        else:
            raise ValueError(f"unknown policy: {policy}")
        choices.append(fallback_home_if_bad(group, selected))
    return np.asarray(choices, dtype=int)


class LinearPolicy:
    """Per-action ridge-linear reward estimator."""

    name = "LinearPolicy"

    def __init__(self, n_actions: int, dimension: int, seed: int = 42) -> None:
        self.n_actions = n_actions
        self.dimension = dimension
        self.rng = np.random.default_rng(seed)
        self.a = np.repeat(np.eye(dimension)[None, :, :], n_actions, axis=0)
        self.b = np.zeros((n_actions, dimension), dtype=float)

    def update(self, action: int, context: np.ndarray, reward: float) -> None:
        vector = np.asarray(context, dtype=float)
        self.a[action] += np.outer(vector, vector)
        self.b[action] += float(reward) * vector

    def _mean_and_uncertainty(
        self, action: int, context: np.ndarray
    ) -> tuple[float, float]:
        inverse = np.linalg.inv(self.a[action])
        theta = inverse @ self.b[action]
        vector = np.asarray(context, dtype=float)
        return float(theta @ vector), float(np.sqrt(vector @ inverse @ vector))

    @staticmethod
    def _feasible_indices(feasible_mask: np.ndarray) -> np.ndarray:
        indices = np.flatnonzero(np.asarray(feasible_mask, dtype=bool))
        if not len(indices):
            raise ValueError("at least one action must be feasible")
        return indices


class EpsilonGreedyLinear(LinearPolicy):
    def __init__(
        self, n_actions: int, dimension: int, epsilon: float = 0.10, seed: int = 42
    ) -> None:
        super().__init__(n_actions, dimension, seed)
        self.epsilon = epsilon
        self.name = f"EpsilonGreedyLinear_eps{epsilon}"

    def select(self, contexts: np.ndarray, feasible_mask: np.ndarray) -> int:
        feasible = self._feasible_indices(feasible_mask)
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(feasible))
        means = [self._mean_and_uncertainty(a, contexts[a])[0] for a in feasible]
        return int(feasible[int(np.argmax(means))])


class LinUCB(LinearPolicy):
    def __init__(
        self, n_actions: int, dimension: int, alpha: float = 0.75, seed: int = 42
    ) -> None:
        super().__init__(n_actions, dimension, seed)
        self.alpha = alpha
        self.name = f"LinUCB_alpha{alpha}"

    def select(self, contexts: np.ndarray, feasible_mask: np.ndarray) -> int:
        feasible = self._feasible_indices(feasible_mask)
        scores = []
        for action in feasible:
            mean, uncertainty = self._mean_and_uncertainty(action, contexts[action])
            scores.append(mean + self.alpha * uncertainty)
        return int(feasible[int(np.argmax(scores))])


class LinearThompsonSampling(LinearPolicy):
    """Linear Thompson sampling using the notebook's diagonal approximation."""

    def __init__(
        self, n_actions: int, dimension: int, v: float = 0.30, seed: int = 42
    ) -> None:
        super().__init__(n_actions, dimension, seed)
        self.v = v
        self.name = f"LinearTS_v{v}"

    def select(self, contexts: np.ndarray, feasible_mask: np.ndarray) -> int:
        feasible = self._feasible_indices(feasible_mask)
        scores = []
        for action in feasible:
            inverse = np.linalg.inv(self.a[action])
            mean = inverse @ self.b[action]
            sampled = self.rng.normal(mean, self.v * np.sqrt(np.diag(inverse)))
            scores.append(float(sampled @ contexts[action]))
        return int(feasible[int(np.argmax(scores))])


class ConstrainedLinUCB(LinUCB):
    def __init__(
        self,
        n_actions: int,
        dimension: int,
        alpha: float = 0.75,
        equity_max_burden: float = 0.45,
        burden_penalty: float = 0.25,
        seed: int = 42,
    ) -> None:
        super().__init__(n_actions, dimension, alpha, seed)
        self.equity_max_burden = equity_max_burden
        self.burden_penalty = burden_penalty
        self.name = f"ConstrainedLinUCB_b{equity_max_burden}"

    def select(
        self,
        contexts: np.ndarray,
        feasible_mask: np.ndarray,
        burdens: np.ndarray,
        equity_priority: int,
    ) -> int:
        constrained = np.asarray(feasible_mask, dtype=bool).copy()
        if equity_priority:
            high_burden = np.asarray(burdens, dtype=float) > self.equity_max_burden
            high_burden[ACTION_TO_ID["HOME_DELIVERY"]] = False
            constrained[high_burden] = False
        feasible = self._feasible_indices(constrained)
        scores = []
        for action in feasible:
            mean, uncertainty = self._mean_and_uncertainty(action, contexts[action])
            scores.append(
                mean
                + self.alpha * uncertainty
                - self.burden_penalty * float(burdens[action])
            )
        return int(feasible[int(np.argmax(scores))])
