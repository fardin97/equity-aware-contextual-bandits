"""Unit tests for the extracted action, reward, and policy logic."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_bandits.policy import (
    ACTION_TO_ID,
    ACTIONS,
    ConstrainedLinUCB,
    LinUCB,
    add_environment_columns,
    best_zone,
    build_action_table,
    choose_static_policy,
    fallback_home_if_bad,
    logit,
    make_action_features_for_row,
    sigmoid,
)


class MathTests(unittest.TestCase):
    def test_logit_sigmoid_round_trip(self):
        probabilities = np.array([0.2, 0.5, 0.8])
        np.testing.assert_allclose(sigmoid(logit(probabilities)), probabilities)

    def test_best_zone_is_robust_to_invalid_candidates(self):
        zone, score = best_zone([3, 2, None, "bad"], {2: 0.7, 3: 0.2})
        self.assertEqual(zone, 2.0)
        self.assertEqual(score, 0.7)
        empty_zone, empty_score = best_zone("not-a-sequence", {})
        self.assertTrue(np.isnan(empty_zone))
        self.assertEqual(empty_score, 0.0)


class ActionEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "home_zone_num": 1,
            "visited_zones": [1, 3],
            "work_school_zones": [2],
            "errand_shop_zones": [],
            "transit_compatible_zones": [3],
            "equity_priority": 1,
        }
        self.lookup = {1: 0.5, 2: 0.0, 3: 0.8}

    def test_action_eligibility_and_burden(self):
        actions = make_action_features_for_row(self.context, self.lookup)
        self.assertEqual([row["action"] for row in actions], list(ACTIONS))
        by_action = {row["action"]: row for row in actions}

        self.assertEqual(by_action["HOME_DELIVERY"]["eligible"], 1)
        self.assertEqual(by_action["HOME_DELIVERY"]["burden"], 0.0)
        self.assertAlmostEqual(by_action["HOME_ZONE_ADL"]["burden"], 0.45)
        self.assertEqual(by_action["WORK_SCHOOL_ADL"]["available"], 0)
        self.assertEqual(by_action["WORK_SCHOOL_ADL"]["eligible"], 0)
        self.assertAlmostEqual(by_action["WORK_SCHOOL_ADL"]["burden"], 0.39)
        self.assertEqual(by_action["TRANSIT_ADL"]["eligible"], 1)
        self.assertAlmostEqual(by_action["TRANSIT_ADL"]["burden"], 0.10)
        self.assertEqual(by_action["MIN_BURDEN_VISITED_ADL"]["candidate_zone"], 3.0)
        self.assertAlmostEqual(by_action["MIN_BURDEN_VISITED_ADL"]["burden"], 0.01)

    def test_environment_home_and_ineligible_invariants(self):
        contexts = pd.DataFrame([self.context, self.context])
        actions = build_action_table(contexts, [0.2, 0.8], self.lookup)
        environment, intercept = add_environment_columns(actions, target_mean=0.5)
        self.assertTrue(np.isfinite(intercept))
        home = environment.loc[environment["action"] == "HOME_DELIVERY"]
        self.assertTrue(
            (
                home[
                    [
                        "p_accept",
                        "accept_reward",
                        "deterministic_cost",
                        "expected_reward",
                    ]
                ]
                == 0
            )
            .all()
            .all()
        )
        ineligible = environment.loc[environment["eligible"] == 0]
        self.assertTrue((ineligible["p_accept"] == 0).all())
        self.assertTrue(environment["p_accept"].between(0, 1).all())

    def test_fallback_and_activity_priority(self):
        contexts = pd.DataFrame([self.context])
        actions = build_action_table(contexts, [0.5], self.lookup)
        environment, _ = add_environment_columns(actions, target_mean=0.5)
        work_id = ACTION_TO_ID["WORK_SCHOOL_ADL"]
        self.assertEqual(
            fallback_home_if_bad(environment, work_id), ACTION_TO_ID["HOME_DELIVERY"]
        )
        choice = choose_static_policy(environment, "ACTIVITY_PRIORITY_RULE")
        self.assertEqual(choice.tolist(), [ACTION_TO_ID["TRANSIT_ADL"]])

    def test_action_table_rejects_mismatched_probabilities(self):
        with self.assertRaises(ValueError):
            build_action_table(pd.DataFrame([self.context]), [], self.lookup)


class BanditTests(unittest.TestCase):
    def test_linear_update(self):
        policy = LinUCB(n_actions=2, dimension=2, seed=1)
        policy.update(0, np.array([1.0, 2.0]), 3.0)
        np.testing.assert_allclose(policy.a[0], [[2.0, 2.0], [2.0, 5.0]])
        np.testing.assert_allclose(policy.b[0], [3.0, 6.0])

    def test_constrained_policy_masks_high_burden_for_equity_context(self):
        policy = ConstrainedLinUCB(
            n_actions=len(ACTIONS),
            dimension=2,
            equity_max_burden=0.45,
            burden_penalty=0.0,
            seed=1,
        )
        contexts = np.ones((len(ACTIONS), 2))
        feasible = np.zeros(len(ACTIONS), dtype=bool)
        feasible[[ACTION_TO_ID["HOME_DELIVERY"], ACTION_TO_ID["HOME_ZONE_ADL"]]] = True
        burdens = np.zeros(len(ACTIONS))
        burdens[ACTION_TO_ID["HOME_ZONE_ADL"]] = 0.60
        selected = policy.select(contexts, feasible, burdens, equity_priority=1)
        self.assertEqual(selected, ACTION_TO_ID["HOME_DELIVERY"])


if __name__ == "__main__":
    unittest.main()
