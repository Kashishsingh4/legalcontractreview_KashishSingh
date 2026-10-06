"""Stage 5 (Kashish): RL-based Uncertainty Verification Agent.

A contextual bandit — a real, lightweight reinforcement-learning formulation,
not a full offline-trained deep-RL policy (no labeled dataset of correct
accept/flag/reanalyze decisions exists to pretrain on). Per docs/INTERFACES.md's
own RL formulation: state = [confidence, max retrieval relevance, agent
agreement, risk_level], actions = accept | flag | reanalyze.

Day-one behavior is initialized to exactly reproduce the old heuristic that
lived in stubs.uncertainty_agent, so wiring this in is not a regression. The
policy then improves via `.update()` whenever a human reviewer confirms or
corrects a decision (features/reward supplied by the caller — no UI wiring
for that yet, see uncertainty_bandit.py's __main__ demo for how it's driven).
"""
from __future__ import annotations

import json
import os
import random
from typing import Any

from interfaces import UncertaintyDecision

ACTIONS = ("accept", "flag", "reanalyze")

FEATURE_NAMES = (
    "bias",
    "confidence",
    "relevance",
    "is_attention_required",
    "is_insufficient_evidence",
    "is_no_risk_found",
    "agreement",
    "low_confidence_flag",
    "low_relevance_flag",
)

# Priority-order weight separation: each threshold dominates every signal
# below it, exactly reproducing the old if/elif heuristic's decision order.
INIT_WEIGHTS: dict[str, dict[str, float]] = {
    "reanalyze": {
        "low_confidence_flag": 1000.0,
        "low_relevance_flag": 100.0,
        "is_insufficient_evidence": 10.0,
    },
    "flag": {
        "is_attention_required": 10.0,
    },
    "accept": {
        "bias": 1.0,
    },
}

EPSILON = 0.10
LEARNING_RATE = 0.05

LOW_CONFIDENCE_THRESHOLD = 0.60
# Calibrated, not guessed: mean-pooled LegalBERT embeddings sit in a narrow
# cone, so cosine relevance is high for almost any text. On 60 unseen LEDGAR
# test clauses the best relevance never fell below 0.831; on 15 non-legal
# texts (recipes, sports, gibberish) it never rose above 0.776. 0.80 sits in
# that gap. The previous 0.45 was below every score observed, so the
# weak-evidence signal could never fire.
LOW_RELEVANCE_THRESHOLD = 0.80

WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bandit_weights.json")


def _zero_weights() -> dict[str, dict[str, float]]:
    return {action: {name: 0.0 for name in FEATURE_NAMES} for action in ACTIONS}


def _full_weights(overrides: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    weights = _zero_weights()
    for action, feats in overrides.items():
        weights[action].update(feats)
    return weights


class UncertaintyBanditAgent:
    """Per-action linear scorer with epsilon-greedy exploration."""

    def __init__(self, weights_file: str = WEIGHTS_FILE):
        self.weights_file = weights_file
        self.weights: dict[str, dict[str, float]] = _full_weights(INIT_WEIGHTS)
        self.visit_counts: dict[str, int] = {action: 0 for action in ACTIONS}
        self.epsilon = EPSILON
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.weights_file):
            return
        try:
            with open(self.weights_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            weights = _zero_weights()
            for action in ACTIONS:
                weights[action].update(data.get("weights", {}).get(action, {}))
            self.weights = weights
            self.visit_counts = {
                action: int(data.get("visit_counts", {}).get(action, 0)) for action in ACTIONS
            }
            self.epsilon = float(data.get("epsilon", EPSILON))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Corrupt state file: fall back to the safe, heuristic-matching init.
            self.weights = _full_weights(INIT_WEIGHTS)
            self.visit_counts = {action: 0 for action in ACTIONS}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.weights_file), exist_ok=True)
        with open(self.weights_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_names": list(FEATURE_NAMES),
                    "weights": self.weights,
                    "visit_counts": self.visit_counts,
                    "epsilon": self.epsilon,
                    "version": 1,
                },
                f,
                indent=2,
            )

    @staticmethod
    def build_features(features: dict[str, Any]) -> dict[str, float]:
        confidence = float(features.get("confidence", 0.0))
        relevance = float(features.get("relevance", 0.0))
        assessment = features.get("risk_assessment", "insufficient_evidence")
        agreement = float(features.get("agreement", 1.0))
        return {
            "bias": 1.0,
            "confidence": confidence,
            "relevance": relevance,
            "is_attention_required": 1.0 if assessment == "attention_required" else 0.0,
            "is_insufficient_evidence": 1.0 if assessment == "insufficient_evidence" else 0.0,
            "is_no_risk_found": 1.0 if assessment == "no_configured_risk_found" else 0.0,
            "agreement": agreement,
            "low_confidence_flag": 1.0 if confidence < LOW_CONFIDENCE_THRESHOLD else 0.0,
            "low_relevance_flag": 1.0 if relevance < LOW_RELEVANCE_THRESHOLD else 0.0,
        }

    def _score(self, action: str, x: dict[str, float]) -> float:
        w = self.weights[action]
        return sum(w[name] * x[name] for name in FEATURE_NAMES)

    def decide(self, clause_id: str, features: dict[str, Any]) -> UncertaintyDecision:
        x = self.build_features(features)
        scores = {action: self._score(action, x) for action in ACTIONS}

        greedy = max(scores, key=scores.get)
        explore = random.random() < self.epsilon
        if explore:
            # Safe exploration: try a different action, but never explore
            # *down* to "accept" when the evidence-driven choice is to send
            # the clause to a human — a random accept on a low-confidence or
            # risky clause is exactly the failure this agent exists to stop.
            candidates = [a for a in ACTIONS if a != greedy and (greedy == "accept" or a != "accept")]
            action = random.choice(candidates)
        else:
            action = greedy

        ranked = sorted(scores.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
        # An exploratory pick isn't evidence-driven, so its policy_confidence
        # must not borrow the margin of whichever action the scores actually
        # favored — report low, honest confidence instead.
        policy_confidence = 0.5 if explore else max(0.5, min(0.95, 0.5 + margin / 20.0))

        reasons = {
            "reanalyze": "Bandit policy selected reanalyze: low classification confidence, weak retrieval evidence, or insufficient supporting evidence.",
            "flag": "Bandit policy selected flag: a configured, source-backed risk factor was matched; human legal review is recommended.",
            "accept": "Bandit policy selected accept: adequate confidence and evidence, no unresolved risk factor.",
        }
        reason = reasons[action]
        if explore:
            reason = (
                f"Bandit policy chose {action} via exploratory random sampling "
                f"(epsilon={self.epsilon}), not because the evidence pointed to it — "
                "this is the RL agent occasionally testing alternatives to keep learning."
            )

        self.visit_counts[action] += 1
        self._save()

        return UncertaintyDecision(
            clause_id=clause_id,
            action=action,
            policy_confidence=round(policy_confidence, 4),
            reason=reason,
        )

    def update(self, clause_id: str, features: dict[str, Any], action_taken: str, reward: float) -> None:
        """Online Widrow-Hoff update on the taken action's weights only
        (bandit feedback — no counterfactual for actions not taken)."""
        if action_taken not in ACTIONS:
            raise ValueError(f"Unknown action '{action_taken}'; must be one of {ACTIONS}")

        x = self.build_features(features)
        w = self.weights[action_taken]
        visits = self.visit_counts[action_taken]
        lr = LEARNING_RATE / (1.0 + visits)

        prediction = sum(w[name] * x[name] for name in FEATURE_NAMES)
        error = reward - prediction
        for name in FEATURE_NAMES:
            w[name] += lr * error * x[name]

        self._save()


_agent: UncertaintyBanditAgent | None = None


def _get_agent() -> UncertaintyBanditAgent:
    global _agent
    if _agent is None:
        _agent = UncertaintyBanditAgent()
    return _agent


def uncertainty_agent(features: dict[str, Any]) -> UncertaintyDecision:
    """Drop-in replacement for the old stubs.uncertainty_agent heuristic.
    features = {clause_id, confidence, relevance, risk_assessment, agreement?}"""
    return _get_agent().decide(features["clause_id"], features)


def record_feedback(clause_id: str, features: dict[str, Any], action_taken: str, reward: float) -> None:
    """Public entry point for a human reviewer's confirm/correct signal."""
    _get_agent().update(clause_id, features, action_taken, reward)


if __name__ == "__main__":
    # Standalone demo / verification (Tier 1 — no model/KB artifacts needed).
    import itertools

    def old_heuristic(features: dict[str, Any]) -> str:
        confidence = features["confidence"]
        relevance = features["relevance"]
        assessment = features["risk_assessment"]
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            return "reanalyze"
        if relevance < LOW_RELEVANCE_THRESHOLD:
            return "reanalyze"
        if assessment == "attention_required":
            return "flag"
        if assessment == "insufficient_evidence":
            return "reanalyze"
        return "accept"

    # 1. Cold-start parity check against the old heuristic.
    fresh = UncertaintyBanditAgent(weights_file=os.path.join("data", "_bandit_demo_coldstart.json"))
    if os.path.isfile(fresh.weights_file):
        os.remove(fresh.weights_file)
        fresh = UncertaintyBanditAgent(weights_file=fresh.weights_file)
    fresh.epsilon = 0.0  # deterministic for the parity check

    grid = list(itertools.product(
        [0.3, 0.5, 0.55, 0.6, 0.7, 0.9],
        [0.4, 0.7, 0.79, 0.8, 0.85, 0.95],
        ["attention_required", "insufficient_evidence", "no_configured_risk_found"],
    ))
    mismatches = []
    for confidence, relevance, assessment in grid:
        feats = {"clause_id": "c_test", "confidence": confidence, "relevance": relevance, "risk_assessment": assessment}
        bandit_action = fresh.decide("c_test", feats).action
        heuristic_action = old_heuristic(feats)
        if bandit_action != heuristic_action:
            mismatches.append((feats, bandit_action, heuristic_action))

    print(f"Cold-start parity check: {len(grid) - len(mismatches)}/{len(grid)} match the old heuristic.")
    if mismatches:
        for feats, bandit_action, heuristic_action in mismatches[:5]:
            print(f"  MISMATCH {feats} -> bandit={bandit_action} heuristic={heuristic_action}")
        raise SystemExit("FAILED — cold-start bandit does not reproduce the old heuristic.")
    print("OK — cold-start bandit exactly reproduces stubs.uncertainty_agent's old behavior.\n")
    os.remove(fresh.weights_file)

    # 2. Online update demo: push a borderline case from accept toward flag.
    demo_path = os.path.join("data", "_bandit_demo_learning.json")
    if os.path.isfile(demo_path):
        os.remove(demo_path)
    learner = UncertaintyBanditAgent(weights_file=demo_path)
    learner.epsilon = 0.0

    borderline = {"clause_id": "c_borderline", "confidence": 0.85, "relevance": 0.9, "risk_assessment": "no_configured_risk_found"}
    before = learner.decide("c_borderline", borderline)
    print(f"Before feedback: action={before.action}")

    for _ in range(40):
        learner.update("c_borderline", borderline, action_taken="accept", reward=-1.0)
        learner.update("c_borderline", borderline, action_taken="flag", reward=1.0)

    after = learner.decide("c_borderline", borderline)
    print(f"After 40 corrective feedback rounds: action={after.action}")
    assert after.action == "flag", "Expected online learning to shift the decision toward 'flag'."
    print("OK — .update() measurably shifts future decisions.\n")

    # 3. Persistence round-trip: reload from disk, confirm the shift stuck.
    reloaded = UncertaintyBanditAgent(weights_file=demo_path)
    reloaded.epsilon = 0.0
    after_reload = reloaded.decide("c_borderline", borderline)
    print(f"After reload from {demo_path}: action={after_reload.action}")
    assert after_reload.action == "flag", "Expected the learned shift to persist across reload."
    print("OK — learned weights persist correctly across reload.\n")

    os.remove(demo_path)

    # 4. Safe exploration: even when *always* exploring, a low-confidence or
    # risky clause must never be explored down to "accept".
    explore_path = os.path.join("data", "_bandit_demo_explore.json")
    explorer = UncertaintyBanditAgent(weights_file=explore_path)
    explorer.epsilon = 1.0
    risky_cases = [
        {"clause_id": "c_lowconf", "confidence": 0.35, "relevance": 0.9, "risk_assessment": "no_configured_risk_found"},
        {"clause_id": "c_risky", "confidence": 0.9, "relevance": 0.9, "risk_assessment": "attention_required"},
        {"clause_id": "c_weak", "confidence": 0.9, "relevance": 0.7, "risk_assessment": "no_configured_risk_found"},
    ]
    for feats in risky_cases:
        picks = {explorer.decide(feats["clause_id"], feats).action for _ in range(200)}
        print(f"Always-exploring picks for {feats['clause_id']}: {sorted(picks)}")
        assert "accept" not in picks, f"Safe exploration violated: explored to accept on {feats['clause_id']}"
    os.remove(explore_path)
    print("OK — exploration never downgrades a clause that needs human review to 'accept'.\n")

    print("ALL CHECKS PASSED.")
