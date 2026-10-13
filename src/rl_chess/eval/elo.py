"""Bradley-Terry Maximum Likelihood Estimation for Elo rating with confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MatchResult:
    opponent_name: str
    opponent_elo: float
    wins: int
    draws: int
    losses: int

    @property
    def total_games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws


@dataclass
class EloEstimate:
    rating: float
    ci_low: float
    ci_high: float
    std_err: float
    total_games: int
    total_score: float


def expected_score(agent_elo: float, opponent_elo: float) -> float:
    """Logistic expected score of agent against opponent."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - agent_elo) / 400.0))


def fit_elo_bradley_terry(
    matches: List[MatchResult],
    min_elo: float = 0.0,
    max_elo: float = 3500.0,
    tol: float = 1e-4,
) -> EloEstimate:
    """Fit Bradley-Terry MLE Elo with opponent ratings fixed.
    
    Solves sum(S_i) == sum(N_i * E_i(R_A)) using binary search.
    """
    total_score = sum(m.score for m in matches)
    total_games = sum(m.total_games for m in matches)

    if total_games == 0:
        return EloEstimate(0.0, 0.0, 0.0, 0.0, 0, 0.0)

    # Edge cases: 0% or 100% win rate
    if total_score <= 0.0:
        min_opp = min(m.opponent_elo for m in matches)
        r = max(0.0, min_opp - 400.0)
        return EloEstimate(r, max(0.0, r - 200.0), r + 200.0, 100.0, total_games, 0.0)
    if total_score >= total_games:
        max_opp = max(m.opponent_elo for m in matches)
        r = max_opp + 400.0
        return EloEstimate(r, r - 200.0, r + 200.0, 100.0, total_games, total_score)

    low = min_elo
    high = max_elo

    for _ in range(100):
        mid = (low + high) / 2.0
        expected = sum(m.total_games * expected_score(mid, m.opponent_elo) for m in matches)

        if abs(expected - total_score) < tol:
            break
        if expected < total_score:
            low = mid
        else:
            high = mid

    fitted_elo = mid

    # Compute variance via Fisher information
    ln10_div_400 = math.log(10.0) / 400.0
    fisher_info = 0.0
    for m in matches:
        e = expected_score(fitted_elo, m.opponent_elo)
        fisher_info += m.total_games * e * (1.0 - e) * (ln10_div_400 ** 2)

    if fisher_info > 0:
        std_err = 1.0 / math.sqrt(fisher_info)
    else:
        std_err = 100.0

    ci_low = max(0.0, fitted_elo - 1.96 * std_err)
    ci_high = fitted_elo + 1.96 * std_err

    return EloEstimate(
        rating=fitted_elo,
        ci_low=ci_low,
        ci_high=ci_high,
        std_err=std_err,
        total_games=total_games,
        total_score=total_score,
    )
