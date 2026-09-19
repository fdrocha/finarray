"""A minimal linear trading-cost model.

Costs are expressed the way US equity venues quote them: mils (tenths of a cent)
per share, and bips (hundredths of a percent) of traded notional. Both are
converted to plain fractions at the point of use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fees:
    """Per-share and per-dollar trading costs.

    ``sell_only`` marks a fee that is charged on sells but not buys (several
    regulatory fees work this way). The backtest engine does not track side, so
    it halves such a fee -- see `as_two_sided` -- which is right on average for
    a strategy that round-trips everything it trades.
    """

    per_share_mils: float = 0.0
    per_dollar_bips: float = 0.0
    sell_only: bool = False

    @classmethod
    def mils(cls, mils: float, sell_only: bool = False) -> Fees:
        """A pure per-share fee, in tenths of a cent per share."""
        return cls(per_share_mils=mils, sell_only=sell_only)

    @classmethod
    def bips(cls, bips: float, sell_only: bool = False) -> Fees:
        """A pure per-notional fee, in hundredths of a percent of dollars traded."""
        return cls(per_dollar_bips=bips, sell_only=sell_only)

    def as_two_sided(self) -> Fees:
        """Spread a sell-only fee over both sides, halving it."""
        if not self.sell_only:
            return self
        return Fees(
            per_share_mils=self.per_share_mils / 2,
            per_dollar_bips=self.per_dollar_bips / 2,
            sell_only=False,
        )

    def calculate(self, shares: float, dollars: float) -> float:
        """The cost of trading `shares` shares for `dollars` of notional."""
        f = self.as_two_sided()
        return (f.per_share_mils * shares + f.per_dollar_bips * dollars) * 1e-4

    def __add__(self, other: Fees) -> Fees:
        a, b = self.as_two_sided(), other.as_two_sided()
        return Fees(
            per_share_mils=a.per_share_mils + b.per_share_mils,
            per_dollar_bips=a.per_dollar_bips + b.per_dollar_bips,
        )

    def __mul__(self, x: float) -> Fees:
        return Fees(self.per_share_mils * x, self.per_dollar_bips * x, self.sell_only)

    def __truediv__(self, x: float) -> Fees:
        return self * (1.0 / x)


#: No trading costs at all.
ZERO = Fees()
