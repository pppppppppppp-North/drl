"""Thai equity transaction-cost model (milestone M1).

The environments charge ``transaction_cost + slippage`` per unit of turnover. Turnover sums the
absolute weight changes, so each unit is one side of a trade (a buy or a sell). This module turns
the published Thai fee schedule into that per-side fraction.

Explicit costs per side, as a fraction of traded value (checked 2026-11-06):

- Broker commission, internet channel, daily value up to THB 5 million: 0.157% (Settrade broker
  fee schedule; higher daily-value tiers are 0.137%, 0.117% and 0.107%).
- Exchange fees: SET trading fee 0.005%, TSD clearing fee 0.001%, SEC regulatory fee 0.001%.
- VAT: 7% of commission and fees.

Total ≈ 0.176% per side. The experiments' 0.15% commission + 0.05% slippage (0.20% per side) covers
it, leaving about 0.025% for market impact on large caps. Use a larger slippage for mid-caps.

Sources: https://wwwa1.settrade.com/brokerpage/023/StaticPage/home/attachfile/equitycomm.html ,
https://th.rhbtradesmart.com/en/equities-menu/equities-commission-and-charges ,
https://www.utrade.com.sg/page/site/public/english/pdf/UTRADE_New_Thailand_Market_Guide_v1.pdf
"""
from __future__ import annotations

from dataclasses import dataclass

INTERNET_COMMISSION_TIERS = (  # (max daily trading value in THB, commission rate before VAT)
    (5_000_000, 0.00157),
    (10_000_000, 0.00137),
    (20_000_000, 0.00117),
    (float("inf"), 0.00107),
)


@dataclass(frozen=True)
class ThaiEquityCosts:
    commission: float = 0.00157
    trading_fee: float = 0.00005
    clearing_fee: float = 0.00001
    regulatory_fee: float = 0.00001
    vat: float = 0.07

    @property
    def per_side(self) -> float:
        """Explicit cost per unit of turnover (one buy or one sell), VAT included."""

        return (self.commission + self.trading_fee + self.clearing_fee + self.regulatory_fee) * (1.0 + self.vat)

    @property
    def round_trip(self) -> float:
        return 2.0 * self.per_side


def commission_for_daily_value(daily_value_thb: float) -> float:
    """Internet-channel commission rate (before VAT) for a day's total trading value."""

    if daily_value_thb < 0:
        raise ValueError("daily trading value must be non-negative")
    for limit, rate in INTERNET_COMMISSION_TIERS:
        if daily_value_thb <= limit:
            return rate
    return INTERNET_COMMISSION_TIERS[-1][1]


def environment_cost_settings(slippage: float = 0.0005, daily_value_thb: float = 1_000_000) -> dict[str, float]:
    """``transaction_cost`` and ``slippage`` values for the environment config."""

    costs = ThaiEquityCosts(commission=commission_for_daily_value(daily_value_thb))
    return {"transaction_cost": round(costs.per_side, 7), "slippage": float(slippage)}
