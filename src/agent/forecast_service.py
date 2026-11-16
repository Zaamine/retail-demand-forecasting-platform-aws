"""Deterministic sales-estimate queries over a validated forecast snapshot."""

from datetime import timedelta

import numpy as np
import pandas as pd

from agent.planning_contract import PlanningContract


class ForecastService:
    """Query one complete forecast snapshot without calling AWS or an LLM."""

    def __init__(self, forecasts: pd.DataFrame, calendar, source: str):
        required = {"store_nbr", "family", "target_date", "prediction"}

        if missing := required - set(forecasts.columns):
            raise ValueError(f"Missing forecast columns: {sorted(missing)}")

        data = forecasts.copy()
        data["target_date"] = pd.to_datetime(data["target_date"]).dt.date
        data["family"] = data["family"].str.strip().str.upper()

        if data.empty or data[list(required)].isna().any().any():
            raise ValueError("Missing forecast values.")

        if not np.isfinite(data["prediction"].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite predictions.")

        if (data["prediction"] < 0).any():
            raise ValueError("Negative sales predictions.")

        if data.duplicated(
            ["store_nbr", "family", "target_date"]
        ).any():
            raise ValueError("Duplicate forecast business keys.")

        if "forecast_origin" in data:
            origins = pd.to_datetime(data["forecast_origin"]).dt.date

            if not origins.eq(calendar.business_date).all():
                raise ValueError(
                    "Forecast origin does not match the calendar."
                )

        first = calendar.business_date + timedelta(days=1)
        last = calendar.business_date + timedelta(
            days=calendar.horizon_days
        )

        if not data["target_date"].between(first, last).all():
            raise ValueError(
                "Forecast dates outside the configured horizon."
            )

        counts = data.groupby(
            ["store_nbr", "family"]
        )["target_date"].nunique()

        if not counts.eq(calendar.horizon_days).all():
            raise ValueError(
                "Incomplete snapshot for at least one series."
            )

        self._data = data
        self.calendar = calendar
        self.source = source

    def _select(self, contract: PlanningContract) -> pd.DataFrame:
        """Retrieve observations matching a validated planning scope."""

        full = self.calendar.resolve("full horizon")

        if (
            contract.period.start < full.start
            or contract.period.end > full.end
        ):
            raise ValueError(
                "Planning contract exceeds the loaded snapshot."
            )

        data = self._data.loc[
            self._data["target_date"].between(
                contract.period.start,
                contract.period.end,
            )
        ]

        if contract.stores is not None:
            data = data.loc[
                data["store_nbr"].isin(contract.stores)
            ]

        if contract.families is not None:
            data = data.loc[
                data["family"].isin(contract.families)
            ]

        if data.empty:
            raise ValueError(
                "No forecasts match the planning contract."
            )

        if (
            contract.stores is not None
            and set(data["store_nbr"]) != set(contract.stores)
        ):
            raise ValueError("Missing requested stores.")

        if (
            contract.families is not None
            and set(data["family"]) != set(contract.families)
        ):
            raise ValueError("Missing requested product families.")

        if contract.stores is not None and contract.families is not None:
            actual = set(zip(data["store_nbr"], data["family"]))

            expected = {
                (store, family)
                for store in contract.stores
                for family in contract.families
            }

            if actual != expected:
                raise ValueError(
                    "Missing requested store-family combinations."
                )

        return data

    def sales_for_store(
        self,
        store_nbr: int,
        family: str,
        when: str,
    ) -> dict:
        """Return daily estimates and their total for one store-family."""

        contract = PlanningContract.create(
            self.calendar,
            when,
            stores=[store_nbr],
            families=[family],
        )

        rows = self._select(contract).sort_values("target_date")

        return {
            "status": "available",
            "store_nbr": store_nbr,
            "family": family.strip().upper(),
            "period_start": contract.period.start.isoformat(),
            "period_end": contract.period.end.isoformat(),
            "daily_predictions": [
                {
                    "date": day.isoformat(),
                    "predicted_sales": float(value),
                }
                for day, value in zip(
                    rows["target_date"],
                    rows["prediction"],
                )
            ],
            "total_predicted_sales": float(
                rows["prediction"].sum()
            ),
            "source": self.source,
        }

    def analyze(
        self,
        when: str,
        group_by: list[str],
        *,
        stores: list[int] | None = None,
        families: list[str] | None = None,
        top_n: int = 10,
        order: str = "highest",
    ) -> dict:
        """Filter, aggregate and rank sales estimates."""

        allowed = {"store_nbr", "family", "target_date"}

        if (
            not group_by
            or len(group_by) != len(set(group_by))
            or not set(group_by).issubset(allowed)
        ):
            raise ValueError("Invalid grouping dimensions.")

        if order not in {"highest", "lowest", "chronological"}:
            raise ValueError("Invalid ordering.")

        if type(top_n) is not int or not 1 <= top_n <= 100:
            raise ValueError("top_n must be between 1 and 100.")

        if order == "chronological" and "target_date" not in group_by:
            raise ValueError(
                "Chronological ordering requires target_date."
            )

        contract = PlanningContract.create(
            self.calendar, when, stores, families
        )
        rows = self._select(contract)

        # Do not silently aggregate different product units together.
        if rows["family"].nunique() > 1 and "family" not in group_by:
            raise ValueError(
                "Group by family to avoid mixing product units."
            )

        summary = (
            rows.groupby(group_by, as_index=False)["prediction"]
            .sum()
            .rename(columns={"prediction": "predicted_sales"})
        )

        if order == "chronological":
            summary = summary.sort_values(
                ["target_date"]
                + [col for col in group_by if col != "target_date"]
            )
        else:
            summary = summary.sort_values(
                ["predicted_sales"] + group_by,
                ascending=[order == "lowest"]
                + [True] * len(group_by),
            )

        if "target_date" in group_by:
            summary["target_date"] = (
                summary["target_date"].astype(str)
            )

        return {
            "status": "available",
            "period_start": contract.period.start.isoformat(),
            "period_end": contract.period.end.isoformat(),
            "stores": sorted(set(rows["store_nbr"].astype(int))),
            "families": sorted(set(rows["family"])),
            "group_by": group_by,
            "rows_analyzed": len(rows),
            "groups_found": len(summary),
            "truncated": len(summary) > top_n,
            "results": summary.head(top_n).to_dict(
                orient="records"
            ),
            "source": self.source,
        }
