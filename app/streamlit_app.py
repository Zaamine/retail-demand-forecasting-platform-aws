import asyncio
import os
import logging
import streamlit as st
import altair as alt
import boto3
from botocore.config import Config

st.set_page_config(
    page_title="Retail Planner",
    page_icon="📊",
    layout="wide",
)

st.title("Retail Planner")
st.caption("Local preview — verified forecasting assistant")

# Keep messages across Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "business_context" not in st.session_state:
    st.session_state.business_context = None

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.session_state.business_context = None
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask about your sales forecasts")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.write(question)

    # The backend handles the question later in this script.
# Forecast explorer V1
from pathlib import Path

import pandas as pd

from planner_app.snapshot import load_verified_forecasts
from planner_app.local_agent import build_local_planner, ask_local_planner
from planner_app.bedrock_agent import build_bedrock_planner, ask_bedrock_planner
from planner_app.live_quota import LocallyCappedBedrock
from planner_app.conversation import plan_turn, context_from_verified_result


@st.cache_resource(show_spinner="Verifying forecast snapshot...")
def get_forecasting_backend():
    """Initialize the read-only forecasting backend once."""
    root = Path(__file__).resolve().parents[1]
    return load_verified_forecasts(root)


st.divider()
st.subheader("Forecast Explorer")
st.caption("Historical forecast origin: August 15, 2017")

try:
    forecasts, forecast_service = get_forecasting_backend()
except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
    st.error(f"Cannot load verified forecasts: {exc}")
    st.stop()

stores = sorted(int(x) for x in forecasts["store_nbr"].unique())

left, middle, right = st.columns(3)

with left:
    store = st.selectbox(
        "Store",
        stores,
        index=stores.index(44),
    )

families = sorted(
    forecasts.loc[
        forecasts["store_nbr"] == store, "family"
    ].unique().tolist()
)

with middle:
    family = st.selectbox(
        "Product family",
        families,
        index=families.index("DAIRY") if "DAIRY" in families else 0,
    )

with right:
    period = st.selectbox(
        "Forecast period",
        ["next week", "next 7 days", "next two weeks"],
    )

result = forecast_service.sales_for_store(
    store_nbr=store,
    family=family,
    when=period,
)

st.metric(
    "Total expected sales",
    f"{result['total_predicted_sales']:,.2f} units",
)

st.caption(
    f"Forecast period: {result['period_start']} "
    f"to {result['period_end']}"
)

daily = pd.DataFrame(result["daily_predictions"])
daily["date"] = pd.to_datetime(daily["date"])

# Display discrete daily forecasts rather than hourly ticks.
chart_data = daily.assign(
    day_label=daily["date"].dt.strftime("%a %d %b")
)

peak = daily.loc[daily["predicted_sales"].idxmax()]

st.caption(
    f"Highest expected sales: {peak['date'].strftime('%A %d %B')} "
    f"({peak['predicted_sales']:,.0f} units)"
)

st.altair_chart(
    alt.Chart(chart_data).mark_line(point=True).encode(
        x=alt.X(
            "day_label:N",
            sort=chart_data["day_label"].tolist(),
            title="Forecast day",
        ),
        y=alt.Y(
            "predicted_sales:Q",
            title="Expected sales (units)",
        ),
        tooltip=[
            alt.Tooltip("day_label:N", title="Day"),
            alt.Tooltip(
                "predicted_sales:Q",
                title="Expected sales (units)",
                format=",.0f",
            ),
        ],
    ),
    width="stretch",
)

with st.expander("Daily sales forecast"):
    planner_daily = daily.rename(columns={
        "date": "Day",
        "predicted_sales": "Expected sales (units)",
    }).copy()

    planner_daily["Day"] = planner_daily["Day"].dt.strftime(
        "%a %d %b"
    )

    st.dataframe(
        planner_daily,
        hide_index=True,
        column_config={
            "Expected sales (units)": st.column_config.NumberColumn(
                format="%.0f",
            ),
        },
    )

# LOCAL_OFFLINE_AGENT_CHAT_V2
if question:
    action = plan_turn(
        question,
        context=st.session_state.business_context,
        known_families=forecasts["family"].unique(),
    )

    if action["mode"] == "reply":
        response = action["answer"]

    else:
        with st.spinner("Checking published sales forecasts..."):
            try:
                # LIVE_BEDROCK_UI_V1
                if os.getenv("RETAIL_PLANNER_ENABLE_BEDROCK") == "1":
                    root = Path(__file__).resolve().parents[1]

                    bedrock_client = boto3.Session(
                        region_name="eu-west-1"
                    ).client(
                        "bedrock-runtime",
                        config=Config(
                            connect_timeout=5,
                            read_timeout=60,
                            retries={"total_max_attempts": 1},
                        ),
                    )

                    capped_client = LocallyCappedBedrock(
                        client=bedrock_client,
                        quota_path=(
                            root / "data/forecasts"
                            / ".streamlit_bedrock_converse_quota"
                        ),
                        max_calls=12,
                    )

                    planner, mcp_server, budget = build_bedrock_planner(
                        root,
                        forecasts,
                        forecast_service,
                        capped_client,
                        enabled=True,
                        max_calls=6,
                    )

                    result = asyncio.run(
                        ask_bedrock_planner(
                            action["question"],
                            planner,
                            mcp_server,
                        )
                    )

                    logging.warning(
                        "Bedrock UI: status=%s, calls=%s, "
                        "input_tokens=%s, output_tokens=%s",
                        result.get("status"),
                        budget.attempts,
                        sum(r["input_tokens"] for r in budget.records),
                        sum(r["output_tokens"] for r in budget.records),
                    )
                else:
                    planner, mcp_server = build_local_planner(
                        Path(__file__).resolve().parents[1],
                        forecasts,
                        forecast_service,
                    )

                    result = asyncio.run(
                        ask_local_planner(
                            action["question"],
                            planner,
                            mcp_server,
                        )
                    )

            except Exception:
                logging.exception("Retail Planner execution failed")
                response = (
                    "I couldn't check the forecasts right now. "
                    "Please try again."
                )

            else:
                if result["status"] in (
                    "verified_analysis_evidence",
                    "verified_sales_evidence",
                ):
                    response = result["answer"]

                    context = context_from_verified_result(result)
                    if context is not None:
                        st.session_state.business_context = context

                elif result["status"] == "blocked_evidence":
                    response = (
                        "I couldn't verify this result against "
                        "our published forecasts."
                    )

                elif result["status"] == "blocked_execution":
                    response = (
                        "I couldn't complete this request safely. "
                        "Please try again later."
                    )

                elif action.get("used_context"):
                    response = (
                        "I couldn't apply that follow-up to the "
                        "available forecasts. Could you restate "
                        "the complete question?"
                    )

                else:
                    response = (
                        "This preview currently supports finding "
                        "leading stores and their busiest days. "
                        "Other forecast questions are not available yet."
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
    })

    st.rerun()
