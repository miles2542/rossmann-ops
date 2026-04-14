import os
from datetime import date

import requests
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Rossmann Ops Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- State & Defaults ---
if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_URL", "http://localhost:8000")


def check_backend_health():
    try:
        response = requests.get(f"{st.session_state.api_url}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


health_data = check_backend_health()

# --- Header & Status ---
col_head, col_stat = st.columns([3, 1])
with col_head:
    st.title("Rossmann Forecast Dashboard")
with col_stat:
    # Right-aligned status indicator
    if health_data:
        status_color = "green" if health_data.get("model_loaded") else "orange"
        status_text = (
            "API: Online | Model: Ready"
            if health_data.get("model_loaded")
            else "API: Online | Model: Missing"
        )
    else:
        status_color = "red"
        status_text = "API: Offline"

    st.markdown(
        f"<div style='text-align: right; padding-top: 1.5rem;'><span style='color:{status_color};'>●</span> <b>{status_text}</b></div>",
        unsafe_allow_html=True,
    )

st.divider()

# --- Main Layout: Form (Left) & Results (Right) ---
col_form, col_result = st.columns([1.5, 1], gap="large")

with col_form:
    st.subheader("Forecast Parameters")
    with st.form("forecast_form"):
        # High-density input grid
        c1, c2, c3 = st.columns(3)
        store_id = c1.number_input(
            "Store ID", min_value=1, value=1, help="Unique store identifier"
        )
        forecast_date = c2.date_input("Date", value=date.today())

        # Streamlit dates default to Monday=0, we match standard 1=Mon, 7=Sun
        default_dow = forecast_date.weekday() + 1

        day_of_week = c3.selectbox(
            "Day of Week",
            options=[1, 2, 3, 4, 5, 6, 7],
            index=default_dow - 1,
            format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][
                x - 1
            ],
        )

        c4, c5, c6 = st.columns(3)
        promo = c4.selectbox(
            "Promo Active",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        holiday = c5.selectbox(
            "State Holiday",
            options=["0", "a", "b", "c"],
            format_func=lambda x: {
                "0": "None",
                "a": "Public",
                "b": "Easter",
                "c": "Christmas",
            }[x],
        )
        comp_dist = c6.number_input(
            "Comp. Distance (m)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Distance to nearest competitor. Leave 0 to use stored default.",
        )

        submitted = st.form_submit_button(
            "Generate Forecast", type="primary", use_container_width=True
        )

with col_result:
    st.subheader("Prediction Result")
    result_placeholder = st.empty()

    if not submitted:
        result_placeholder.info("Fill out the parameters and click Generate Forecast.")
    else:
        if not health_data:
            result_placeholder.error("Cannot predict: Backend API is offline.")
        elif not health_data.get("model_loaded"):
            result_placeholder.error(
                "Cannot predict: Model is missing on the backend. Needs retraining/export."
            )
        else:
            payload = {
                "Store": int(store_id),
                "DayOfWeek": int(day_of_week),
                "Date": forecast_date.isoformat(),
                "Promo": int(promo),
                "StateHoliday": holiday,
                "CompetitionDistance": comp_dist if comp_dist > 0 else None,
            }

            try:
                with st.spinner("Inference in progress..."):
                    res = requests.post(
                        f"{st.session_state.api_url}/predict", json=payload
                    )

                if res.status_code == 200:
                    data = res.json()
                    st.metric(
                        label=f"Predicted Sales (Store {data['Store']})",
                        value=f"€ {data['PredictedSales']:,.2f}",
                        delta=f"Model: {data['ModelVersion']}",
                        delta_color="off",
                    )
                else:
                    result_placeholder.error(
                        f"API Error: {res.json().get('detail', res.text)}"
                    )
            except Exception as e:
                result_placeholder.error(f"Connection failed: {str(e)}")

# --- Diagnostics Section ---
st.divider()
with st.expander("📊 Model Diagnostics & Explainability (SHAP)", expanded=False):
    st.markdown(
        "Global feature importance plot. If predictions are heavily skewed by unexpected features, investigate data drift."
    )
    if st.button("Fetch SHAP Summary"):
        shap_url = f"{st.session_state.api_url}/health/shap"
        try:
            r = requests.get(shap_url)
            if r.status_code == 200:
                st.image(r.content, use_column_width=True)
            else:
                st.warning(
                    "SHAP plot not found on server. Ensure model was exported correctly."
                )
        except Exception:
            st.error("Failed to connect to API.")

# --- Hidden Settings ---
with st.expander("⚙️ System Settings", expanded=False):
    new_url = st.text_input("Backend API URL", value=st.session_state.api_url)
    if st.button("Update Configuration"):
        st.session_state.api_url = new_url
        st.success("API Target Updated. Please refresh or interact to check health.")
