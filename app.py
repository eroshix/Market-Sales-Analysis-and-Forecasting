import os
import io
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def _st_show(*args, **kwargs):
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close()
plt.show = _st_show

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from data_loader import load_raw_data
from data_cleaning import clean_data
import visualization as viz
from modeling import run_modeling
from time_series import run_time_series


st.set_page_config(page_title="Sales Analytics — Streamlit", layout="wide")
st.title("📊 Market Sales Analytics & Forecasting")

with st.sidebar:
    data_mode = st.radio(
        "Choose Data Source",
        ["Use 'train.csv' from project folder", "Upload CSV file"],
        index=0
    )
    default_path = st.text_input(
        "File Path (for file in project folder)",
        value="train.csv",
        help="If the file is in the same folder as app.py, simply type 'train.csv'."
    )
    uploaded = None
    if data_mode == "Upload CSV file":
        uploaded = st.file_uploader("Select and upload a CSV file", type=["csv"])

    st.markdown("---")
    st.caption("Chart rendering may take a moment — thank you for your patience 😊")

@st.cache_data(show_spinner=False)
def _read_uploaded_csv(file_like) -> pd.DataFrame:
    return pd.read_csv(file_like)

@st.cache_data(show_spinner=False)
def _load_default_csv(path: str) -> pd.DataFrame:
    return load_raw_data(path)

def _load_and_clean():
    if uploaded is not None:
        df_raw = _read_uploaded_csv(uploaded)
    else:
        df_raw = _load_default_csv(default_path)

    df_clean = clean_data(df_raw)
    return df_raw, df_clean


st.subheader("1) Data Loading & Cleaning")
if st.button("Load and Clean Data", type="primary"):
    with st.spinner("Loading and cleaning..."):
        try:
            df_raw, df_clean = _load_and_clean()
            st.success("Data is ready!")
            st.session_state["df_raw"] = df_raw
            st.session_state["df_clean"] = df_clean
        except Exception as e:
            st.error(f"An error occurred: {e}")


if "df_clean" in st.session_state:
    df_raw = st.session_state["df_raw"]
    df_clean = st.session_state["df_clean"]

    st.write("**Raw Data (first 5 rows):**")
    st.dataframe(df_raw.head(), use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Row Count (Raw)", len(df_raw))
    with c2: st.metric("Column Count (Raw)", df_raw.shape[1])
    with c3: st.metric("Row Count (Clean)", len(df_clean))
    with c4: st.metric("Column Count (Clean)", df_clean.shape[1])

    with st.expander("Column Types (Clean Data)"):
        dtypes_df = pd.DataFrame({"column": df_clean.columns, "dtype": df_clean.dtypes.astype(str)})
        st.dataframe(dtypes_df, use_container_width=True)

    st.subheader("2) Exploratory Visualizations")
    st.caption("Select the chart(s) you want to generate.")
    
    viz_options = {
        "Sales Distribution (Histogram)": viz.plot_sales_distribution,
        "Daily Sales Trend": viz.plot_daily_sales_trend,
        "Weekly Sales Trend": viz.plot_weekly_sales_trend,
        "Monthly Sales Trend": viz.plot_monthly_sales_trend,
        "Monthly Sales by Sub-category": viz.plot_monthly_sales_by_subcategory,
        "Sales Distribution + KDE": viz.plot_sales_distribution_with_kde,
        "Top-N Cities by Total Sales": viz.plot_top_n_cities_sales,
        "Total Sales by Sub-category": viz.plot_sales_by_subcategory,
        "Total Sales by Segment": viz.plot_total_sales_by_segment,
        "Total Sales by Region": viz.plot_total_sales_by_region,
        "Total Sales by Shipping Mode": viz.plot_total_sales_by_shipping_mode,
        "Sales Distribution by Segment (Violin Plot)": viz.plot_sales_distribution_by_segment_violin,
        "Moving Average Trend": viz.plot_sales_trend_with_moving_average,
        "Total Sales by Category": viz.plot_total_sales_by_category,
    }

    selected_viz = st.multiselect("Select chart(s) to draw", list(viz_options.keys()))
    if st.button("Generate Selected Charts"):
        with st.spinner("Generating charts..."):
            for k in selected_viz:
                st.markdown(f"**{k}**")
                try:
                    viz_options[k](df_clean)
                except Exception as e:
                    st.error(f"Error while generating {k}: {e}")

    st.subheader("3) Basic Modeling (Linear Regression)")
    st.caption("Expanded features → Sales; metrics don’t change, plots use zoom/log for readability.")
    clip_lo, clip_hi = st.sidebar.slider("Clip percentiles for Linear Regression (plots only)", 0, 100, (1, 99), 1)
    use_log = st.sidebar.checkbox("Use log1p axes for plots", False)
    if st.button("Run Modeling"):
        with st.spinner("Training..."):
            try:
                out = run_modeling(
                    df_clean,
                    show_plots=True,
                    clip_quantiles=(clip_lo, clip_hi),
                    log_scale=use_log,
                )
                if isinstance(out, dict) and "metrics_last" in out:
                    m = out["metrics_last"]
                    st.success("Modeling completed!")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("MSE", f'{m.get("mse", float("nan")):.3f}')
                    c2.metric("RMSE", f'{m.get("rmse", float("nan")):.3f}')
                    c3.metric("R²", f'{m.get("r2", float("nan")):.3f}')
                    c4.metric("MAPE (%)", f'{m.get("mape", float("nan")):.2f}')
                else:
                    st.info("The function did not return a metrics dictionary; check the console output.")
            except Exception as e:
                st.error(f"Error during modeling: {e}")

    st.subheader("4) Time Series — 7-Day Forecast (Prophet & SARIMA)")
    st.caption("Runs the prepared function to forecast the next 7 days and compare results.")
    if st.button("Run Time Series Analysis & Forecast"):
        with st.spinner("Training and forecasting with Prophet/SARIMA..."):
            try:
                ts_out = run_time_series(df_clean)
                if isinstance(ts_out, dict):
                    p = ts_out.get("prophet", {})
                    s = ts_out.get("sarima", {})
                    st.success("Forecast completed!")
                    st.markdown("**Summary Metrics (Test Set)**")
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Prophet RMSE", f'{p.get("rmse", float("nan")):.2f}')
                    cc2.metric("Prophet MAPE (%)", f'{p.get("mape", float("nan")):.2f}')
                    cc3.metric("Prophet R²", f'{p.get("r2", float("nan")):.2f}')

                    dd1, dd2, dd3 = st.columns(3)
                    dd1.metric("SARIMA RMSE", f'{s.get("rmse", float("nan")):.2f}')
                    dd2.metric("SARIMA MAPE (%)", f'{s.get("mape", float("nan")):.2f}')
                    dd3.metric("SARIMA R²", f'{s.get("r2", float("nan")):.2f}')

                    if p.get("best_params"):
                        st.json({"Prophet best_params": p["best_params"]})
                else:
                    st.info("The function did not return a metrics dictionary; check the console output.")
            except Exception as e:
                st.error(f"Error during time series analysis: {e}")

else:
    st.info("Select a data source from the sidebar and click **Load and Clean Data**.")
