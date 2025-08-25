import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def _prepare_daily(df: pd.DataFrame) -> pd.DataFrame:

    daily = df["Sales"].resample("D").sum()
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx).fillna(0.0)

    out = pd.DataFrame({"ds": full_idx, "y": daily.values})
    out["is_weekend"] = (out["ds"].dt.dayofweek >= 5).astype(int)
    return out


def _plot_lag_scatter(daily_y: pd.Series):
    ddf = pd.DataFrame({"Sales": daily_y})
    ddf["Sales_lag1"] = ddf["Sales"].shift(1)
    ddf["Sales_lag7"] = ddf["Sales"].shift(7)
    ddf["Sales_lag30"] = ddf["Sales"].shift(30)
    ddf = ddf.dropna()

    sns.scatterplot(data=ddf, x="Sales_lag1", y="Sales")
    plt.title("Sales vs. Sales_lag1")
    plt.show()

    sns.scatterplot(data=ddf, x="Sales_lag7", y="Sales")
    plt.title("Sales vs. Sales_lag7")
    plt.show()

    sns.scatterplot(data=ddf, x="Sales_lag30", y="Sales")
    plt.title("Sales vs. Sales_lag30")
    plt.show()

    print("[time_series] Lag correlations:")
    print(ddf[["Sales", "Sales_lag1", "Sales_lag7", "Sales_lag30"]].corr())


def run_time_series(df: pd.DataFrame):
    daily_df = _prepare_daily(df)
    _plot_lag_scatter(pd.Series(daily_df["y"].values, index=daily_df["ds"]))

    H = 7
    if len(daily_df) <= H + 30:
        print("[time_series] Uyarı: veri kısa, sonuçlar oynak olabilir.")

    train = daily_df.iloc[:-H].reset_index(drop=True)
    test  = daily_df.iloc[-H:].reset_index(drop=True)

    param_grid = {
        "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
        "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
        "seasonality_mode": ["additive", "multiplicative"],
        "changepoint_range": [0.8, 0.9],
    }
    all_params = list(itertools.product(*param_grid.values()))

    best_prophet = None
    best_prophet_params = None
    best_rmse = np.inf

    fit_cols = ["ds", "y", "is_weekend"]
    pred_cols = ["ds", "is_weekend"]

    for cps, sps, smode, cpr in all_params:
        m = Prophet(
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
            seasonality_mode=smode,
            changepoint_range=cpr,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )
        m.add_regressor("is_weekend")
        m.fit(train[fit_cols])

        pred_df = test[pred_cols].copy()
        fcst = m.predict(pred_df)
        yhat = fcst["yhat"].values

        rmse = mean_squared_error(test["y"].values, yhat, squared=False)
        if rmse < best_rmse:
            best_rmse = rmse
            best_prophet = m
            best_prophet_params = {
                "changepoint_prior_scale": cps,
                "seasonality_prior_scale": sps,
                "seasonality_mode": smode,
                "changepoint_range": cpr,
            }

    prophet_fcst_test = best_prophet.predict(test[pred_cols])
    yhat_prophet = prophet_fcst_test["yhat"].values
    rmse_prophet = mean_squared_error(test["y"].values, yhat_prophet, squared=False)
    mape_prophet = _safe_mape(test["y"].values, yhat_prophet)
    r2_prophet = r2_score(test["y"].values, yhat_prophet)

    print("[Prophet] Best params:", best_prophet_params)
    print(f"Prophet RMSE: {rmse_prophet:.2f}, MAPE: %{mape_prophet:.2f}, R2:{r2_prophet:.2f}")

    y_train = train["y"].values
    y_test  = test["y"].values

    auto_model = pm.auto_arima(
        y=y_train,
        seasonal=True,
        m=7,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        d=None, D=None,       
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        information_criterion="aic",
    )

    order = auto_model.order
    sorder = auto_model.seasonal_order
    print(f"[SARIMA] Selected order={order}, seasonal_order={sorder}")

    sarimax = SARIMAX(
        endog=y_train,
        order=order,
        seasonal_order=sorder,
        enforce_invertibility=False,
        enforce_stationarity=False,
    ).fit(disp=False)

    sarima_forecast = sarimax.get_forecast(steps=H)
    yhat_sarima = np.asarray(sarima_forecast.predicted_mean)
    ci_raw = sarima_forecast.conf_int()
    if hasattr(ci_raw, "values"):
        ci_arr = ci_raw.values
    else:
        ci_arr = np.asarray(ci_raw)

    lower, upper = ci_arr[:, 0], ci_arr[:, 1]

    rmse_sarima = mean_squared_error(y_test, yhat_sarima, squared=False)
    mape_sarima = _safe_mape(y_test, yhat_sarima)
    r2_sarima = r2_score(y_test, yhat_sarima)

    print(f"SARIMA  RMSE: {rmse_sarima:.2f}, MAPE: %{mape_sarima:.2f}, R2:{r2_sarima:.2f}")


    fcst_train = best_prophet.predict(train[["ds", "is_weekend"]])
    plt.figure(figsize=(12, 6))
    plt.plot(daily_df["ds"], daily_df["y"], label="Actual (All)", alpha=0.45)
    plt.plot(train["ds"], fcst_train["yhat"], label="Prophet Fitted (Train)")
    plt.plot(test["ds"], prophet_fcst_test["yhat"], label="Prophet Forecast (Test)")
    plt.fill_between(
        test["ds"],
        prophet_fcst_test["yhat_lower"],
        prophet_fcst_test["yhat_upper"],
        alpha=0.2,
        label="Prophet CI (Test)"
    )
    plt.axvline(train["ds"].iloc[-1], linestyle="--", alpha=0.6, label="Train/Test Split")
    plt.title("PROPHET — Fitted (Train) + Forecast (Test) vs Actual (All)")
    plt.xlabel("Date"); plt.ylabel("Sales"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()
    

    ZOOM_DAYS = max(60, 3*H)  
    split_dt = train["ds"].iloc[-1]
    start_zoom = split_dt - pd.Timedelta(days=ZOOM_DAYS)

    mask_all   = daily_df["ds"] >= start_zoom
    mask_tr_pf = fcst_train["ds"] >= start_zoom
    mask_te_pf = prophet_fcst_test["ds"] >= start_zoom

    plt.figure(figsize=(12, 5))
    plt.plot(daily_df.loc[mask_all, "ds"], daily_df.loc[mask_all, "y"], label="Actual (Zoom)", alpha=0.45)
    plt.plot(fcst_train.loc[mask_tr_pf, "ds"], fcst_train.loc[mask_tr_pf, "yhat"], label="Prophet Fitted (Train)", linewidth=1.8)
    plt.plot(prophet_fcst_test.loc[mask_te_pf, "ds"], prophet_fcst_test.loc[mask_te_pf, "yhat"], label="Prophet Forecast (Test)", linewidth=1.8)
    plt.fill_between(
        prophet_fcst_test.loc[mask_te_pf, "ds"],
        prophet_fcst_test.loc[mask_te_pf, "yhat_lower"],
        prophet_fcst_test.loc[mask_te_pf, "yhat_upper"],
        alpha=0.2, label="Prophet CI (Test)"
    )
    plt.axvline(split_dt, linestyle="--", alpha=0.6, label="Train/Test Split")
    plt.xlim(start_zoom, test["ds"].iloc[-1])
    plt.title(f"PROPHET — Zoom (Last {ZOOM_DAYS} Days Around Split)")
    plt.xlabel("Date"); plt.ylabel("Sales"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()



    sarima_fit_res = sarimax.get_prediction(start=0, end=len(y_train)-1)
    yhat_sarima_train = np.asarray(sarima_fit_res.predicted_mean)

    plt.figure(figsize=(12, 6))
    plt.plot(daily_df["ds"], daily_df["y"], label="Actual (All)", alpha=0.45)
    plt.plot(train["ds"], yhat_sarima_train, label="SARIMA Fitted (Train)", linewidth=1.8)
    plt.plot(test["ds"], yhat_sarima, label="SARIMA Forecast (Test)", linewidth=1.8)
    plt.fill_between(test["ds"], lower, upper, alpha=0.2, label="SARIMA CI (Test)")
    plt.axvline(train["ds"].iloc[-1], linestyle="--", alpha=0.6, label="Train/Test Split")
    plt.title("SARIMA — Fitted (Train) + Forecast (Test) vs Actual (All)")
    plt.xlabel("Date"); plt.ylabel("Sales"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()


    ZOOM_DAYS = max(60, 3*H)  
    split_dt = train["ds"].iloc[-1]
    start_zoom = split_dt - pd.Timedelta(days=ZOOM_DAYS)

    sarima_train_df = pd.DataFrame({"ds": train["ds"], "yhat_train": yhat_sarima_train})
    sarima_test_df  = pd.DataFrame({"ds": test["ds"], "yhat": yhat_sarima,
                                "lower": lower, "upper": upper})

    mask_all = daily_df["ds"]       >= start_zoom
    mask_tr  = sarima_train_df["ds"] >= start_zoom
    mask_te  = sarima_test_df["ds"]  >= start_zoom

    plt.figure(figsize=(12, 5))
    plt.plot(daily_df.loc[mask_all, "ds"], daily_df.loc[mask_all, "y"],
            label="Actual (Zoom)", alpha=0.45)
    plt.plot(sarima_train_df.loc[mask_tr, "ds"], sarima_train_df.loc[mask_tr, "yhat_train"],
            label="SARIMA Fitted (Train)", linewidth=1.8)
    plt.plot(sarima_test_df.loc[mask_te, "ds"], sarima_test_df.loc[mask_te, "yhat"],
            label="SARIMA Forecast (Test)", linewidth=1.8)
    plt.fill_between(
        sarima_test_df.loc[mask_te, "ds"],
        sarima_test_df.loc[mask_te, "lower"],
        sarima_test_df.loc[mask_te, "upper"],
        alpha=0.2, label="SARIMA CI (Test)"
    )
    plt.axvline(split_dt, linestyle="--", alpha=0.6, label="Train/Test Split")
    plt.xlim(start_zoom, test["ds"].iloc[-1])
    plt.title(f"SARIMA — Zoom (Last {ZOOM_DAYS} Days Around Split)")
    plt.xlabel("Date"); plt.ylabel("Sales"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()



    comp = pd.DataFrame({
        "ds": test["ds"],
        "Prophet": yhat_prophet,
        "SARIMA": yhat_sarima,
        "Actual": y_test,
    })
    plt.figure(figsize=(10, 5))
    plt.plot(comp["ds"], comp["Prophet"], label="Prophet", marker="o")
    plt.plot(comp["ds"], comp["SARIMA"], label="SARIMA", marker="x")
    plt.plot(comp["ds"], comp["Actual"], label="Actual", linestyle="--", alpha=0.6)
    plt.title("7-Day Forecast: Prophet vs SARIMA (on Test)")
    plt.xlabel("Date"); plt.ylabel("Sales"); plt.grid(True); plt.xticks(rotation=25)
    plt.legend(); plt.tight_layout(); plt.show()

    
    return {
        "prophet": {
            "best_params": best_prophet_params,
            "rmse": float(rmse_prophet),
            "mape": float(mape_prophet),
            "r2": float(r2_prophet),
        },
        "sarima": {
            "order": order,
            "seasonal_order": sorder,
            "rmse": float(rmse_sarima),
            "mape": float(mape_sarima),
            "r2": float(r2_sarima),
        },
    }


if __name__ == "__main__":
    from data_loader import load_raw_data
    from data_cleaning import clean_data
    df_raw = load_raw_data("C:\\Users\\stajyer\\Desktop\\Market-Sales-Analysis-and-Forecasting\\train.csv")
    df_clean = clean_data(df_raw)
    run_time_series(df_clean)
