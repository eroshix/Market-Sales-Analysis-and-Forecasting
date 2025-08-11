import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_time_series(df: pd.DataFrame):
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["day_of_week"] = df.index.dayofweek
    df["week"] = df.index.isocalendar().week
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df["season"] = df["month"].apply(get_season)

    daily_sales = df.groupby("Order Date")["Sales"].sum()

    daily_sales_df = pd.DataFrame({'Sales': daily_sales})
    daily_sales_df['Sales_lag1'] = daily_sales_df['Sales'].shift(1)
    daily_sales_df['Sales_lag7'] = daily_sales_df['Sales'].shift(7)
    daily_sales_df['Sales_lag30'] = daily_sales_df['Sales'].shift(30)
    daily_sales_df = daily_sales_df.dropna()

    df = df.join(daily_sales_df[['Sales_lag1', 'Sales_lag7', 'Sales_lag30']])

    df.dropna(subset=['Sales_lag1', 'Sales_lag7', 'Sales_lag30'], inplace=True)

    sns.scatterplot(data=daily_sales_df, x='Sales_lag1', y='Sales')
    plt.title('Sales vs. Sales_lag1')
    plt.show()

    sns.scatterplot(data=daily_sales_df, x='Sales_lag7', y='Sales')
    plt.title('Sales vs. Sales_lag7')
    plt.show()

    sns.scatterplot(data=daily_sales_df, x='Sales_lag30', y='Sales')
    plt.title('Sales vs. Sales_lag30')
    plt.show()

    print("[time_series] Lag korelasyonları:")
    print(daily_sales_df[['Sales', 'Sales_lag1', 'Sales_lag7', 'Sales_lag30']].corr())

    daily_sales_prophet = df.groupby("Order Date")["Sales"].sum().reset_index()
    daily_sales_prophet.columns = ["ds", "y"]

    train_size = int(len(daily_sales_prophet) * 0.85)
    train = daily_sales_prophet.iloc[:train_size]
    test = daily_sales_prophet.iloc[train_size:]

    prophet = Prophet()
    prophet.fit(daily_sales_prophet)

    future = prophet.make_future_dataframe(periods=7, freq="D")
    forecasts_prophet = prophet.predict(future)

    forecasts_prophet_7 = forecasts_prophet[["ds", "yhat"]].tail(7)

    prophet.plot(forecasts_prophet)
    plt.legend()
    plt.show()

    prophet.plot_components(forecasts_prophet)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_sales_prophet['ds'], daily_sales_prophet['y'], label='Real Values', color='black')
    plt.plot(forecasts_prophet['ds'], forecasts_prophet['yhat'], label='Prediction (yhat)', color='blue')
    plt.fill_between(
        forecasts_prophet['ds'],
        forecasts_prophet['yhat_lower'],
        forecasts_prophet['yhat_upper'],
        color='skyblue',
        alpha=0.3,
        label='Uncertainty Interval'
    )
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Real vs Prophet Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    daily_sales_sarima = df.groupby("Order Date")["Sales"].sum().reset_index()
    daily_sales_sarima.set_index("Order Date", inplace=True)

    full_range = pd.date_range(start=daily_sales_sarima.index.min(), end=daily_sales_sarima.index.max(), freq="D")
    daily_sales_sarima = daily_sales_sarima.reindex(full_range)
    daily_sales_sarima.index.name = "Order Date"
    daily_sales_sarima = daily_sales_sarima.ffill()

    def adf_test(series):
        result = adfuller(series.dropna())
        print("ADF Test Result")
        print("-------------------------")
        print(f"ADF statistics: {result[0]}")
        print(f"p-value: {result[1]}")
        print("Critical Values:")
        for key, value in result[4].items():
            print(f"   {key}: {value}")
        if result[1] <= 0.05:
            print("This series is stationary. (p ≤ 0.05)")
        else:
            print("This series is not stationary. (p > 0.05)")

    adf_test(daily_sales_sarima['Sales'])

    sarima = SARIMAX(
        endog=daily_sales_sarima,
        order=(2, 0, 0),
        seasonal_order=(2, 1, 0, 7),
        enforce_invertibility=False,
        enforce_stationarity=False
    )

    results = sarima.fit(disp=False)
    print(results.summary())

    forecast_steps = 7
    forecasts_sarima = results.get_forecast(steps=forecast_steps)
    forecast_sarima_mean = forecasts_sarima.summary_frame().reset_index()
    forecast_sarima_mean.rename(columns={'index': 'ds', 'mean': 'yhat_sarima'}, inplace=True)

    forecast_ci = forecasts_sarima.conf_int()

    ax = daily_sales_sarima['Sales'].plot(label='Sales', figsize=(20, 6))
    forecasts_sarima.predicted_mean.plot(ax=ax, label='SARIMA Predict', color='red')
    ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='red', alpha=0.2)
    plt.title("SARIMA Forecast vs Sales")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    last_23 = daily_sales_sarima.tail(23)
    forecast_index = forecasts_sarima.predicted_mean.index
    forecast_values = forecasts_sarima.predicted_mean
    forecast_ci = forecasts_sarima.conf_int()
    transition_index = [last_23.index[-1], forecast_index[0]]
    transition_values = [last_23['Sales'].iloc[-1], forecast_values.iloc[0]]
    plt.plot(last_23.index, last_23['Sales'], label='Actual Sales', color='blue')
    plt.plot(transition_index, transition_values, color='purple', linestyle='--', linewidth=2, alpha=0.6)
    plt.plot(forecast_index, forecast_values, label='SARIMA Forecast', color='red', linewidth=2)
    plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='red', alpha=0.2, label='Confidence Interval')
    plt.title("SARIMA Forecast – Last 30 Days (Zoomed In)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    comparison_df = pd.merge(forecasts_prophet_7, forecast_sarima_mean[['ds', 'yhat_sarima']], on='ds', how='inner')
    plt.figure(figsize=(10, 5))
    plt.plot(comparison_df['ds'], comparison_df['yhat'], label='Prophet Forecast', marker='o', color="#0997F0")
    plt.plot(comparison_df['ds'], comparison_df['yhat_sarima'], label='SARIMA Forecast', marker='x', color="#F1F509")
    plt.title('7-Day Forecast: Prophet vs SARIMA')
    plt.xlabel('Date')
    plt.ylabel('Sales Forecast')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()

    forecast_sarima_mean.columns
    y_test = df.groupby("Order Date")["Sales"].sum().reset_index().tail(7)['Sales'].values

    def mean_absolute_percentage_error_local(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    yhat_prophet = forecasts_prophet_7['yhat'].values
    yhat_sarima = forecast_sarima_mean['yhat_sarima'].values

    rmse_prophet = np.sqrt(mean_squared_error(y_test, yhat_prophet))
    rmse_sarima = np.sqrt(mean_squared_error(yhat_sarima, yhat_prophet))

    mape_prophet = mean_absolute_percentage_error_local(y_test, yhat_prophet)
    mape_sarima = mean_absolute_percentage_error_local(y_test, yhat_sarima)

    r2_prophet = r2_score(y_test, yhat_prophet)
    r2_sarima = r2_score(y_test, yhat_sarima)

    print(f"Prophet RMSE: {rmse_prophet:.2f}, MAPE: %{mape_prophet:.2f}, R2:{r2_prophet:.2f}")
    print(f"SARIMA  RMSE: {rmse_sarima:.2f}, MAPE: %{mape_sarima:.2f}, R2:{r2_sarima:.2f}")

    param_grid_prophet = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    all_params_prophet = [dict(zip(param_grid_prophet.keys(), v)) for v in itertools.product(*param_grid_prophet.values())]
    errors = []
    df_prophet = daily_sales_prophet[['ds', 'y']].copy()

    for params in all_params_prophet:
        model = Prophet(**params)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        y_true = df_prophet['y'].values
        y_pred = forecast['yhat'][:len(y_true)].values
        error = mean_absolute_error(y_true, y_pred)
        errors.append(error)

    best_params_prophet = all_params_prophet[errors.index(min(errors))]
    print("Best Parameters", best_params_prophet)

    best_params = {
        'changepoint_prior_scale': 0.01,
        'seasonality_prior_scale': 1.0,
        'seasonality_mode': 'additive'
    }
    prophet_tuning = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        seasonality_mode=best_params['seasonality_mode']
    )

    prophet_tuning.fit(daily_sales_prophet)
    future_prophet_tuning = model.make_future_dataframe(periods=7)  # NOTE:
    forecast_tuning = prophet_tuning.predict(future_prophet_tuning)

    prophet_tuning.plot(forecast_tuning)
    plt.legend()
    plt.show()

    y_true = df_prophet["y"]
    y_pred = forecasts_prophet["yhat"][:len(df_prophet)]

    mae_ht = mean_absolute_error(y_true, y_pred)
    rmse_ht = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_ht = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2_ht = r2_score(y_true, y_pred)

    print(f"MAE  : {mae_ht:.2f}")
    print(f"RMSE : {rmse_ht:.2f}")
    print(f"MAPE : {mape_ht:.2f}%")
    print(f"R²   : {r2_ht:.4f}")


if __name__ == "__main__":
    from data_loader import load_raw_data
    from data_cleaning import clean_data
    df_raw   = load_raw_data("C:\\Users\\stajyer\\Desktop\\GPT\\train.csv")
    df_clean = clean_data(df_raw)
    run_time_series(df_clean)