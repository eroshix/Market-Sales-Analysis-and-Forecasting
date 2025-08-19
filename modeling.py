import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



def run_modeling(
    df: pd.DataFrame,
    show_plots: bool = True,
    top_k_coeffs: int = 15,
    clip_quantiles=(1, 99),
    log_scale: bool = False
):


    def mape_ignore_zeros(y_true, y_pred):
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)
        nz = y_true != 0
        if nz.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz]) * 100))

    df1 = df.copy()

    if not isinstance(df1.index, pd.DatetimeIndex):
        if "Order Date" in df1.columns:
            df1["Order Date"] = pd.to_datetime(df1["Order Date"], dayfirst=True, errors="coerce")
            df1 = df1.set_index("Order Date").sort_index()
        else:
            raise ValueError("Order Date index not found.")

    df1["order_month"] = df1.index.month
    df1["order_dow"] = df1.index.dayofweek

    if "Ship Date" in df1.columns:
        ship_dt = pd.to_datetime(df1["Ship Date"], errors="coerce")
        df1["ship_delay_days"] = (ship_dt - df1.index).dt.days
    else:
        df1["ship_delay_days"] = np.nan

    y = df1["Sales"].astype(float)

    num_cols = ["Quantity", "Discount", "order_month", "order_dow", "ship_delay_days"]
    cat_cols = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]

    num_cols = [c for c in num_cols if c in df1.columns]
    cat_cols = [c for c in cat_cols if c in df1.columns]

    X = df1[num_cols + cat_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("linreg", LinearRegression())
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    mape = float(mape_ignore_zeros(y_test, y_pred))

    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE (%): {mape:.2f}")

    if show_plots:
        def _clip_mask(x, y, q):
            """x,y dizileri için yüzde q (q_low,q_high) dışında kalanları gizler."""
            if q is None:
                return np.isfinite(x) & np.isfinite(y)
            ql_x, qh_x = np.nanpercentile(x, q)
            ql_y, qh_y = np.nanpercentile(y, q)
            return (
                np.isfinite(x) & np.isfinite(y) &
                (x >= ql_x) & (x <= qh_x) &
                (y >= ql_y) & (y <= qh_y)
            )

        xa = y_test.values.astype(float)
        ya = y_pred.astype(float)
        if log_scale:
            xa = np.log1p(np.clip(xa, a_min=0, a_max=None))
            ya = np.log1p(np.clip(ya, a_min=0, a_max=None))

        mask = _clip_mask(xa, ya, clip_quantiles)
        kept, total = int(mask.sum()), len(mask)

        plt.figure(figsize=(7, 5))
        plt.scatter(xa[mask], ya[mask], alpha=0.5)
        lims = [
            float(np.nanmin([xa[mask].min(), ya[mask].min()])),
            float(np.nanmax([xa[mask].max(), ya[mask].max()]))
        ]
        plt.plot(lims, lims, "--", linewidth=2, label="y = x")
        plt.xlim(lims); plt.ylim(lims)
        plt.xlabel("log1p(Actual Sales)" if log_scale else "Actual Sales")
        plt.ylabel("log1p(Predicted Sales)" if log_scale else "Predicted Sales")
        plt.title(f"Predicted vs Actual (zoomed) — R²={r2:.2f} | kept {kept}/{total}")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

        residuals = y_test.values.astype(float) - y_pred.astype(float)
        xf = y_pred.astype(float)
        if log_scale:
            xf = np.log1p(np.clip(xf, a_min=0, a_max=None))
        m2 = _clip_mask(xf, residuals, clip_quantiles)

        plt.figure(figsize=(7, 5))
        plt.scatter(xf[m2], residuals[m2], alpha=0.5)
        plt.axhline(0.0, linestyle="--", linewidth=1.5)
        plt.xlabel("Fitted (Predicted)" + (" [log1p]" if log_scale else ""))
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted (zoomed)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

        try:
            feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
        except AttributeError:
            num_names = num_cols
            cat_names = []
            if cat_cols:
                ohe = pipe.named_steps["preprocess"].transformers_[1][1]
                cat_names = ohe.get_feature_names_out(cat_cols)
            feature_names = np.array(num_names + list(cat_names))

        coefs = pipe.named_steps["linreg"].coef_.ravel()
        order = np.argsort(np.abs(coefs))[::-1][:top_k_coeffs]
        plt.figure(figsize=(8, max(4, 0.4 * len(order))))
        plt.barh(range(len(order)), coefs[order][::-1])
        plt.yticks(range(len(order)), feature_names[order][::-1])
        plt.title(f"Top {len(order)} Coefficients (|value|)")
        plt.xlabel("Coefficient")
        plt.tight_layout(); plt.show()

    return {
        "LinModel": pipe,
        "metrics_last": {"mse": float(mse), "rmse": float(rmse), "r2": float(r2), "mape": float(mape)},
        "metrics": {
            "model": {"mse": float(mse), "rmse": float(rmse), "r2": float(r2), "mape": float(mape)},
        },
    }


if __name__ == "__main__":
    from data_loader import load_raw_data
    from data_cleaning import clean_data
    df_raw   = load_raw_data("C:\\Users\\stajyer\\Desktop\\GPT\\train.csv")
    df_clean = clean_data(df_raw)
    run_modeling(df_clean, show_plots=True, clip_quantiles=(1, 98), log_scale=True)
