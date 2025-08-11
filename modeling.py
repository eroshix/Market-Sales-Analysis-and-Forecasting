import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def run_modeling(df: pd.DataFrame):

    df = df.dropna(subset=["Postal Code", "Sales"])
    X = df[["Postal Code"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    LinModel = LinearRegression()
    LinModel.fit(X_train, y_train)

    y_pred = LinModel.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R² Score:", r2)

    df.columns = df.columns.str.strip()

    df_encoded = pd.get_dummies(df, columns=["Category"], drop_first=True)

    X = df_encoded[["Category_Office Supplies", "Category_Technology"]]
    y = df_encoded["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    LinModel2 = LinearRegression()
    LinModel2.fit(X_train, y_train)

    y_pred = LinModel2.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R² Score:", r2)


    rmse = np.sqrt(mse)

    def mean_absolute_percentage_error_local(y_test_local, y_pred_local): 
        y_test_local, y_pred_local = np.array(y_test_local), np.array(y_pred_local)
        non_zero_idx = y_test_local != 0
        return np.mean(np.abs((y_test_local[non_zero_idx] - y_pred_local[non_zero_idx]) / y_test_local[non_zero_idx])) * 100

    mape = mean_absolute_percentage_error_local(y_test, y_pred)

    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAPE (%):", mape)

    print(df.columns)

    return {
        "LinModel": LinModel,
        "LinModel2": LinModel2,
        "metrics_last": {"mse": float(mse), "rmse": float(rmse), "r2": float(r2), "mape": float(mape)}
    }

if __name__ == "__main__":
    from data_loader import load_raw_data
    from data_cleaning import clean_data
    df_raw   = load_raw_data("C:\\Users\\stajyer\\Desktop\\GPT\\train.csv")
    df_clean = clean_data(df_raw)
    run_modeling(df_clean)