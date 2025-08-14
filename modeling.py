import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_modeling(df: pd.DataFrame, show_plots: bool = True):

    def mape_ignore_zeros(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        nz = y_true != 0
        if nz.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz]) * 100))  

    df1 = df.copy()
    df1.columns = df1.columns.str.strip()

    df1_dum = pd.get_dummies(df1, columns=["Category"], drop_first=True)
    needed = ["Category_Office Supplies", "Category_Technology"]
    for col in needed:
        if col not in df1_dum:
            df1_dum[col] = 0
    X = df1_dum[needed]

    y = df1_dum["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    mape = float(mape_ignore_zeros(y_test, y_pred))

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE (%): {mape:.2f}")



    if show_plots:
        ops  = (X_test["Category_Office Supplies"] > 0.5).to_numpy()
        tech = (X_test["Category_Technology"] > 0.5).to_numpy()
        
        cat_labels = np.where(ops, "Office Supplies",
                        np.where(tech, "Technology", "Furniture"))
        
        
        df_plot = pd.DataFrame({
            "Category":  cat_labels,
            "Actual":    y_test.to_numpy(),
            "Predicted": y_pred
        })
        
        grp = df_plot.groupby("Category")[["Actual", "Predicted"]].mean() 

        order_cats = [c for c in ["Furniture", "Office Supplies", "Technology"] if c in grp.index]
        grp = grp.loc[order_cats]

        x = np.arange(len(grp))
        width = 0.38
        plt.figure(figsize=(8, 4.6))
        plt.bar(x - width/2, grp["Actual"].values,  width, label="Actual")
        plt.bar(x + width/2, grp["Predicted"].values, width, label="Predicted")
        plt.xticks(x, order_cats)
        plt.ylabel("Mean Sales (Test)")
        plt.title("Actual vs Predicted by Category (Mean)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "LinModel":  model,
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
    run_modeling(df_clean, show_plots=True)