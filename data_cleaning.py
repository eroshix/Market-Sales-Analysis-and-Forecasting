import pandas as pd
import numpy as np  

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # 1) Missing value check
    df = df.dropna(subset=["Postal Code"]).copy()
    
    # 2) Type conversion
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    # 3) Dropping unnecessary columns
    df.drop(columns=['Row ID', "Customer Name", "Country"], inplace=True)

    # 4) Removing duplicate rows
    df.drop_duplicates(inplace=True)

    # 5) Converting necessary data to categorical data
    for col in ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]:
        df[col] = df[col].astype("category")

    # 6) Setting the Order Date column as the index and sorting
    df.sort_values("Order Date", inplace=True)
    df.set_index("Order Date", inplace=True)

    

    # Outputs
    print(f"[data_cleaning] Cleaned rows/columns: {df.shape}")
    print("--------------------------------------------------------------------------------------------------------------------")
    print(f"[data_cleaning] First five rows: \n{df.head()}")
    print("--------------------------------------------------------------------------------------------------------------------")
    print(f"[data_cleaning] Missing value counts: \n{df.isnull().sum()}")
    print("--------------------------------------------------------------------------------------------------------------------")
    print(f"[data_cleaning] Column information: \n{df.dtypes}")
    print("--------------------------------------------------------------------------------------------------------------------")
    print("[data_cleaning] Unique values:")
    for col in df.select_dtypes(include='category').columns:
        print(f"{col}: {df[col].unique()}")
    print("--------------------------------------------------------------------------------------------------------------------")
    return df


if __name__ == "__main__":
    from data_loader import load_raw_data
    df_raw = load_raw_data("C:\\Users\\stajyer\\Desktop\\Market-Sales-Analysis-and-Forecasting\\train.csv")
    df_clean = clean_data(df_raw)
    print(df_clean.head())