import pandas as pd
import numpy as np  

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # 1) Boş değer kontrolü
    df = df.dropna(subset=["Postal Code"]).copy()
    
    # 2) Tür dönüşümü
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    # 3) Gereksiz sütunları atma
    df.drop(columns=['Row ID', "Customer Name", "Country"], inplace=True)

    # 4) Tekrar eden satırları kaldırma
    df.drop_duplicates(inplace=True)

    # 5) Gerekli verileri kategorik hale getirme
    for col in ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]:
        df[col] = df[col].astype("category")

    # 6) Order Date sütununu indeks olarak ayarlama ve sıralama
    df.sort_values("Order Date", inplace=True)
    df.set_index("Order Date", inplace=True)

    

    # Çıktılar
    print(f"[data_cleaning] Temizlenmiş satır/sütun: {df.shape}")
    """
    print(f"[data_cleaning] Eksik Değer Sayıları: \n{df.isnull().sum()}")
    
    print(f"[data_cleaning] Sütun bilgisi: \n{df.dtypes}")

    print("[data_cleaning] Unique Değerler:")
    for col in df.select_dtypes(include='category').columns:
        print(f"{col}: {df[col].unique()}")

    """
    

    return df


if __name__ == "__main__":
    from data_loader import load_raw_data
    df_raw = load_raw_data("C:\\Users\\stajyer\\Desktop\\GPT\\train.csv")
    df_clean = clean_data(df_raw)
    print(df_clean.head())
