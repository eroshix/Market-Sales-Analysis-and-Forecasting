import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)          
    print(f"[data_loader] Yüklenen satır/sütun: {df.shape}")
    return df


if __name__ == "__main__":
    df_raw = load_raw_data("C:\\Users\\stajyer\\Desktop\\GPT\\train.csv")
    print(df_raw.head())
