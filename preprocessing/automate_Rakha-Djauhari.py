import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def run_preprocessing():
    print("Memulai proses otomatisasi preprocessing...")

    raw_path = "mobile_price_raw/train.csv"
    processed_dir = "preprocessing/mobile_price_preprocessing"
    processed_path = os.path.join(processed_dir, "clean_mobile_price.csv")

    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(raw_path)

    df_clean_initial = df.dropna().drop_duplicates()

    X = df_clean_initial.drop("price_range", axis=1)
    y = df_clean_initial["price_range"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    df_final = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

    df_final.to_csv(processed_path, index=False)
    print(f"Preprocessing selesai! Data disimpan di: {processed_path}")


if __name__ == "__main__":
    run_preprocessing()
