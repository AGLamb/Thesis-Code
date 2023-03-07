import pandas as pd


def get_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    path = '/Users/main/Vault/Thesis/Data/pm25_weer.csv'
    df = get_data(path)

    cutout = len(df)//2
    train_data = df.iloc[:cutout, :].copy()
    test_data = df.iloc[cutout:, :].copy()

    train_data.to_csv("./Data/Core/train_data.csv")
    test_data.to_csv("./Data/Core/test_data.csv")
    return


if __name__ == "__main__":
    main()
