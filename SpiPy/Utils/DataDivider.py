from pandas import DataFrame, read_csv


def get_data(path: str) -> DataFrame:
    """
    :param path: filepath to the raw data
    :return: dataframe with raw data
    """
    return read_csv(path)


def main():
    path = '/Users/main/Vault/Thesis/Data/pm25_weer.csv'
    df = get_data(path)

    cutout = len(df)//2
    train_data = df.iloc[:cutout, :].copy()
    test_data = df.iloc[cutout:, :].copy()

    train_data.to_csv("./Data/train_data.csv")
    test_data.to_csv("./Data/test_data.csv")
    return


if __name__ == "__main__":
    main()
