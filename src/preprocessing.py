import pandas as pd


def load_data(path: str):
    df = pd.read_csv(path)
    return df


def create_target(df: pd.DataFrame):
    def map_severity(score):
        if score < 4:
            return 0  # LOW
        elif score < 7:
            return 1  # MEDIUM
        elif score < 9:
            return 2  # HIGH
        else:
            return 3  # CRITICAL

    df = df.dropna(subset=["cvss", "summary"])
    df["severity"] = df["cvss"].apply(map_severity)

    return df


def preprocess(df: pd.DataFrame):
    print("Initial shape:", df.shape)

    df = df.drop_duplicates()

    df = create_target(df)

    # текст
    df["summary"] = df["summary"].fillna("")

    # фичи
    df["desc_len"] = df["summary"].apply(len)

    keywords = ["remote", "execute", "overflow", "denial", "privilege"]

    for word in keywords:
        df[f"has_{word}"] = df["summary"].str.contains(word, case=False).astype(int)

    # дата
    df["year"] = pd.to_datetime(df["pub_date"], errors="coerce").dt.year
    df["year"] = df["year"].fillna(df["year"].median())

    print("Final shape:", df.shape)

    return df