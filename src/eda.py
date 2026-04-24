import matplotlib.pyplot as plt
import seaborn as sns
import os

from preprocessing import load_data, preprocess


def main():
    os.makedirs("reports", exist_ok=True)

    df = load_data("data/cve.csv")
    df = preprocess(df)

    # распределение классов
    plt.figure()
    sns.countplot(x="severity", data=df)
    plt.title("Severity distribution")
    plt.savefig("reports/severity_distribution.png")

    # длина текста
    plt.figure()
    sns.histplot(df["desc_len"], bins=50)
    plt.title("Description length distribution")
    plt.savefig("reports/desc_length.png")

    # CVSS
    plt.figure()
    sns.histplot(df["cvss"], bins=30)
    plt.title("CVSS distribution")
    plt.savefig("reports/cvss_distribution.png")

    print("EDA plots saved!")


if __name__ == "__main__":
    main()