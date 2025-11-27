import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

RESULTS_FILE = "8_BERT_cross_content_results.csv"

def generate_plots(df):

    # =======================================================
    # 1) CROSS-CONTENT OVERALL (AVERAGED ACROSS MODELS)
    # =======================================================
    dir_df = df[
        (df["domain"] == "OVERALL") &
        (df["source"] == "OVERALL")
    ].copy()

    # Create direction variable
    dir_df["direction"] = dir_df["train_source"] + "→" + dir_df["test_source"]

    plt.figure(figsize=(7,5))
    sns.barplot(
        data=dir_df,
        x="direction",
        y="f1",
        ci="sd",
        palette="Set2"
    )
    plt.title("Cross-Content Transfer: Overall F1 (Averaged Across Models)")
    plt.ylabel("F1 Score")
    plt.xlabel("Training → Testing Direction")
    plt.tight_layout()
    plt.savefig("CrossContent_Direction_Overall.png", dpi=300)
    plt.close()


    # =======================================================
    # 2) MODEL-SPECIFIC CROSS-CONTENT COMPARISON
    # =======================================================
    plt.figure(figsize=(9,5))
    sns.barplot(
        data=dir_df,
        x="model",
        y="f1",
        hue="direction",
        ci="sd",
        palette="Set1"
    )
    plt.title("Per-Model Cross-Content F1 Performance")
    plt.ylabel("F1 Score")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.legend(title="Direction")
    plt.tight_layout()
    plt.savefig("CrossContent_ByModel.png", dpi=300)
    plt.close()


    # =======================================================
    # 3) DOMAIN-LEVEL CROSS-CONTENT PERFORMANCE
    # =======================================================
    dom_df = df[
        df["domain"].isin(["Health", "Politics", "War"])
    ].copy()

    dom_df["direction"] = dom_df["train_source"] + "→" + dom_df["test_source"]

    plt.figure(figsize=(10,5))
    sns.barplot(
        data=dom_df,
        x="domain",
        y="f1",
        hue="direction",
        ci="sd",
        palette="Set3"
    )
    plt.title("Cross-Content Transfer by Domain")
    plt.ylabel("F1 Score")
    plt.xlabel("Domain")
    plt.tight_layout()
    plt.savefig("CrossContent_ByDomain.png", dpi=300)
    plt.close()

    print("\nSaved:")
    print(" • CrossContent_Direction_Overall.png")
    print(" • CrossContent_ByModel.png")
    print(" • CrossContent_ByDomain.png\n")


def main():
    print("Loading results:", RESULTS_FILE)
    df = pd.read_csv(RESULTS_FILE)
    generate_plots(df)
    print("Done!")


if __name__ == "__main__":
    main()
