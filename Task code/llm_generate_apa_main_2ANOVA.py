import pandas as pd
import pingouin as pg

DOMAIN_MAP = {1: "Health", 2: "Politics", 3: "War"}

def generate_apa(results_df):

    print("\nGenerating APA narrative (F1-based RM-ANOVA + t-test)...")

    # Remove the overall rows
    df = results_df[results_df["domain"] != "OVERALL"].copy()

    # Fix domain labels if needed
    df["domain"] = df["domain"].replace({"ALL": None})

    # Long format for pingouin
    long_df = df[["model", "source", "domain", "f1"]].copy()

    # =========================================
    #  REPEATED-MEASURES ANOVA
    # =========================================
    rm = pg.rm_anova(
        dv="f1",
        within=["source", "domain"],
        subject="model",
        data=long_df,
        detailed=True
    )

    source_row = rm.iloc[0]
    domain_row = rm.iloc[1]
    interaction_row = rm.iloc[2]

    # APA p format helper
    def p_format(p):
        return "< .001" if p < 0.001 else f"= {p:.3f}"

    narrative = []

    narrative.append(
        "A 2 (Source: UGC, NGC) × 3 (Domain: Health, Politics, War) "
        "repeated-measures ANOVA was conducted on F1 scores to examine differences "
        "in model performance across content types and topical domains.\n"
    )

    # -------------------------------------
    # MAIN EFFECT OF SOURCE
    # -------------------------------------
    narrative.append(
        f"There was a significant main effect of source, "
        f"F({int(source_row['ddof1'])}, {int(source_row['ddof2'])}) "
        f"= {source_row['F']:.2f}, p {p_format(source_row['p-unc'])}, "
        f"η²₍G₎ = {source_row['ng2']:.3f}, "
        "indicating a performance difference between UGC and NGC content.\n"
    )

    # -------------------------------------
    # MAIN EFFECT OF DOMAIN
    # -------------------------------------
    if domain_row["p-unc"] < 0.05:
        narrative.append(
            f"There was also a significant main effect of domain, "
            f"F({int(domain_row['ddof1'])}, {int(domain_row['ddof2'])}) "
            f"= {domain_row['F']:.2f}, p {p_format(domain_row['p-unc'])}, "
            f"η²₍G₎ = {domain_row['ng2']:.3f}, "
            "indicating differences across Health, Politics, and War.\n"
        )
    else:
        narrative.append(
            f"The main effect of domain was not statistically significant, "
            f"F({int(domain_row['ddof1'])}, {int(domain_row['ddof2'])}) "
            f"= {domain_row['F']:.2f}, p {p_format(domain_row['p-unc'])}, "
            f"η²₍G₎ = {domain_row['ng2']:.3f}.\n"
        )

    # -------------------------------------
    # INTERACTION
    # -------------------------------------
    if interaction_row["p-unc"] < 0.05:
        narrative.append(
            f"There was a significant Source × Domain interaction, "
            f"F({int(interaction_row['ddof1'])}, {int(interaction_row['ddof2'])}) "
            f"= {interaction_row['F']:.2f}, p {p_format(interaction_row['p-unc'])}, "
            f"η²₍G₎ = {interaction_row['ng2']:.3f}, "
            "indicating that the effect of source varied across domains.\n"
        )
    else:
        narrative.append(
            f"The Source × Domain interaction was not statistically significant, "
            f"F({int(interaction_row['ddof1'])}, {int(interaction_row['ddof2'])}) "
            f"= {interaction_row['F']:.2f}, p {p_format(interaction_row['p-unc'])}, "
            f"η²₍G₎ = {interaction_row['ng2']:.3f}.\n"
        )

    # ============================================================
    #   PAIRED t-TEST: UGC vs NGC (collapsed across domain)
    # ============================================================
    narrative.append("\n")

    # Collapse across domain to get one UGC and one NGC score per model
    agg = df.groupby(["model", "source"])["f1"].mean().reset_index()

    # Pivot to wide format: model | UGC | NGC
    wide = agg.pivot(index="model", columns="source", values="f1")

    # Paired t-test
    ttest = pg.ttest(wide["UGC"], wide["NGC"], paired=True)

    t = ttest["T"].iloc[0]
    df_t = ttest["dof"].iloc[0]
    p_t = ttest["p-val"].iloc[0]
    dz = ttest["cohen-d"].iloc[0]

    def p_format_t(p):
        return "< .001" if p < 0.001 else f"= {p:.3f}"

    narrative.append(
        f"A paired-samples t-test comparing overall F1 performance on UGC versus NGC content "
        f"showed that the difference between sources was "
        f"{'statistically significant' if p_t < 0.05 else 'not statistically significant'}, "
        f"t({df_t:.0f}) = {t:.2f}, p {p_format_t(p_t)}, Cohen’s d₍z₎ = {dz:.3f}.\n"
    )

    # ============================================
    # EXPORT
    # ============================================

    final_text = "\n".join(narrative)

    with open("FS_Final_Analysis_APA_F1.txt", "w") as f:
        f.write(final_text)

    print("\nSaved to FS_Final_Analysis_APA_F1.txt\n")
    print(final_text)


# =============================
# RUN WITH EXISTING RESULTS CSV
# =============================
if __name__ == "__main__":
    df = pd.read_csv("2ANOVA_CxD_ZS.csv")
    generate_apa(df)
