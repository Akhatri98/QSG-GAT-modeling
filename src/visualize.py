import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from utils import MASTER_INDEX_FILE, NOTE_DIR, RANDOM_SEED, TRAIN_CUTOFF

np.random.seed(RANDOM_SEED)

PLOTS_DIR = NOTE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
})
BLUE   = "#0051FF"
RED    = "#FF0000"
GREEN  = "#0DFF00"
GRAY   = "#757575"
ORANGE = "#FF5900"

# AI generated boilerplates in press releases
BOILERPLATE_RE = re.compile(
    "|".join([
        r"this press release (?:does not|shall not) constitute an offer to sell.*",
        r"forward.looking statements.*",
        r"a registration statement.*",
        r"a prospectus supplement.*",
        r"copies of the prospectus.*",
        r"for more (?:detailed|additional) information.*",
        r"about .{0,60}(?:inc|ltd|corp|group|therapeutics|systems|technologies)\..*",
        r"http\S+", r"&quot;",
        r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
    ]),
    flags=re.IGNORECASE | re.DOTALL,
)


def stripBoilerplate(text):
    if not isinstance(text, str): return ""
    return re.sub(r"\s+", " ", BOILERPLATE_RE.sub(" ", text)).strip()


DOLLAR_RE = re.compile(r"\$\s*([\d,.]+)\s*(billion|million|bn|mn|m|b)\b", re.IGNORECASE)
MULTIPLIERS = {"billion": 1e9, 
               "bn": 1e9, 
               "b": 1e9, 
               "million": 1e6, 
               "mn": 1e6, 
               "m": 1e6}


def extractOfferingSize(text):
    if not isinstance(text, str): return np.nan
    m = DOLLAR_RE.search(text)
    if not m: return np.nan
    return float(m.group(1).replace(",", "")) * MULTIPLIERS.get(m.group(2).lower(), 1)


def buildScalarFeatures(df):
    out = pd.DataFrame(index=df.index)
    out["headline_word_count"] = df["headline"].fillna("").str.split().str.len()
    out["is_priced"] = df["headline"].str.contains(r"\bpric(?:es|ed)\b",          case=False, na=False).astype(int)
    out["is_proposed"] = df["headline"].str.contains(r"\b(?:proposed|announces)\b", case=False, na=False).astype(int)
    out["is_after_close"] = (df["news_session"] == "after_close").astype(int)
    raw = df["headline"].apply(extractOfferingSize).clip(upper=10e9)
    out["offering_size_log"] = np.log1p(raw.fillna(0))
    return out


def buildFeatures(index, trainIdx, testIdx, nHeadline=100, nBody=100):
    # TODO: tfidf on headlines + body, stack with scalars
    bodyClean = index["body"].apply(stripBoilerplate)
    hVec = TfidfVectorizer(max_features=nHeadline, ngram_range=(1, 2),
                             sublinear_tf=True, min_df=2, stop_words="english")
    hTrain = hVec.fit_transform(index.loc[trainIdx, "headline"].fillna(""))
    hTest = hVec.transform(index.loc[testIdx, "headline"].fillna(""))
    bVec = TfidfVectorizer(max_features=nBody, ngram_range=(1, 2),
                             sublinear_tf=True, min_df=2, stop_words="english")
    bTrain = bVec.fit_transform(bodyClean.iloc[trainIdx])
    bTest = bVec.transform(bodyClean.iloc[testIdx])
    scalar = buildScalarFeatures(index)
    sTrain = csr_matrix(scalar.iloc[trainIdx].values)
    sTest = csr_matrix(scalar.iloc[testIdx].values)
    xTrain = hstack([hTrain, bTrain, sTrain])
    xTest = hstack([hTest,  bTest,  sTest])
    return xTrain, xTest, hVec, bVec


def plotReturnDistribution(index, trainMask):
    # TODO: histogram of returns, train vs test side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    bins = np.linspace(-0.6, 0.6, 60)
    for ax, (label, mask), color in zip(
        axes,
        [("Train (2021-2022)", trainMask), ("Test (2024)", ~trainMask)],
        [BLUE, RED]
    ):
        data = index.loc[mask, "return"].clip(-0.6, 0.6)
        ax.hist(data, bins=bins, color=color, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--", label="zero")
        ax.axvline(data.mean(), color=ORANGE, linewidth=1.5, linestyle="-",
                   label=f"mean={data.mean():.3f}")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Open-to-close return (clipped at ±60%)")
        ax.set_ylabel("Count")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.suptitle("Return Distribution (Train vs Test)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "return_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plotReturnBySession(index, trainMask):
    # TODO: boxplot of returns split by news session
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (label, mask), color in zip(
        axes,
        [("Train", trainMask), ("Test", ~trainMask)],
        [BLUE, RED]
    ):
        split = index[mask].copy()
        sessions = ["pre_open", "after_close"]
        data = [split.loc[split["news_session"] == s, "return"].clip(-0.6, 0.6) for s in sessions]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2))
        for patch, c in zip(bp["boxes"], [BLUE, RED]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Pre-open\n(same day)", "After-close\n(next day)"])
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{label}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Open-to-close return")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        for i, (d, s) in enumerate(zip(data, sessions), 1):
            ax.text(i, d.mean() + 0.02, f"μ={d.mean():.3f}", ha="center", fontsize=8, color="black")
    fig.suptitle("Return by News Session", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "return_by_session.png", bbox_inches="tight")
    plt.close(fig)


def plotDirectionByYear(index):
    # bar chart showing up vs down counts per year
    temp = index.copy()
    temp["year"] = pd.to_datetime(temp["target_date"]).dt.year
    grouped = temp.groupby("year")["direction"].value_counts().unstack(fill_value=0)
    grouped.columns = ["Down (0)", "Up (1)"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(grouped))
    w = 0.35
    ax.bar(x - w/2, grouped["Down (0)"], w, label="Down", color=RED,   alpha=0.8)
    ax.bar(x + w/2, grouped["Up (1)"],   w, label="Up",   color=GREEN, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index.astype(str))
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.set_title("Direction Balance by Year", fontsize=13, fontweight="bold")
    for i, (_, row) in enumerate(grouped.iterrows()):
        total = row.sum()
        pct = row["Up (1)"] / total
        ax.text(i, total + 3, f"{pct:.0%} up", ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "direction_balance.png", bbox_inches="tight")
    plt.close(fig)


def plotConfusionMatrices(ridgePreds, logitPreds, yRegTest, yClsTest):
    # TODO: normalized confusion matrix for ridge + logit
    trueDir = (yRegTest > 0).astype(int)
    ridgeDir = (ridgePreds > 0).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (label, preds, truth) in zip(
        axes,
        [("Ridge (threshold=0)", ridgeDir, trueDir), ("Logistic", logitPreds, yClsTest)]
    ):
        cm = confusion_matrix(truth, preds)
        norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Down", "Pred Up"])
        ax.set_yticklabels(["Act Down", "Act Up"])
        ax.set_title(label, fontsize=11, fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({norm[i,j]:.0%})",
                        ha="center", va="center",
                        color="white" if norm[i,j] > 0.6 else "black", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion Matrices (Test)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def plotTopFeatures(ridgeModel, logitModel, hVec, bVec, nH=100, nB=100):
    # bar chart of top 15 coefficients for ridge and logit
    scalarNames = ["headline_word_count", "is_priced", "is_proposed", "is_after_close", "offering_size_log"]
    hInv = {v: k for k, v in hVec.vocabulary_.items()}
    bInv = {v: k for k, v in bVec.vocabulary_.items()}

    def fname(i):
        if i < nH:       
            return f"[H] {hInv[i]}"
        if i < nH + nB:  
            return f"[B] {bInv[i - nH]}"
        return f"[S] {scalarNames[i - nH - nB]}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (label, coefs) in zip(axes, [("Ridge", ridgeModel.coef_), ("Logistic", logitModel.coef_[0])]):
        n = 15
        top = np.argsort(np.abs(coefs))[::-1][:n]
        vals = coefs[top]
        labs = [fname(i) for i in top]
        colors = [GREEN if v > 0 else RED for v in vals]
        yPos = np.arange(n)
        ax.barh(yPos, vals[::-1], color=colors[::-1], alpha=0.85, edgecolor="white")
        ax.set_yticks(yPos)
        ax.set_yticklabels(labs[::-1], fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{label} - Top {n} |coef|", fontsize=11, fontweight="bold")
        ax.set_xlabel("Coefficient value")
    fig.suptitle("Feature Coefficients  ([H]=headline, [B]=body, [S]=scalar)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "top_features.png", bbox_inches="tight")
    plt.close(fig)


def plotPredictedVsActual(ridgePreds, yRegTest):
    # scatter: predicted vs actual return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ridgePreds, yRegTest, alpha=0.25, s=15, color=BLUE, edgecolors="none")
    lims = [-0.6, 0.6]
    ax.axhline(0, color=GRAY, linewidth=0.5, linestyle=":")
    ax.axvline(0, color=GRAY, linewidth=0.5, linestyle=":")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Predicted return")
    ax.set_ylabel("Actual return")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Ridge: Predicted vs Actual Return (Test)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "predicted_vs_actual.png", bbox_inches="tight")
    plt.close(fig)


def main():
    index = pd.read_csv(MASTER_INDEX_FILE, parse_dates=["target_date"])
    index = index.sort_values("target_date").reset_index(drop=True)

    isTrain = index["target_date"] <= TRAIN_CUTOFF
    trainIdx = index[isTrain].index
    testIdx = index[~isTrain].index
    trainMask = isTrain

    yRegTrain = index.loc[trainIdx, "return"].values
    yRegTest = index.loc[testIdx, "return"].values
    yClsTrain = index.loc[trainIdx, "direction"].values
    yClsTest = index.loc[testIdx, "direction"].values

    xTrain, xTest, hVec, bVec = buildFeatures(index, trainIdx, testIdx)

    ridge = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    ridge.fit(xTrain, yRegTrain)
    ridgePreds = ridge.predict(xTest)

    logit = LogisticRegression(C=1.0, max_iter=5000, random_state=RANDOM_SEED, solver="saga")
    logit.fit(xTrain, yClsTrain)
    logitPreds = logit.predict(xTest)

    plotReturnDistribution(index, trainMask)
    plotReturnBySession(index, trainMask)
    plotDirectionByYear(index)
    plotConfusionMatrices(ridgePreds, logitPreds, yRegTest, yClsTest)
    plotTopFeatures(ridge, logit, hVec, bVec)
    plotPredictedVsActual(ridgePreds, yRegTest)


if __name__ == "__main__":
    main()