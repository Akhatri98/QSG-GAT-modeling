import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
)
from utils import MASTER_INDEX_FILE, RANDOM_SEED, TRAIN_CUTOFF, VAL_CUTOFF

np.random.seed(RANDOM_SEED)

# AI generated boilerplates in press releases
BOILERPLATE_RE = re.compile(
    "|".join([
        r"this press release (?:does not|shall not) constitute an offer to sell.*",
        r"forward looking statements.*",
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
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", BOILERPLATE_RE.sub(" ", text)).strip()


DOLLAR_RE = re.compile(r"\$\s*([\d,.]+)\s*(billion|million|bn|mn|m|b)\b", re.IGNORECASE)
MULTIPLIERS = {"billion": 1e9, 
               "bn": 1e9, 
               "b": 1e9, 
               "million": 1e6, 
               "mn": 1e6, 
               "m": 1e6}


def extractOfferingSize(text):
    if not isinstance(text, str):
        return np.nan
    m = DOLLAR_RE.search(text)
    if not m:
        return np.nan
    return float(m.group(1).replace(",", "")) * MULTIPLIERS.get(m.group(2).lower(), 1)


def buildScalarFeatures(df):
    out = pd.DataFrame(index=df.index)
    out["headline_word_count"] = df["headline"].fillna("").str.split().str.len()
    out["is_priced"] = df["headline"].str.contains(r"\bpric(?:es|ed)\b", case=False, na=False).astype(int)
    out["is_proposed"] = df["headline"].str.contains(r"\b(?:proposed|announces)\b", case=False, na=False).astype(int)
    out["is_after_close"] = (df["news_session"] == "after_close").astype(int)
    raw = df["headline"].apply(extractOfferingSize).clip(upper=10e9)
    out["offering_size_log"] = np.log1p(raw.fillna(0))
    return out


def buildFeatures(index, trainIdx, testIdx, nHeadline, nBody):
    # TODO: vectorize headline + body, stack with scalar features
    bodyClean = index["body"].apply(stripBoilerplate)

    hVec = TfidfVectorizer(max_features=nHeadline, ngram_range=(1, 2),
                             sublinear_tf=True, min_df=2, stop_words="english")
    hTrain = hVec.fit_transform(index.loc[trainIdx, "headline"].fillna(""))
    hTest = hVec.transform(index.loc[testIdx, "headline"].fillna(""))

    scalar = buildScalarFeatures(index)
    sTrain = csr_matrix(scalar.iloc[trainIdx].values)
    sTest = csr_matrix(scalar.iloc[testIdx].values)

    if nBody > 0:
        bVec   = TfidfVectorizer(max_features=nBody, ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=2, stop_words="english")
        bTrain = bVec.fit_transform(bodyClean.iloc[trainIdx])
        bTest = bVec.transform(bodyClean.iloc[testIdx])
        xTrain = hstack([hTrain, bTrain, sTrain])
        xTest = hstack([hTest,  bTest,  sTest])
    else:
        bVec = None
        xTrain = hstack([hTrain, sTrain])
        xTest = hstack([hTest,  sTest])

    return xTrain, xTest, hVec, bVec


def scoreConfig(xTrain, xTest, yRegTrain, yRegTest, yClsTrain, yClsTest, ridgeAlpha, logitBalanced):
    # TODO: fit ridge + logit, return scores
    ridge = Ridge(alpha=ridgeAlpha, random_state=RANDOM_SEED)
    ridge.fit(xTrain, yRegTrain)
    ridgePreds = ridge.predict(xTest)
    ridgeDirAcc = accuracy_score((yRegTest > 0).astype(int), (ridgePreds > 0).astype(int))

    cw = "balanced" if logitBalanced else None
    logit = LogisticRegression(C=1.0, max_iter=5000, class_weight=cw,
                               random_state=RANDOM_SEED, solver="saga")
    logit.fit(xTrain, yClsTrain)
    logitAcc = accuracy_score(yClsTest, logit.predict(xTest))

    return {
        "ridge_dir_acc": ridgeDirAcc,
        "ridge_r2": r2_score(yRegTest, ridgePreds),
        "ridge_mae": mean_absolute_error(yRegTest, ridgePreds),
        "logit_acc": logitAcc,
        "ridge_model": ridge,
        "logit_model": logit,
        "ridge_preds": ridgePreds,
    }


# sweep configs
FEATURE_CONFIGS = [(100, 0), (100, 100), (150, 150), (150, 500)]
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
LOGIT_BALANCED = [False, True]


def runSweep(index, trainIdx, testIdx, yRegTrain, yRegTest, yClsTrain, yClsTest):
    # sweep 1: try different feature budgets
    print("\nSweep 1: (alpha=1.0, balanced=False)")
    print(f"{'headline':>9s}  {'body':>6s}  {'n_feats':>8s}  {'ridge_dir':>10s}  {'ridge_r2':>9s}  {'logit_acc':>10s}")
    print(f"{'─'*9}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*9}  {'─'*10}")
    featResults = []
    for nH, nB in FEATURE_CONFIGS:
        xTrain, xTest, hVec, bVec = buildFeatures(index, trainIdx, testIdx, nH, nB)
        scores = scoreConfig(xTrain, xTest, yRegTrain, yRegTest, yClsTrain, yClsTest,
                             ridgeAlpha=1.0, logitBalanced=False)
        print(f"{nH:>9d}  {nB:>6d}  {xTrain.shape[1]:>8d}  "
              f"{scores['ridge_dir_acc']:>10.4f}  {scores['ridge_r2']:>9.4f}  {scores['logit_acc']:>10.4f}")
        featResults.append({"n_headline": nH, "n_body": nB, "h_vec": hVec, "b_vec": bVec, **scores})

    bestFeat = max(featResults, key=lambda r: r["logit_acc"])

    # sweep 2: try different ridge alphas with best feature config
    print("\nSweep 2: (best feature config, balanced=False)")
    print(f"{'alpha':>8s}  {'ridge_dir':>10s}  {'ridge_r2':>9s}  {'ridge_mae':>10s}")
    print(f"{'─'*8}  {'─'*10}  {'─'*9}  {'─'*10}")

    xTrain, xTest, hVec, bVec = buildFeatures(index, trainIdx, testIdx,
                                               bestFeat["n_headline"], bestFeat["n_body"])
    alphaResults = []
    for alpha in RIDGE_ALPHAS:
        scores = scoreConfig(xTrain, xTest, yRegTrain, yRegTest, yClsTrain, yClsTest,
                             ridgeAlpha=alpha, logitBalanced=False)
        print(f"{alpha:>8.1f}  {scores['ridge_dir_acc']:>10.4f}  {scores['ridge_r2']:>9.4f}  {scores['ridge_mae']:>10.4f}")
        alphaResults.append({"ridge_alpha": alpha, **scores})

    bestAlpha = max(alphaResults, key=lambda r: r["ridge_dir_acc"])["ridge_alpha"]

    # sweep 3: try balanced vs unbalanced logit
    print("\nSweep 3: (best feature + alpha)")
    print(f"{'balanced':>10s}  {'logit_acc':>10s}  {'up_recall':>10s}  {'up_precision':>13s}")
    print(f"{'─'*10}  {'─'*10}  {'─'*10}  {'─'*13}")

    balancedResults = []
    for balanced in LOGIT_BALANCED:
        scores = scoreConfig(xTrain, xTest, yRegTrain, yRegTest, yClsTrain, yClsTest,
                             ridgeAlpha=bestAlpha, logitBalanced=balanced)
        preds = scores["logit_model"].predict(xTest)
        tp = ((preds == 1) & (yClsTest == 1)).sum()
        fp = ((preds == 1) & (yClsTest == 0)).sum()
        fn = ((preds == 0) & (yClsTest == 1)).sum()
        upRecall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        upPrecision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        print(f"  {str(balanced):>10s}  {scores['logit_acc']:>10.4f}  {upRecall:>10.4f}  {upPrecision:>13.4f}")
        balancedResults.append({"logit_balanced": balanced, "h_vec": hVec, "b_vec": bVec,
                                 "ridge_alpha": bestAlpha, **scores})

    # pick best overall
    best = max(balancedResults, key=lambda r: (r["logit_acc"], r["ridge_dir_acc"]))
    best.update({"n_headline": bestFeat["n_headline"], "n_body": bestFeat["n_body"]})

    print(f"\nBest overall: headline={best['n_headline']}, body={best['n_body']}, "
          f"alpha={best['ridge_alpha']}, balanced={best['logit_balanced']}")
    return best


def fullEval(best, index, trainIdx, testIdx, yRegTrain, yRegTest, yClsTrain, yClsTest):
    # TODO: rebuild features with best config, retrain models, print all metrics + feature importance
    xTrain, xTest, hVec, bVec = buildFeatures(
        index, trainIdx, testIdx, best["n_headline"], best["n_body"])

    ridge = Ridge(alpha=best["ridge_alpha"], random_state=RANDOM_SEED)
    ridge.fit(xTrain, yRegTrain)
    ridgePreds = ridge.predict(xTest)
    
    cw = "balanced" if best["logit_balanced"] else None
    logit = LogisticRegression(C=1.0, max_iter=5000, class_weight=cw,
                               random_state=RANDOM_SEED, solver="saga")
    logit.fit(xTrain, yClsTrain)
    logitPreds = logit.predict(xTest)

    trueDirTrain = (yRegTrain > 0).astype(int)
    trueDirTest = (yRegTest  > 0).astype(int)

    print(f"\nFull evaluation: headline={best['n_headline']}, body={best['n_body']}, "
          f"alpha={best['ridge_alpha']}, balanced={best['logit_balanced']} ==")

    # ridge metrics
    predsTrain = ridge.predict(xTrain)
    print(f"Train MAE: {mean_absolute_error(yRegTrain, predsTrain):.4f}  R2: {r2_score(yRegTrain, predsTrain):.4f}")
    print(f"Test MAE: {mean_absolute_error(yRegTest, ridgePreds):.4f}  R2: {r2_score(yRegTest, ridgePreds):.4f}")
    print(f"Train direction acc: {accuracy_score(trueDirTrain, (predsTrain > 0).astype(int)):.4f}")
    print(f"Test direction acc: {accuracy_score(trueDirTest, (ridgePreds > 0).astype(int)):.4f}")
    print("\nClassification (ridge at 0):")
    print(classification_report(trueDirTest, (ridgePreds > 0).astype(int), target_names=["down", "up"]))

    # logit metrics
    print(f"Train accuracy: {accuracy_score(yClsTrain, logit.predict(xTrain)):.4f}")
    print(f"Test  accuracy: {accuracy_score(yClsTest, logitPreds):.4f}")
    print(f"\nPredicted up: {(logitPreds==1).sum():,} ({(logitPreds==1).mean():.1%})  "
          f"down: {(logitPreds==0).sum():,} ({(logitPreds==0).mean():.1%})")
    print("\nClassification:")
    print(classification_report(yClsTest, logitPreds, target_names=["down", "up"]))
    print("Confusion matrix:")
    cm = confusion_matrix(yClsTest, logitPreds)
    print(f"   Predicted:   down   up")
    print(f"   Actual down: {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"   Actual up:   {cm[1,0]:5d}  {cm[1,1]:5d}")

    # feature importance via coefficients
    nH = best["n_headline"]
    nB = best["n_body"]
    scalarNames = ["headline_word_count", "is_priced", "is_proposed", "is_after_close", "offering_size_log"]
    hInv = {v: k for k, v in hVec.vocabulary_.items()}
    bInv = {v: k for k, v in bVec.vocabulary_.items()} if bVec is not None else {}

    def fname(i):
        if i < nH:
            return f"[headline] {hInv[i]}"
        if i < nH + nB:
            return f"[body] {bInv[i - nH]}"
        return f"[scalar] {scalarNames[i - nH - nB]}"

    print("\n top features by absolute coefficient value:")
    for label, coefs in [("Ridge", ridge.coef_), ("Logistic", logit.coef_[0])]:
        print(f"\n{label}:")
        for rank, idx in enumerate(np.argsort(np.abs(coefs))[::-1][:15], 1):
            print(f"  {rank:2d}. {fname(idx):45s}  coef={coefs[idx]:+.4f}")

    # summary table vs baseline
    baselineDir = np.zeros_like(trueDirTest)
    print("\nSummary of key metrics:")
    print(f"{'Model':25s}  {'Dir. Acc':>10s}  {'MAE':>10s}  {'R2':>10s}")
    print(f"{'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}")
    print(f"{'Baseline (always down)':25s}  "
          f"{accuracy_score(trueDirTest, baselineDir):>10.4f}  "
          f"{mean_absolute_error(yRegTest, np.full_like(yRegTest, yRegTrain.mean())):>10.4f}  {'N/A':>10s}")
    print(f"{'Ridge (thresholded)':25s}  "
          f"{accuracy_score(trueDirTest, (ridgePreds > 0).astype(int)):>10.4f}  "
          f"{mean_absolute_error(yRegTest, ridgePreds):>10.4f}  {r2_score(yRegTest, ridgePreds):>10.4f}")
    print(f"{'Logistic':25s}  "
          f"{accuracy_score(yClsTest, logitPreds):>10.4f}  {'N/A':>10s}  {'N/A':>10s}")


def main():
    index = pd.read_csv(MASTER_INDEX_FILE, parse_dates=["target_date"])
    index = index.sort_values("target_date").reset_index(drop=True)

    isTrain = index["target_date"] <= TRAIN_CUTOFF
    isVal = (index["target_date"] > TRAIN_CUTOFF) & (index["target_date"] <= VAL_CUTOFF)
    isTest = index["target_date"] > VAL_CUTOFF
    
    trainIdx = index[isTrain].index
    valIdx = index[isVal].index
    testIdx = index[isTest].index

    yRegTrain = index.loc[trainIdx, "return"].values
    yRegVal = index.loc[valIdx, "return"].values
    yRegTest = index.loc[testIdx,  "return"].values
    yClsTrain = index.loc[trainIdx, "direction"].values
    yClsVal = index.loc[valIdx, "direction"].values
    yClsTest = index.loc[testIdx,  "direction"].values

    best = runSweep(index, trainIdx, valIdx, yRegTrain, yRegVal, yClsTrain, yClsVal)
    # save locked config
    with open("locked_config.txt", "w") as f:
        f.write(f"headline={best['n_headline']}, body={best['n_body']}, ")
        f.write(f"alpha={best['ridge_alpha']}, balanced={best['logit_balanced']}\n")

    print("\nFinal eval with locked config:")
    print(f"Locked config: headline={best['n_headline']}, body={best['n_body']}, "
          f"alpha={best['ridge_alpha']}, balanced={best['logit_balanced']}")
    fullEval(best, index, trainIdx, testIdx, yRegTrain, yRegTest, yClsTrain, yClsTest)


if __name__ == "__main__":
    main()