import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack, save_npz, csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import (
    MASTER_INDEX_FILE, FEATURES_FILE, META_FILE,
    CLEAN_DIR, TRAIN_CUTOFF, RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)

# AI generated boilerplates in press releases
BOILERPLATE_PATTERNS = [
    r"this press release (does not|shall not) constitute an offer to sell.*",
    r"forward.looking statements.*",
    r"a registration statement.*",
    r"a prospectus supplement.*",
    r"copies of the prospectus.*",
    r"for more (detailed|additional) information.*",
    r"about .{0,60}(inc|ltd|corp|group|therapeutics|systems|technologies)\..*",
    r"http\S+",
    r"&quot;",
    r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
]
BOILERPLATE_RE = re.compile(
    "|".join(BOILERPLATE_PATTERNS),
    flags=re.IGNORECASE | re.DOTALL,
)


def stripBoilerplate(text):
    # remove noise from body text
    if not isinstance(text, str):
        return ""
    out = BOILERPLATE_RE.sub(" ", text)
    out = re.sub(r"\s+", " ", out).strip()
    return out


DOLLAR_RE = re.compile(
    r"\$\s*([\d,.]+)\s*(billion|million|bn|mn|m|b)\b",
    flags=re.IGNORECASE,
)
MULTIPLIERS = {"billion": 1e9, 
               "bn": 1e9, 
               "b": 1e9, 
               "million": 1e6, 
               "mn": 1e6, 
               "m": 1e6}


def extractOfferingSize(text):
    # TODO: find first dollar amount, return as raw float
    if not isinstance(text, str):
        return np.nan
    m = DOLLAR_RE.search(text)
    if not m:
        return np.nan
    amt = float(m.group(1).replace(",", ""))
    mult = MULTIPLIERS.get(m.group(2).lower(), 1)
    return amt * mult


def buildScalarFeatures(df):
    # TODO: hand-crafted features from headline text + session
    out = pd.DataFrame(index=df.index)
    out["headline_word_count"] = df["headline"].fillna("").str.split().str.len()
    out["is_priced"] = df["headline"].str.contains(r"\bpric(?:es|ed)\b", case=False, na=False).astype(int)
    out["is_proposed"] = df["headline"].str.contains(r"\b(?:proposed|announces)\b", case=False, na=False).astype(int)
    out["is_after_close"] = (df["news_session"] == "after_close").astype(int)
    rawSize = df["headline"].apply(extractOfferingSize)
    rawSize = rawSize.clip(upper=10e9)  # cap at 10B, higher amounts skew results
    out["offering_size_log"] = np.log1p(rawSize.fillna(0))
    return out


def main():
    index = pd.read_csv(MASTER_INDEX_FILE, parse_dates=["target_date"])
    index = index.sort_values("target_date").reset_index(drop=True)

    # split by cutoff, fit only on train
    isTrain = index["target_date"] <= TRAIN_CUTOFF
    trainIdx = index[isTrain].index
    testIdx = index[~isTrain].index

    # TODO: fit tfidf on train headlines only
    hVec = TfidfVectorizer(
        max_features=150, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, stop_words="english",
    )
    hTrain = hVec.fit_transform(index.loc[trainIdx, "headline"].fillna(""))
    hTest = hVec.transform(index.loc[testIdx, "headline"].fillna(""))

    # TODO: same for body text
    bodyClean = index["body"].apply(stripBoilerplate)
    bVec = TfidfVectorizer(
        max_features=150, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, stop_words="english",
    )
    bTrain = bVec.fit_transform(bodyClean.iloc[trainIdx])
    bTest = bVec.transform(bodyClean.iloc[testIdx])

    # scalar features
    scalar = buildScalarFeatures(index)
    sTrain = csr_matrix(scalar.iloc[trainIdx].values)
    sTest = csr_matrix(scalar.iloc[testIdx].values)

    # stack all features together
    xAll = vstack([
        hstack([hTrain, bTrain, sTrain]),
        hstack([hTest,  bTest,  sTest]),
    ])
    save_npz(str(FEATURES_FILE), xAll.tocsr())

    # save vocab so train script can label features
    hVocab = pd.DataFrame(list(hVec.vocabulary_.items()), columns=["term", "index"]).sort_values("index").reset_index(drop=True)
    bVocab = pd.DataFrame(list(bVec.vocabulary_.items()), columns=["term", "index"]).sort_values("index").reset_index(drop=True)
    hVocab.to_csv(CLEAN_DIR / "vocab_headline.csv", index=False)
    bVocab.to_csv(CLEAN_DIR / "vocab_body.csv", index=False)

    # save metadata
    metaTrain = index.loc[trainIdx, ["target_date", "symbol", "return", "direction", "news_session"]].copy()
    metaTest = index.loc[testIdx, ["target_date", "symbol", "return", "direction", "news_session"]].copy()
    metaTrain["is_train"] = 1
    metaTest["is_train"] = 0
    meta = pd.concat([metaTrain, metaTest], ignore_index=True)
    meta.to_csv(META_FILE, index=False)


if __name__ == "__main__":
    main()