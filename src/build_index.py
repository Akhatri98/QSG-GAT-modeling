
import pandas as pd
from utils import (
    OFFERINGS_FILES, PRICES_FILE, MASTER_INDEX_FILE, 
    MARKET_OPEN_H, MARKET_OPEN_M, MARKET_CLOSE_H, MARKET_CLOSE_M,
)


def loadPrices(path):
    # TODO: read tsv, parse dates, sort by symbol + date
    out = pd.read_csv(path, sep="\t", parse_dates=["date"])
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def buildNextTradingDay(prices):
    # TODO: for each ticker, map date -> next trading date
    lookup = {}
    for sym, grp in prices.groupby("symbol"):
        s = grp["date"].sort_values()
        nxt = s.shift(-1)
        for d, nd in zip(s.values, nxt.values):
            if pd.isna(nd):
                continue
            lookup[(sym, d)] = nd
    return lookup


def loadOfferings(paths):
    # TODO: read all offerings tsvs, concat, drop bad timestamps
    frames = []
    for p in paths:
        temp = pd.read_csv(p, sep="\t")
        temp["source_file"] = p.name
        frames.append(temp)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])  # parse timestamps
    n = len(out)
    out = out.dropna(subset=["timestamp"])
    if n - len(out):
        print(f"Removed {n - len(out)} rows from offerings")
    return out


def assignTargetDate(offerings, nextDayLookup):
    # TODO: drop intraday rows, label session, figure out target date
    rows = []
    nIntraday = 0
    nNoNext = 0

    for _, row in offerings.iterrows():
        ts = row["timestamp"]
        sym = row["symbol"]
        calDate = ts.date()

        # skip anything during market hours
        if (MARKET_OPEN_H, MARKET_OPEN_M) <= (ts.hour, ts.minute) < (MARKET_CLOSE_H, MARKET_CLOSE_M):
            nIntraday += 1
            continue

        if (ts.hour, ts.minute) < (MARKET_OPEN_H, MARKET_OPEN_M):
            session = "pre_open"
            target = calDate
        else:
            # after close so use next trading day
            session = "after_close"
            key = (sym, calDate)
            if key not in nextDayLookup:
                nNoNext += 1
                continue
            target = nextDayLookup[key]

        rows.append({**row.to_dict(), "news_date": calDate, "news_session": session, "target_date": target})

    print(f"Removed {nIntraday:,} rows by intraday")
    print(f"Removed {nNoNext:,} rows by no next day")
    return pd.DataFrame(rows)


def attachLabels(filtered, prices):
    # TODO: join open/close prices on target date, compute return + direction label
    priceLookup = prices.set_index(["symbol", "date"])[["open", "high", "low", "close", "volume"]]
    out = []
    nMissing = 0

    for _, row in filtered.iterrows():
        key = (row["symbol"], row["target_date"])
        if key not in priceLookup.index:
            nMissing += 1
            continue
        px = priceLookup.loc[key]
        ret = (px["close"] - px["open"]) / px["open"]
        out.append({
            **row.to_dict(),
            "open_px": px["open"],
            "close_px": px["close"],
            "high_px": px["high"],
            "low_px": px["low"],
            "volume": px["volume"],
            "return": ret,
            "direction": int(ret > 0)
        })

    print(f"Removed {nMissing:,} rows by no price")
    return pd.DataFrame(out)


def sanityCheck(index):
    # TODO: assert no intraday leaks, bad dates, or nan returns
    index["tsCheck"] = pd.to_datetime(index["timestamp"])
    leaks = index[index["tsCheck"].apply(
        lambda ts: (MARKET_OPEN_H, MARKET_OPEN_M) <= (ts.hour, ts.minute) < (MARKET_CLOSE_H, MARKET_CLOSE_M)
    )]
    assert len(leaks) == 0, f"ERROR: {len(leaks)} intraday rows still in index"

    badDates = index[pd.to_datetime(index["target_date"]) < pd.to_datetime(index["news_date"])]
    assert len(badDates) == 0, f"ERROR: {len(badDates)} rows where target < news date"

    assert index["return"].isna().sum() == 0, "ERROR: NaN returns in index"

    # print some stats
    print(f"\nReturn stats:\n{index['return'].describe().round(4)}")
    print(f"\nDirection balance:\n{index['direction'].value_counts().to_string()}")
    print(f"\nRows per year:")
    index["year"] = pd.to_datetime(index["target_date"]).dt.year
    print(index["year"].value_counts().sort_index().to_string())
    print(f"\nTotal symbols: {index['symbol'].nunique():,}")
    index.drop(columns=["tsCheck", "year"], inplace=True)
    


def main():
    prices = loadPrices(PRICES_FILE)
    nextDay = buildNextTradingDay(prices)
    offerings = loadOfferings(OFFERINGS_FILES)

    filtered = assignTargetDate(offerings, nextDay)
    index = attachLabels(filtered, prices)

    # convert to strings before saving
    index["timestamp"] = index["timestamp"].astype(str)
    index["target_date"] = index["target_date"].astype(str)
    index["news_date"] = index["news_date"].astype(str)

    sanityCheck(index)

    # reorder cols for readability
    frontCols = [
        "target_date", "news_date", "news_session",
        "symbol", "timestamp",
        "open_px", "close_px", "return", "direction",
        "headline", "body", "source_file",
    ]
    remaining = [c for c in index.columns if c not in frontCols]
    index = index[frontCols + remaining]

    index.to_csv(MASTER_INDEX_FILE, index=False)


if __name__ == "__main__":
    main()