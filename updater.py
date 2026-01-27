import importlib
import pandas as pd

import sys
sys.path.append("src")

import fetch, util
importlib.reload(fetch)
importlib.reload(util)

def load_raw_df(PATH):
    df = pd.read_csv(PATH, index_col='DATE', parse_dates=['DATE'])
    df.sort_index(inplace=True)
    return df

def append_raw_data(EXISTING_DF, NEW_DF):
    if NEW_DF is None or NEW_DF.empty:
        return EXISTING_DF
    NEW_DF.index = pd.to_datetime(NEW_DF.index)
    df = pd.concat([EXISTING_DF, NEW_DF])
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df

def save_raw_df(DF, PATH):
    DF.to_csv(PATH, index=True, index_label="DATE")

def fetch_new_data(SOURCE, IDENTIFIER, START, END, FRED_API_KEY, OANDA_API_KEY):
    if SOURCE == "oanda":
        return fetch.get_oanda_data(INSTRUMENTS=IDENTIFIER, START=START, END=END, TOKEN=OANDA_API_KEY)
    if SOURCE == "fred":
        return fetch.get_fred_data(SERIES=IDENTIFIER, START=START, END=END, API_KEY=FRED_API_KEY)
    if SOURCE == "yfinance":
        return fetch.get_yfinance_data(TICKERS=IDENTIFIER, START=START, END=END)
    raise ValueError(f"Unknown source: {SOURCE}")

def update_asset(NAME, DF, CONFIG, FRED_API_KEY, OANDA_API_KEY):
    last_date = DF.index.max() + pd.Timedelta(days=1)
    today = pd.Timestamp.today().normalize()
    new_df = fetch_new_data(
        SOURCE=CONFIG["source"],
        IDENTIFIER={CONFIG["identifier"] : NAME},
        START=last_date.strftime("%Y-%m-%d"),
        END=today.strftime("%Y-%m-%d"),
        FRED_API_KEY=FRED_API_KEY,
        OANDA_API_KEY=OANDA_API_KEY
    )
    DF = append_raw_data(DF, new_df)
    save_raw_df(DF, CONFIG["path"])
    return DF

def daily_update(ASSETS, FRED_API_KEY, OANDA_API_KEY):
    RAW_DFS = {
        name: load_raw_df(config["path"])
        for name, config in ASSETS.items()
    }
    for name, config in ASSETS.items():
        RAW_DFS[name] = update_asset(
            name, RAW_DFS[name], config, FRED_API_KEY, OANDA_API_KEY
        )
    MERGED_DF = util.merge_df(list(RAW_DFS.values()))
    return RAW_DFS, MERGED_DF

ASSETS = {
    "XAUUSD": {
        "source": "oanda",
        "identifier": "XAU_USD",
        "path": "data/xauusd_raw_df.csv"
    },
    "WTICO": {
        "source": "oanda",
        "identifier": "WTICO_USD",
        "path": "data/wtico_raw_df.csv"
    },
    "DFII10": {
        "source": "fred",
        "identifier": "DFII10",
        "path": "data/dfii10_raw_df.csv"
    },
    "T10YIE": {
        "source": "fred",
        "identifier": "T10YIE",
        "path": "data/t10yie_raw_df.csv"
    },
    "DXY": {
        "source": "yfinance",
        "identifier": "DX-Y.NYB",
        "path": "data/dxy_raw_df.csv"
    },
    "SPX": {
        "source": "yfinance",
        "identifier": "^GSPC",
        "path": "data/spx_raw_df.csv"
    },
    "VIX": {
        "source": "yfinance",
        "identifier": "^VIX",
        "path": "data/vix_raw_df.csv"
    },
    "GVZ": {
        "source": "yfinance",
        "identifier": "^GVZ",
        "path": "data/gvz_raw_df.csv"
    },
    "USDBDT": {
        "source": "yfinance",
        "identifier": "USDBDT=X",
        "path": "data/usdbdt_raw_df.csv"
    }
}