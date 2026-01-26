import pandas as pd
import time
import yfinance as yf
from fredapi import Fred
from oandapyV20 import API
import oandapyV20.endpoints.instruments as Instruments

def get_yfinance_data(TICKERS, START, END):
    END = (pd.to_datetime(END) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    df_list = []
    for TICKER, col_name in TICKERS.items():
      t = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
      
      if not t.empty and "Close" in t.columns:
        s = t["Close"].squeeze()
        s.name = col_name
        df_list.append(s)
    
      time.sleep(0.5)
    
    df = pd.concat(df_list, axis=1)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.index.name = 'DATE'
    return df

def get_fred_data(SERIES, START, END, API_KEY):
    fred = Fred(api_key=API_KEY)
    df_list = []
    for SID, col_name in SERIES.items():
        s = fred.get_series(SID, observation_start=START, observation_end=END)
        s.name = col_name
        df_list.append(s)
        time.sleep(0.2)

    df = pd.concat(df_list, axis=1)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.index.name = 'DATE'
    return df

def get_oanda_data(INSTRUMENTS, START, END, TOKEN):
    END = (pd.to_datetime(END) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    client = API(access_token=TOKEN)
    df_list = []
    for INSTRUMENT, col_name in INSTRUMENTS.items():
      PARAMS = {
        "from": f"{START}T00:00:00Z",
        "to": f"{END}T23:59:59Z",
        "granularity": "D",
        "price": "M"
      }
      r = Instruments.InstrumentsCandles(instrument=INSTRUMENT, params=PARAMS)
      client.request(r)
      candles = [
        {
            #"Date": c['time'][:10],
            "Date": (pd.to_datetime(c["time"]) + pd.Timedelta(days=1)).date(),
            col_name: float(c['mid']['c'])
        }
        for c in r.response.get('candles', []) if c['complete']
      ]
      df_list.append(pd.DataFrame(candles).set_index('Date'))
      time.sleep(0.5)

    df = pd.concat(df_list, axis=1)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.index.name = 'DATE'
    return df