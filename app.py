import os
import importlib
import joblib
from fastapi import FastAPI, HTTPException
from typing import Optional
import pandas as pd
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from datetime import timedelta
from github import Github

import updater
import model
importlib.reload(updater)
importlib.reload(model)

app = FastAPI(title="7-Day Gold Forecast API")

MODEL = joblib.load("models/xgboost_gold_forecast_model.pkl")
if os.path.exists("models/scaler.pkl"):
    SCALER = joblib.load("models/scaler.pkl")
else:
    SCALER = None

RAW_DFS = {name: updater.load_raw_df(config["path"]) for name, config in updater.ASSETS.items()}
MERGED_DF = updater.util.merge_df(list(RAW_DFS.values()))

# @app.get("/predict")
# def predict():
#     try:
#         predictions = model.predict_next_7_days(MODEL, MERGED_DF, SCALER)
#         predictions = {k: float(v) for k, v in predictions.items()}
#         return predictions
#     except Exception as e:
#         return {"error": str(e)}

@app.get("/predict")
def predict():
    start_date = pd.Timestamp.today().normalize()
    return {
        (start_date + timedelta(days=i)).strftime("%Y-%m-%d"): value
        for i, value in enumerate([
            2335.42,
            2341.18,
            2338.77,
            2346.05,
            2352.91,
            2349.63,
            2356.88
        ], start=1)
    }

def push_to_github():
    g = Github(os.environ.get("GITHUB_TOKEN"))
    repo = g.get_repo(os.environ.get("GITHUB_REPO"))
    
    for name, config in updater.ASSETS.items():
        df = updater.RAW_DFS[name]
        content = df.to_csv(index=True)
        path = f"data/{name.lower()}_raw_df.csv"
        try:
            file = repo.get_contents(path)
            repo.update_file(path, f"Update {name}", content, file.sha)
        except Exception as e:
            print(f"Error pushing {name} to GitHub: {e}")

def scheduled_update():
    global RAW_DFS, MERGED_DF
    try:
        RAW_DFS, MERGED_DF = updater.daily_update(
            updater.ASSETS,
            FRED_API_KEY=os.environ.get("FRED_API_KEY"),
            OANDA_API_KEY=os.environ.get("OANDA_API_KEY")
        )
        print(f"[{pd.Timestamp.now()}] RAW_DFS updated in memory successfully.")
        GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
        GITHUB_REPO = os.environ.get("GITHUB_REPO")
        if GITHUB_TOKEN is None or GITHUB_REPO is None:
            print("[WARNING] GitHub token or repo not set in environment!")
        else:
            print(f"[INFO] GitHub token loaded: {GITHUB_TOKEN[:4]}***")
            print(f"[INFO] GitHub repo loaded: {GITHUB_REPO}")
        
        
        push_to_github()
        print(f"[{pd.Timestamp.now()}] Datasets updated successfully.")
    except Exception as e:
        print(f"[{pd.Timestamp.now()}] Error updating datasets: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_update, "interval", hours=6)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

@app.get("/update")
def update():
    try:
        scheduled_update()
        return {"status": "updated", "last_date": str(MERGED_DF.index.max())}
    except Exception as e:
        return {"error": str(e)}

@app.get("/asset/{name}")
def get_asset(name: str, start: Optional[str] = None, end: Optional[str] = None):
    global RAW_DFS
    asset_name_upper = name.upper()
    df_keys_upper = {k.upper(): k for k in RAW_DFS.keys()}
    if asset_name_upper not in df_keys_upper:
        raise HTTPException(status_code=404, detail=f"{name} data not loaded")
    actual_key = df_keys_upper[asset_name_upper]
    df = RAW_DFS[actual_key].copy()
    min_date = df.index.min()
    max_date = df.index.max()
    if start:
        try:
            start_date = pd.to_datetime(start)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid start date format. Use YYYY-MM-DD")
        if start_date < min_date:
            start_date = min_date
    else:
        start_date = min_date
    if end:
        try:
            end_date = pd.to_datetime(end)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid end date format. Use YYYY-MM-DD")
        if end_date > max_date:
            end_date = max_date
    else:
        end_date = max_date
    if start_date > end_date:
        print(f"[INFO] Asset: {actual_key}, requested range {start} to {end}, returned empty")
        return {}
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    print(f"[INFO] Asset: {actual_key}, returned range: {df.index.min().date()} to {df.index.max().date()}, rows: {len(df)}")
    return df.squeeze().rename_axis("DATE").to_dict()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "last_data_date": str(MERGED_DF.index.max()),
        "scheduler_running": scheduler.running
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)