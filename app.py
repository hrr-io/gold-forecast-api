import os
import importlib
import joblib
from fastapi import FastAPI
import pandas as pd
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
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
        push_to_github()
        print(f"[{pd.Timestamp.now()}] Datasets updated successfully.")
    except Exception as e:
        print(f"[{pd.Timestamp.now()}] Error updating datasets: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_update, "interval", hours=6)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

@app.get("/health")
def health():
    return {
        "status": "ok",
        "last_data_date": str(MERGED_DF.index.max()),
        "scheduler_running": scheduler.running
    }

# @app.get("/predict")
# def predict():
#     try:
#         predictions = model.predict_next_7_days(MODEL, MERGED_DF, SCALER)
#         predictions = {k: float(v) for k, v in predictions.items()}
#         return predictions
#     except Exception as e:
#         return {"error": str(e)}

from datetime import timedelta

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

@app.get("/update")
def update():
    try:
        scheduled_update()
        return {"status": "updated", "last_date": str(MERGED_DF.index.max())}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)