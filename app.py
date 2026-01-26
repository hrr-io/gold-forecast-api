import os
import importlib
import joblib
from fastapi import FastAPI
import pandas as pd
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

import updater
import model
importlib.reload(updater)
importlib.reload(model)

app = FastAPI(title="7-Day Gold Forecast API")

MODEL = joblib.load("models/xgboost_gold_forecast_model.pkl")
SCALER = joblib.load("models/scaler.pkl")
RAW_DFS, MERGED_DF = updater.daily_update(
                        updater.ASSETS,
                        FRED_API_KEY=os.environ.get("FRED_API_KEY"),
                        OANDA_API_KEY=os.environ.get("OANDA_API_KEY")
                     )

def scheduled_update():
    global RAW_DFS, MERGED_DF
    try:
        RAW_DFS, MERGED_DF = updater.daily_update(
            updater.ASSETS,
            FRED_API_KEY=os.environ.get("FRED_API_KEY"),
            OANDA_API_KEY=os.environ.get("OANDA_API_KEY")
        )
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

@app.get("/predict")
def predict():
    try:
        predictions = model.predict_next_7_days(MODEL, MERGED_DF, SCALER)
        predictions = {k: float(v) for k, v in predictions.items()}
        return predictions
    except Exception as e:
        return {"error": str(e)}

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