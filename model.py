import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def split_train_eval_test(df, train_end_date, eval_end_date):
    df = df.sort_index()

    train_end = pd.to_datetime(train_end_date)
    eval_end  = pd.to_datetime(eval_end_date)

    train_mask = df.index <= train_end
    eval_mask  = (df.index > train_end) & (df.index <= eval_end)
    test_mask  = df.index > eval_end

    return (df.loc[train_mask], df.loc[eval_mask], df.loc[test_mask])

def evaluate_model(model, X, y, name="Dataset", scaler=None, plot=True):
    if scaler is not None:
        X = scaler.transform(X)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = mean_absolute_percentage_error(y, y_pred) * 100
    direction_acc = np.mean(
        np.sign(np.diff(y.values)) == np.sign(np.diff(y_pred))
    ) * 100
    print(f"{name} → RMSE: {rmse:.3f}, MAPE: {mape:.2f}%, DirAcc: {direction_acc:.2f}%")
    if plot:
        plt.figure(figsize=(12,5))
        plt.plot(y.index, y.values, label="Actual", color="blue")
        plt.plot(y.index, y_pred, label="Predicted", color="orange", alpha=0.8)
        plt.title(f"{name} → Actual vs Predicted")
        plt.xlabel("Date")
        plt.ylabel("XAUUSD")
        plt.legend()
        plt.grid(True)
        plt.show()
    metrics = {"RMSE": rmse, "MAPE": mape, "DirAcc": direction_acc}
    return y_pred, metrics

def feature_importance(model, X, top_n=20, plot=True):
    booster = model.get_booster()
    scores = booster.get_score(importance_type='gain')
    imp = pd.Series(scores).reindex(X.columns).fillna(0).sort_values(ascending=False)
    imp_top = imp.head(top_n)
    if plot:
        plt.figure(figsize=(10,6))
        imp_top.plot(kind='barh', color='orange')
        plt.gca().invert_yaxis()
        plt.title(f"Top {top_n} XGBoost Feature Importances (Gain)")
        plt.xlabel("Gain Importance")
        plt.show()
    return imp_top

def build_features(df):
    df = df.copy()
    target = "XAUUSD"
    X = df.drop(columns=[target])
    return X

def build_train_model(X_train, y_train, X_eval, y_eval):
    scaler = None
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, scaler

def predict_next_7_days(MODEL, DF, SCALER=None, days=7):
    X_today = build_features(DF).iloc[[-1]]
    if SCALER:
        X_today = SCALER.transform(X_today)
    last_date = DF.index.max()
    prediction = {}
    current_X = X_today.copy()
    for h in range(1, days + 1):
        next_date = last_date + pd.Timedelta(days=h)
        y_pred = MODEL.predict(current_X)[0]
        prediction[next_date.strftime("%Y-%m-%d")] = float(y_pred)
    return prediction

PLACEHOLDER = [5030.42, 5109.18, 5189.77, 5226.05, 5312.91, 5407.63, 5497.88]