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

def pick_col(df, options):
    df_cols_lower = [c.lower() for c in df.columns]
    for c in options:
        if c.lower() in df_cols_lower:
            return df.columns[df_cols_lower.index(c.lower())]
    return None

def add_features(df):
    df = df.copy()
    if "DATE" not in df.columns:
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "DATE"}, inplace=True)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)

    col_gold = pick_col(df, ["xauusd", "Gold_Price"])
    col_oil  = pick_col(df, ["wtico", "Oil_Price"])
    col_usd  = pick_col(df, ["dxy", "Dollar Strength", "Dollar_Strength"])
    col_fear = pick_col(df, ["vix", "Stock_Fear_Index"])
    col_spx  = pick_col(df, ["spx", "S&P500 Price", "SPX", "S&P500"])
    col_real = pick_col(df, ["dfii10", "Yield10Inf", "Real_Interest_Rate"])
    col_bre  = pick_col(df, ["t10yie", "Expected_Inflation"])

    if col_bre is None:
        col_nominal10 = pick_col(df, ["Yield10", "DGS10", "Yield_10Y", "10Y_Nominal"])
        if col_nominal10 and col_real:
            df["breakeven_10y"] = df[col_nominal10] - df[col_real]
            col_bre = "breakeven_10y"
        else:
            raise ValueError("Cannot derive breakeven inflation.")

    df["gold_lr"] = np.log(df[col_gold]).diff()
    df["oil_lr"]  = np.log(df[col_oil]).diff()
    df["usd_lr"]  = np.log(df[col_usd]).diff()
    df["spx_lr"]  = np.log(df[col_spx]).diff()
    df["real_d"]  = df[col_real].diff()
    df["bre_d"]   = df[col_bre].diff()
    df["fear_d"]  = df[col_fear].diff()

    df["y"] = df["gold_lr"].shift(-1)
    df["y_date"] = df["DATE"].shift(-1)

    def rule_score(r):
        s = 0
        s += 1 if r["oil_lr"] > 0 else -1
        s += -1 if r["usd_lr"] > 0 else 1
        s += -1 if r["real_d"] > 0 else 1
        s += 1 if r["bre_d"] > 0 else -1
        s += 1 if r["fear_d"] > 0 else -1
        s += -1 if r["spx_lr"] > 0 else 1
        return s

    df["rule_score"] = df.apply(rule_score, axis=1)
    df["rule_dir"] = np.sign(df["rule_score"])
    df["rule_strength"] = df["rule_score"].abs()

    lags = [1,2,3,5,10,20]
    for L in lags:
        for c in ["gold_lr","oil_lr","usd_lr","spx_lr","real_d","bre_d","fear_d"]:
            df[f"{c}_lag{L}"] = df[c].shift(L)

    df["gold_lr_rollstd20"]  = df["gold_lr"].shift(1).rolling(20).std()
    df["gold_lr_rollmean5"]  = df["gold_lr"].shift(1).rolling(5).mean()

    return df, col_gold

def build_features(df):
    feature_cols = [
        "rule_score", "rule_dir", "rule_strength",
        "oil_lr", "usd_lr", "real_d", "bre_d", "fear_d", "spx_lr",
        "gold_lr_rollstd20", "gold_lr_rollmean5"
    ]
    lags = [1,2,3,5,10,20]
    feature_cols += [c for c in df.columns if any(c.endswith(f"_lag{L}") for L in lags)]
    return df[feature_cols], feature_cols

def build_train_model(df):
    df, col_gold = add_features(df)
    X, feature_cols = build_features(df)
    model_df = df.dropna(subset=feature_cols + ["y", col_gold]).copy()

    split_idx = int(len(model_df) * 0.90)
    train_df = model_df.iloc[:split_idx].copy()
    test_df  = model_df.iloc[split_idx:].copy()

    pos_mean = train_df.loc[train_df["y"]>0,"y"].mean()
    neg_mean = train_df.loc[train_df["y"]<0,"y"].mean()
    pos_mean = 0.0 if pd.isna(pos_mean) else pos_mean
    neg_mean = 0.0 if pd.isna(neg_mean) else neg_mean

    train_df.loc[:, "rule_pred_y"] = train_df["rule_dir"].map(lambda d: pos_mean if d>0 else neg_mean if d<0 else 0)
    test_df.loc[:, "rule_pred_y"]  = test_df["rule_dir"].map(lambda d: pos_mean if d>0 else neg_mean if d<0 else 0)

    feature_cols_plus = feature_cols + ["rule_pred_y"]

    mono = []
    for f in feature_cols_plus:
        if f=="oil_lr": mono.append(1)
        elif f=="usd_lr": mono.append(-1)
        elif f=="real_d": mono.append(-1)
        elif f=="bre_d": mono.append(1)
        elif f=="fear_d": mono.append(1)
        elif f=="spx_lr": mono.append(-1)
        else: mono.append(0)

    model = XGBRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        monotone_constraints=tuple(mono)
    )

    model.fit(train_df[feature_cols_plus], train_df["y"])

    return model, feature_cols_plus, col_gold, train_df, test_df

def predict_next_day(model, df, feature_cols, col_gold):
    df, _ = add_features(df)
    latest = df.dropna().iloc[-1]
    X = latest[feature_cols].values.reshape(1,-1)
    pred_lr = model.predict(X)[0]
    last_price = latest[col_gold]
    pred_price = last_price * np.exp(pred_lr)
    return {
        "date": latest["DATE"] + pd.Timedelta(days=1),
        "predicted_log_return": float(pred_lr),
        "predicted_price": float(pred_price)
    }

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

PLACEHOLDER = [2515.42, 2673.18, 2707.77, 2789.05, 2819.91, 2863.63, 2962.88]