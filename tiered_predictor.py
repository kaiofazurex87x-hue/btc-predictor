import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import ta
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os
import sqlite3
import pytz
import warnings
import config

warnings.filterwarnings('ignore')

MODELS_DIR = 'models/tiered'
SCALER_PATH = 'models/tiered_scaler.joblib'
DB_PATH     = 'data/hourly.db'
TZ          = pytz.timezone(config.TIMEZONE)


def now_ct():
    return datetime.now(pytz.utc).astimezone(TZ)


class TieredHourlyPredictor:
    def __init__(self):
        self.exchange = ccxt.kraken({'enableRateLimit': True})
        self.scaler   = RobustScaler()
        self.models   = {}
        self.is_trained = False
        self._init_db()
        self._init_models()

    def _init_db(self):
        os.makedirs('data', exist_ok=True)
        conn = self._db()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS hourly (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT,
                open_price  REAL,
                direction   TEXT,
                confidence  REAL,
                target_time TEXT,
                safe        INTEGER,
                modest      INTEGER,
                aggressive  INTEGER,
                close_price REAL,
                actual      TEXT,
                correct     INTEGER
            )''')
        conn.commit()
        conn.close()

    def _db(self):
        return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)

    def _init_models(self):
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=400, max_depth=8, learning_rate=0.03,
                random_state=42, eval_metric='logloss', use_label_encoder=False),
            'random_forest': RandomForestClassifier(
                n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42)
        }
        self._load_models()

    def _load_models(self):
        if not os.path.exists(MODELS_DIR):
            return
        loaded = 0
        for name in self.models:
            p = f'{MODELS_DIR}/{name}.joblib'
            if os.path.exists(p):
                try:
                    self.models[name] = joblib.load(p)
                    loaded += 1
                except Exception:
                    pass
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
        self.is_trained = (loaded == len(self.models))

    def save_models(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        for name, m in self.models.items():
            joblib.dump(m, f'{MODELS_DIR}/{name}.joblib')
        joblib.dump(self.scaler, SCALER_PATH)

    def fetch_data(self, limit=5000):
        ohlcv = self.exchange.fetch_ohlcv('BTC/USD', '1m', limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(TZ)
        df.set_index('timestamp', inplace=True)
        return df

    def _features(self, df):
        d = df.copy()
        for p in [5, 10, 15, 30, 60]:
            d[f'ret_{p}'] = d['close'].pct_change(p) * 100
        for p in [10, 20, 30, 50]:
            d[f'sma_{p}'] = ta.trend.sma_indicator(d['close'], window=p)
            d[f'ema_{p}'] = ta.trend.ema_indicator(d['close'], window=p)
        d['rsi']       = ta.momentum.rsi(d['close'], window=14)
        d['macd']      = ta.trend.macd(d['close'])
        d['macd_sig']  = ta.trend.macd_signal(d['close'])
        hi = ta.volatility.bollinger_hband(d['close'], window=20, window_dev=2)
        lo = ta.volatility.bollinger_lband(d['close'], window=20, window_dev=2)
        d['bb_pos']    = (d['close'] - lo) / (hi - lo + 1e-9)
        d['vol_ratio'] = d['volume'] / (d['volume'].rolling(20).mean() + 1e-9)
        d['hour']      = d.index.hour
        d['minute']    = d.index.minute
        return d.ffill().fillna(0).replace([np.inf, -np.inf], 0)

    def _label(self, df):
        pct = (df['close'].shift(-60) - df['close']) / df['close'] * 100
        lbl = np.where(pct > 0.1, 1, np.where(pct < -0.1, 0, np.nan))
        return pd.Series(lbl, index=df.index)

    def _xcols(self, df):
        return [c for c in df.columns if c not in {'open','high','low','close','volume'}]

    def _tiers(self, price, projected):
        move = abs(projected - price)
        up   = projected > price
        r    = lambda v: int(round(v / config.ROUND_TO) * config.ROUND_TO)
        if up:
            s, m, a = r(price + move*0.25), r(price + move*0.75), r(price + move*1.25)
        else:
            s, m, a = r(price - move*0.25), r(price - move*0.75), r(price - move*1.25)
        prices = sorted([s, m, a])
        return {
            'safe':       {'price': prices[0], 'formatted': f'${prices[0]:,}'},
            'modest':     {'price': prices[1], 'formatted': f'${prices[1]:,}'},
            'aggressive': {'price': prices[2], 'formatted': f'${prices[2]:,}'}
        }

    def train(self, df=None):
        print("Training hourly predictor...")
        if df is None:
            df = self.fetch_data(5000)
        df = self._features(df)
        y  = self._label(df)
        mask = ~y.isna()
        X = np.nan_to_num(df[mask][self._xcols(df[mask])].values)
        y = y[mask].values
        split = int(len(X) * 0.8)
        Xtr = self.scaler.fit_transform(X[:split])
        Xte = self.scaler.transform(X[split:])
        for name, m in self.models.items():
            m.fit(Xtr, y[:split])
            print(f"  {name}: {m.score(Xte, y[split:]):.2%}")
        self.save_models()
        self.is_trained = True

    def predict(self, override_price=None):
        if not self.is_trained:
            raise RuntimeError("Hourly model not trained.")
        df = self.fetch_data(200)
        if override_price is not None:
            df.iloc[-1, df.columns.get_loc('close')] = float(override_price)
        df_f = self._features(df)
        X    = np.nan_to_num(df_f[self._xcols(df_f)].values[-1:])
        Xsc  = self.scaler.transform(X)
        weights = {'xgboost': 0.35, 'random_forest': 0.35, 'gradient_boosting': 0.30}
        up_prob = sum(
            self.models[n].predict_proba(Xsc)[0][1] * 100 * w
            for n, w in weights.items()
        )
        up_prob    = round(min(100.0, max(0.0, up_prob)), 1)
        direction  = "UP" if up_prob >= 50 else "DOWN"
        confidence = round(min(100.0, abs(up_prob - 50) * 2), 1)
        price      = float(override_price) if override_price is not None else float(df['close'].iloc[-1])
        move_pct   = (up_prob - 50) / 50 * 1.5
        projected  = price * (1 + move_pct / 100)
        tiers      = self._tiers(price, projected)
        ct_now     = now_ct()
        target     = ct_now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        conn = self._db()
        conn.execute(
            """INSERT INTO hourly
               (timestamp, open_price, direction, confidence, target_time, safe, modest, aggressive)
               VALUES (?,?,?,?,?,?,?,?)""",
            (ct_now.isoformat(), price, direction, confidence, target.isoformat(),
             tiers['safe']['price'], tiers['modest']['price'], tiers['aggressive']['price'])
        )
        conn.commit()
        conn.close()
        return {
            'direction':    direction,
            'confidence':   confidence,
            'up_prob':      up_prob,
            'open_price':   price,
            'projected':    round(projected, 0),
            'move_pct':     round(move_pct, 2),
            'target_time':  target.strftime('%I:%M %p CT'),
            'tiers':        tiers
        }

    def get_accuracy(self):
        conn = self._db()
        try:
            df = pd.read_sql_query(
                "SELECT correct FROM hourly WHERE correct IS NOT NULL", conn)
        except Exception:
            df = pd.DataFrame(columns=['correct'])
        conn.close()
        if len(df) < 5:
            return {'overall': 50.0, 'recent': 50.0, 'total': len(df)}
        overall = round(df['correct'].mean() * 100, 1)
        recent  = round(df.tail(100)['correct'].mean() * 100, 1)
        return {'overall': overall, 'recent': recent, 'total': len(df)}

    def get_history(self, limit=20):
        conn = self._db()
        try:
            df = pd.read_sql_query(
                """SELECT timestamp, open_price, direction, confidence,
                          safe, modest, aggressive, actual, correct
                   FROM hourly ORDER BY id DESC LIMIT ?""",
                conn, params=(limit,))
        except Exception:
            df = pd.DataFrame()
        conn.close()
        return df.to_dict('records')
