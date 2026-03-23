import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import ta
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
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
        self.exchange   = ccxt.kraken({'enableRateLimit': True})
        self.scaler     = RobustScaler()
        self.models     = {}
        self.is_trained = False
        self._init_db()
        self._init_models()

    def _init_db(self):
        os.makedirs('data', exist_ok=True)
        conn = self._db()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS hourly (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT,
                current_price   REAL,
                predicted_price REAL,
                confidence      REAL,
                predicted_move  REAL,
                safe            REAL,
                modest          REAL,
                aggressive      REAL,
                target_time     TEXT,
                actual_price    REAL,
                verified        INTEGER DEFAULT 0
            )''')
        conn.commit()
        conn.close()

    def _db(self):
        return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)

    def _init_models(self):
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=300, max_depth=10,
                random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=300, max_depth=6,
                learning_rate=0.05, random_state=42),
            'ridge': Ridge(alpha=1.0)
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

    def fetch_data(self, limit=3000):
        ohlcv = self.exchange.fetch_ohlcv('BTC/USD', '1m', limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(
            df['timestamp'], unit='ms', utc=True).dt.tz_convert(TZ)
        df.set_index('timestamp', inplace=True)
        return df

    def _features(self, df):
        d = df.copy()
        for p in [5, 10, 15, 30, 60]:
            d[f'ret_{p}'] = d['close'].pct_change(p) * 100
        for p in [10, 20, 30, 50]:
            d[f'sma_{p}'] = ta.trend.sma_indicator(d['close'], window=p)
            d[f'ema_{p}'] = ta.trend.ema_indicator(d['close'], window=p)
        for p in [10, 20, 50]:
            d[f'dist_sma_{p}'] = (
                d['close'] - d[f'sma_{p}']) / d[f'sma_{p}'] * 100
        d['rsi']      = ta.momentum.rsi(d['close'], window=14)
        d['macd']     = ta.trend.macd(d['close'])
        d['macd_sig'] = ta.trend.macd_signal(d['close'])
        hi = ta.volatility.bollinger_hband(
            d['close'], window=20, window_dev=2)
        lo = ta.volatility.bollinger_lband(
            d['close'], window=20, window_dev=2)
        d['bb_width'] = (hi - lo) / d['close'] * 100
        d['bb_pos']   = (d['close'] - lo) / (hi - lo + 1e-9)
        d['vol_ratio'] = d['volume'] / (
            d['volume'].rolling(20).mean() + 1e-9)
        d['hour']   = d.index.hour
        d['minute'] = d.index.minute
        return d.ffill().fillna(0).replace([np.inf, -np.inf], 0)

    def _label(self, df):
        return df['close'].shift(-60)

    def _xcols(self, df):
        return [c for c in df.columns
                if c not in {'open','high','low','close','volume'}]

    def train(self, df=None):
        print("Training hourly price regression model...")
        if df is None:
            df = self.fetch_data(3000)
        df = self._features(df)
        y  = self._label(df)
        mask = ~y.isna()
        df_c = df[mask]
        y_c  = y[mask]
        cols = self._xcols(df_c)
        X    = np.nan_to_num(df_c[cols].values)
        y_vals = y_c.values
        split = int(len(X) * 0.8)
        X_tr  = self.scaler.fit_transform(X[:split])
        X_te  = self.scaler.transform(X[split:])
        y_tr  = y_vals[:split]
        y_te  = y_vals[split:]
        scores = {}
        for name, m in self.models.items():
            m.fit(X_tr, y_tr)
            scores[name] = m.score(X_te, y_te)
            preds = m.predict(X_te)
            mae   = np.mean(np.abs(preds - y_te))
            print(f"  {name}: R²={scores[name]:.3f}  MAE=${mae:.0f}")
        self.save_models()
        self.is_trained = True
        print(f"  Ensemble avg R²: {np.mean(list(scores.values())):.3f}")
        return scores

    def predict(self, override_price=None):
        if not self.is_trained:
            raise RuntimeError("Hourly model not trained.")
        df = self.fetch_data(200)
        if df.empty:
            raise RuntimeError("No market data.")
        current_price = float(df['close'].iloc[-1])
        df_f  = self._features(df)
        cols  = self._xcols(df_f)
        X     = np.nan_to_num(df_f[cols].values[-1:])
        X_sc  = self.scaler.transform(X)
        weights = {
            'random_forest':    0.40,
            'gradient_boosting':0.40,
            'ridge':            0.20
        }
        predicted_price = 0.0
        preds_list      = []
        for name, m in self.models.items():
            try:
                pred = float(m.predict(X_sc)[0])
            except Exception:
                pred = current_price
            predicted_price += pred * weights.get(name, 0)
            preds_list.append(pred)
        predicted_price = round(predicted_price, 2)
        predicted_move  = round(predicted_price - current_price, 2)
        std_dev    = np.std(preds_list)
        confidence = round(
            max(0.0, min(100.0, (1 - std_dev / 500) * 100)), 1)

        def fmt(p):
            r = round(p / config.ROUND_TO) * config.ROUND_TO
            return {'price': r, 'formatted': f'${r:,.0f}'}

        tiers = {
            'safe':       fmt(predicted_price - config.SAFE_OFFSET),
            'modest':     fmt(predicted_price - config.MODEST_OFFSET),
            'aggressive': fmt(predicted_price - config.AGGRESSIVE_OFFSET)
        }
        ct_now = now_ct()
        target = ct_now.replace(
            minute=0, second=0, microsecond=0) + timedelta(hours=1)
        conn = self._db()
        conn.execute(
            """INSERT INTO hourly
               (timestamp, current_price, predicted_price, confidence,
                predicted_move, safe, modest, aggressive, target_time)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ct_now.isoformat(), current_price, predicted_price,
             confidence, predicted_move,
             tiers['safe']['price'],
             tiers['modest']['price'],
             tiers['aggressive']['price'],
             target.isoformat())
        )
        conn.commit()
        conn.close()
        return {
            'current_price':   current_price,
            'predicted_price': predicted_price,
            'predicted_move':  predicted_move,
            'confidence':      confidence,
            'target_time':     target.strftime('%I:%M %p CT'),
            'tiers':           tiers
        }

    def verify_pending(self):
        ct_now = now_ct()
        conn   = self._db()
        rows   = conn.execute(
            """SELECT id, predicted_price, target_time
               FROM hourly WHERE verified=0"""
        ).fetchall()
        verified = 0
        for row_id, predicted_price, target_str in rows:
            target = datetime.fromisoformat(target_str)
            if target.tzinfo is None:
                target = TZ.localize(target)
            if ct_now < target + timedelta(minutes=1):
                continue
            try:
                since        = int(target.timestamp() * 1000)
                ohlcv        = self.exchange.fetch_ohlcv(
                    'BTC/USD', '1m', since=since, limit=1)
                if not ohlcv:
                    continue
                actual_price = ohlcv[0][4]
                conn.execute(
                    "UPDATE hourly SET actual_price=?, verified=1 WHERE id=?",
                    (actual_price, row_id))
                verified += 1
                print(f"[Hourly verify] Predicted ${predicted_price:.0f} "
                      f"Actual ${actual_price:.0f} "
                      f"Error ${abs(actual_price - predicted_price):.0f}")
            except Exception as e:
                print(f"[Hourly verify] error id={row_id}: {e}")
        conn.commit()
        conn.close()
        return verified

    def get_accuracy(self):
        conn = self._db()
        try:
            df = pd.read_sql_query(
                """SELECT predicted_price, actual_price FROM hourly
                   WHERE verified=1 AND actual_price IS NOT NULL""",
                conn)
        except Exception:
            df = pd.DataFrame()
        conn.close()
        if len(df) < 3:
            return {
                'within_500':  0,
                'within_1000': 0,
                'avg_error':   0,
                'total':       len(df)
            }
        df['error'] = (df['predicted_price'] - df['actual_price']).abs()
        return {
            'within_500':  round((df['error'] <= 500).mean()  * 100, 1),
            'within_1000': round((df['error'] <= 1000).mean() * 100, 1),
            'avg_error':   round(df['error'].mean(), 0),
            'total':       len(df)
        }

    def get_history(self, limit=10):
        conn = self._db()
        try:
            df = pd.read_sql_query(
                """SELECT timestamp, current_price, predicted_price,
                          confidence, predicted_move,
                          safe, modest, aggressive,
                          target_time, actual_price, verified
                   FROM hourly ORDER BY id DESC LIMIT ?""",
                conn, params=(limit,))
        except Exception:
            df = pd.DataFrame()
        conn.close()
        return df.to_dict('records')
