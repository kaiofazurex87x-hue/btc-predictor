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

MODELS_DIR = 'models/predictor'
SCALER_PATH = 'models/predictor_scaler.joblib'
DB_PATH     = 'data/predictions.db'
TZ          = pytz.timezone(config.TIMEZONE)


def now_ct():
    return datetime.now(pytz.utc).astimezone(TZ)


class BTCPredictor:
    def __init__(self):
        self.exchange = ccxt.kraken({'enableRateLimit': True})
        self.scaler   = RobustScaler()
        self.models   = {}
        self.is_trained = False
        self.verified_since_retrain = 0
        self._init_db()
        self._init_models()

    # ── Database ──────────────────────────────────────────────
    def _init_db(self):
        os.makedirs('data', exist_ok=True)
        conn = self._db()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT,
                open_price    REAL,
                direction     TEXT,
                confidence    REAL,
                up_prob       REAL,
                target_time   TEXT,
                close_price   REAL,
                actual        TEXT,
                correct       INTEGER,
                kalshi_yes    REAL,
                kalshi_no     REAL,
                recommendation TEXT,
                edge          REAL
            )''')
        conn.commit()
        conn.close()

    def _db(self):
        return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)

    # ── Models ────────────────────────────────────────────────
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
        loaded = sum(
            1 for name in self.models
            if os.path.exists(f'{MODELS_DIR}/{name}.joblib')
            and self._load_one(name)
        )
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
        self.is_trained = (loaded == len(self.models))

    def _load_one(self, name):
        try:
            self.models[name] = joblib.load(f'{MODELS_DIR}/{name}.joblib')
            return True
        except Exception:
            return False

    def save_models(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        for name, m in self.models.items():
            joblib.dump(m, f'{MODELS_DIR}/{name}.joblib')
        joblib.dump(self.scaler, SCALER_PATH)

    # ── Data ──────────────────────────────────────────────────
    def fetch_data(self, limit=5000):
        ohlcv = self.exchange.fetch_ohlcv('BTC/USD', '1m', limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(TZ)
        df.set_index('timestamp', inplace=True)
        return df

    def _features(self, df):
        d = df.copy()
        for p in [1, 3, 5, 10, 15]:
            d[f'ret_{p}'] = d['close'].pct_change(p) * 100
        for p in [5, 10, 15, 20, 30]:
            d[f'sma_{p}'] = ta.trend.sma_indicator(d['close'], window=p)
            d[f'ema_{p}'] = ta.trend.ema_indicator(d['close'], window=p)
        d['rsi']        = ta.momentum.rsi(d['close'], window=14)
        d['macd']       = ta.trend.macd(d['close'])
        d['macd_sig']   = ta.trend.macd_signal(d['close'])
        d['macd_hist']  = ta.trend.macd_diff(d['close'])
        hi = ta.volatility.bollinger_hband(d['close'], window=20, window_dev=2)
        lo = ta.volatility.bollinger_lband(d['close'], window=20, window_dev=2)
        d['bb_pos']     = (d['close'] - lo) / (hi - lo + 1e-9)
        d['vol_ratio']  = d['volume'] / (d['volume'].rolling(20).mean() + 1e-9)
        d['hour']       = d.index.hour
        d['minute']     = d.index.minute
        return d.ffill().fillna(0).replace([np.inf, -np.inf], 0)

    def _label(self, df):
        pct = (df['close'].shift(-15) - df['close']) / df['close'] * 100
        lbl = np.where(pct > 0.1, 1, np.where(pct < -0.1, 0, np.nan))
        return pd.Series(lbl, index=df.index)

    def _xcols(self, df):
        return [c for c in df.columns if c not in {'open','high','low','close','volume'}]

    # ── Training ──────────────────────────────────────────────
    def train(self, df=None):
        print("Training 15-min predictor...")
        if df is None:
            df = self.fetch_data(5000)
        df = self._features(df)
        y  = self._label(df)
        mask = ~y.isna()
        X = np.nan_to_num(df[mask][self._xcols(df[mask])].values)
        y = y[mask].values
        split = int(len(X) * 0.8)
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]
        Xtr = self.scaler.fit_transform(Xtr)
        Xte = self.scaler.transform(Xte)
        scores = {}
        for name, m in self.models.items():
            m.fit(Xtr, ytr)
            scores[name] = m.score(Xte, yte)
            print(f"  {name}: {scores[name]:.2%}")
        self.save_models()
        self.is_trained = True
        self.verified_since_retrain = 0
        print(f"  Ensemble avg: {np.mean(list(scores.values())):.2%}")
        return scores

    # ── Predict ───────────────────────────────────────────────
    def predict(self, override_price=None):
        """
        Main prediction. override_price is optional — if not provided,
        the app fetches the live BTC price automatically.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        df = self.fetch_data(200)
        if df.empty:
            raise RuntimeError("No market data available.")

        # Optional price override
        if override_price is not None:
            df.iloc[-1, df.columns.get_loc('close')] = float(override_price)

        df_f  = self._features(df)
        X     = np.nan_to_num(df_f[self._xcols(df_f)].values[-1:])
        Xsc   = self.scaler.transform(X)

        weights = {'xgboost': 0.35, 'random_forest': 0.35, 'gradient_boosting': 0.30}
        up_prob = sum(
            self.models[n].predict_proba(Xsc)[0][1] * 100 * w
            for n, w in weights.items()
        )
        up_prob    = round(min(100.0, max(0.0, up_prob)), 1)
        down_prob  = round(100.0 - up_prob, 1)
        direction  = "UP" if up_prob >= 50 else "DOWN"
        confidence = round(min(100.0, abs(up_prob - 50) * 2), 1)

        ct_now    = now_ct()
        next_min  = ((ct_now.minute // 15) + 1) * 15
        if next_min >= 60:
            target = ct_now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            target = ct_now.replace(minute=next_min, second=0, microsecond=0)

        open_price = float(override_price) if override_price is not None else float(df['close'].iloc[-1])

        conn = self._db()
        cur = conn.execute(
            """INSERT INTO predictions
               (timestamp, open_price, direction, confidence, up_prob, target_time)
               VALUES (?,?,?,?,?,?)""",
            (ct_now.isoformat(), open_price, direction, confidence, up_prob, target.isoformat())
        )
        pred_id = cur.lastrowid
        conn.commit()
        conn.close()

        return {
            'id':           pred_id,
            'direction':    direction,
            'confidence':   confidence,
            'up_prob':      up_prob,
            'down_prob':    down_prob,
            'open_price':   open_price,
            'target_time':  target.strftime('%I:%M %p CT'),
            'target_iso':   target.isoformat(),
        }

    # ── Kalshi BUY/SKIP ───────────────────────────────────────
    def kalshi_check(self, pred_id, up_prob, kalshi_yes=None, kalshi_no=None):
        """
        Both kalshi_yes and kalshi_no are optional.
        Pass whichever price(s) you see on the Kalshi app.
        Returns a recommendation for each side that was provided.
        """
        result = {'yes': None, 'no': None, 'best_action': None, 'best_edge': 0}

        if kalshi_yes is not None:
            yes_edge = round(up_prob - float(kalshi_yes), 1)
            result['yes'] = {
                'side':           'YES (UP)',
                'our_prob':       up_prob,
                'kalshi_price':   kalshi_yes,
                'edge':           yes_edge,
                'recommendation': 'BUY' if yes_edge >= config.MIN_EDGE_CENTS else 'SKIP'
            }

        if kalshi_no is not None:
            down_prob = round(100.0 - up_prob, 1)
            no_edge   = round(down_prob - float(kalshi_no), 1)
            result['no'] = {
                'side':           'NO (DOWN)',
                'our_prob':       down_prob,
                'kalshi_price':   kalshi_no,
                'edge':           no_edge,
                'recommendation': 'BUY' if no_edge >= config.MIN_EDGE_CENTS else 'SKIP'
            }

        # Pick whichever side has the bigger edge
        edges = []
        if result['yes']: edges.append((result['yes']['edge'], 'YES', result['yes']['recommendation']))
        if result['no']:  edges.append((result['no']['edge'],  'NO',  result['no']['recommendation']))
        if edges:
            best = max(edges, key=lambda x: x[0])
            result['best_action'] = best[2]
            result['best_edge']   = best[0]
            result['best_side']   = best[1]

        # Save to DB
        conn = self._db()
        conn.execute(
            """UPDATE predictions SET
               kalshi_yes=?, kalshi_no=?, recommendation=?, edge=?
               WHERE id=?""",
            (kalshi_yes, kalshi_no, result.get('best_action'), result.get('best_edge'), pred_id)
        )
        conn.commit()
        conn.close()

        return result

    # ── Auto-verify ───────────────────────────────────────────
    def verify_pending(self):
        """
        Runs automatically in background every 2 minutes.
        Checks if any predictions have passed their target time,
        fetches the actual BTC price, and records correct/incorrect.
        Triggers retrain if enough new data is verified.
        """
        ct_now = now_ct()
        conn   = self._db()
        rows   = conn.execute(
            "SELECT id, open_price, direction, target_time FROM predictions WHERE actual IS NULL"
        ).fetchall()
        verified = 0

        for pred_id, open_price, direction, target_str in rows:
            target = datetime.fromisoformat(target_str)
            if target.tzinfo is None:
                target = TZ.localize(target)
            if ct_now < target + timedelta(minutes=1):
                continue
            try:
                since  = int(target.timestamp() * 1000)
                ohlcv  = self.exchange.fetch_ohlcv('BTC/USD', '1m', since=since, limit=1)
                if not ohlcv:
                    continue
                close_price = ohlcv[0][4]
                actual      = "UP" if close_price > open_price else "DOWN"
                correct     = 1 if actual == direction else 0
                conn.execute(
                    "UPDATE predictions SET close_price=?, actual=?, correct=? WHERE id=?",
                    (close_price, actual, correct, pred_id)
                )
                verified += 1
                self.verified_since_retrain += 1
            except Exception as e:
                print(f"Verify error id={pred_id}: {e}")

        conn.commit()
        conn.close()

        if self.verified_since_retrain >= config.RETRAIN_AFTER_N_VERIFIED:
            print(f"Auto-retraining after {self.verified_since_retrain} new verified predictions...")
            try:
                self.train()
            except Exception as e:
                print(f"Auto-retrain error: {e}")

        return verified

    # ── Accuracy ──────────────────────────────────────────────
    def get_accuracy(self):
        conn = self._db()
        try:
            df = pd.read_sql_query(
                "SELECT correct FROM predictions WHERE correct IS NOT NULL", conn)
        except Exception:
            df = pd.DataFrame(columns=['correct'])
        conn.close()
        if len(df) < 5:
            return {'overall': 50.0, 'recent': 50.0, 'total': len(df), 'edge': 0.0}
        overall = round(df['correct'].mean() * 100, 1)
        recent  = round(df.tail(100)['correct'].mean() * 100, 1)
        return {
            'overall': overall,
            'recent':  recent,
            'total':   len(df),
            'edge':    round(overall - 50.0, 1)
        }

    # ── History ───────────────────────────────────────────────
    def get_history(self, limit=30):
        conn = self._db()
        try:
            df = pd.read_sql_query(
                """SELECT id, timestamp, open_price, direction, confidence,
                          target_time, close_price, actual, correct,
                          kalshi_yes, kalshi_no, recommendation, edge
                   FROM predictions ORDER BY id DESC LIMIT ?""",
                conn, params=(limit,))
        except Exception:
            df = pd.DataFrame()
        conn.close()
        return df.to_dict('records')

    # ── Auto-prune ────────────────────────────────────────────
    def prune_old_data(self):
        acc = self.get_accuracy()
        if acc['overall'] < config.PRUNE_MIN_ACCURACY:
            return 0
        cutoff = (now_ct() - timedelta(days=config.AUTO_PRUNE_DAYS)).isoformat()
        conn   = self._db()
        n      = conn.execute(
            "DELETE FROM predictions WHERE timestamp < ?", (cutoff,)
        ).rowcount
        conn.commit()
        conn.close()
        if n:
            print(f"Pruned {n} old predictions")
        return n
