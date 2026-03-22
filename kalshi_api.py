import requests
import time
import base64
from datetime import datetime
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import pytz
import config

TZ = pytz.timezone(config.TIMEZONE)


def now_ct():
    return datetime.now(pytz.utc).astimezone(TZ)


class KalshiAPI:
    def __init__(self):
        self.base_url = "https://api.kalshi.com/trade-api/v2"
        self.api_key  = config.KALSHI_API_KEY
        self.priv_key = None
        if config.KALSHI_PRIVATE_KEY:
            try:
                self.priv_key = serialization.load_pem_private_key(
                    config.KALSHI_PRIVATE_KEY.encode(), password=None)
                print("[Kalshi] Private key loaded OK")
            except Exception as e:
                print(f"[Kalshi] Key load error: {e}")

    def _sign(self, method, path):
        if not self.priv_key:
            return "", 0
        ts  = int(time.time() * 1000)
        msg = f"{method}{path}{ts}"
        try:
            sig = self.priv_key.sign(
                msg.encode(), padding.PKCS1v15(), hashes.SHA256())
            return base64.b64encode(sig).decode(), ts
        except Exception as e:
            print(f"[Kalshi] Sign error: {e}")
            return "", 0

    def _get(self, path):
        if not self.priv_key:
            return {}
        sig, ts = self._sign("GET", path)
        if not sig:
            return {}
        try:
            r = requests.get(
                f"{self.base_url}{path}",
                headers={
                    "Content-Type":  "application/json",
                    "API-KEY":       self.api_key,
                    "API-SIGNATURE": sig,
                    "API-TIMESTAMP": str(ts)
                },
                timeout=10
            )
            if r.status_code == 200:
                return r.json()
            print(f"[Kalshi] HTTP {r.status_code}: {r.text[:200]}")
            return {}
        except Exception as e:
            print(f"[Kalshi] Request error: {e}")
            return {}

    def get_btc_15min_line(self):
        try:
            now      = now_ct()
            slot     = (now.minute // 15) * 15
            window   = now.replace(minute=slot, second=0, microsecond=0)
            date_str = window.strftime('%y%b%d').upper()
            time_str = window.strftime('%H:%M')
            ticker   = f"KXBTC-{date_str}{time_str}"
            data     = self._get(f"/markets/{ticker}")
            if data:
                strike = data.get('floor_strike') or data.get('cap_strike')
                if strike:
                    print(f"[Kalshi] 15-min line: ${strike}")
                    return float(strike)
            return self._search_active_btc()
        except Exception as e:
            print(f"[Kalshi] get_btc_15min_line error: {e}")
            return None

    def _search_active_btc(self):
        try:
            data = self._get(
                "/markets?status=open&ticker_contains=KXBTC&limit=10")
            if not data or 'markets' not in data:
                return None
            markets = data['markets']
            if not markets:
                return None
            markets.sort(key=lambda m: m.get('close_time', ''))
            strike = (markets[0].get('floor_strike') or
                      markets[0].get('cap_strike'))
            if strike:
                print(f"[Kalshi] Found via search: ${strike}")
                return float(strike)
            return None
        except Exception as e:
            print(f"[Kalshi] search error: {e}")
            return None

    def get_btc_hourly_line(self):
        try:
            now      = now_ct()
            hour     = now.replace(minute=0, second=0, microsecond=0)
            date_str = hour.strftime('%y%b%d').upper()
            time_str = hour.strftime('%H:00')
            ticker   = f"KXBTCD-{date_str}{time_str}"
            data     = self._get(f"/markets/{ticker}")
            if data:
                strike = data.get('floor_strike') or data.get('cap_strike')
                if strike:
                    print(f"[Kalshi] Hourly line: ${strike}")
                    return float(strike)
            return None
        except Exception as e:
            print(f"[Kalshi] get_btc_hourly_line error: {e}")
            return None
