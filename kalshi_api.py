"""
Kalshi API Client - Uses your real API keys from fly.io secrets
"""

import requests
import base64
import hashlib
import hmac
import time
from datetime import datetime, timedelta
import config


class KalshiAPI:
    def __init__(self):
        self.api_key_id = config.KALSHI_KEY_ID
        self.private_key = config.KALSHI_PRIVATE_KEY
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        if self.api_key_id and self.private_key:
            print(f"[Kalshi] Initialized with API Key: {self.api_key_id[:8]}...")
        else:
            print("[Kalshi] No API keys set - manual mode only")
    
    def _sign(self, message):
        """Create RSA signature for Kalshi API"""
        try:
            key_lines = self.private_key.strip().split('\n')
            key_base64 = ''.join([line for line in key_lines if line and '---' not in line])
            key_material = key_base64.encode('utf-8')
            hmac_obj = hmac.new(key_material, message.encode('utf-8'), hashlib.sha256)
            return base64.b64encode(hmac_obj.digest()).decode('utf-8')
        except Exception as e:
            print(f"[Kalshi] Signing error: {e}")
            return None
    
    def get_headers(self, method, path, body=""):
        """Generate authenticated headers"""
        if not self.api_key_id or not self.private_key:
            return None
        
        timestamp = str(int(time.time() * 1000))
        message = timestamp + method + path + body
        signature = self._sign(message)
        
        if not signature:
            return None
        
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }
    
    def _request(self, method, path, params=None, use_auth=False):
        """Make request to Kalshi API"""
        url = f"{self.base_url}{path}"
        headers = {}
        
        if use_auth and self.api_key_id and self.private_key:
            headers = self.get_headers(method, path)
            if not headers:
                return None
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=10)
            else:
                return None
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[Kalshi] API Error {response.status_code}")
                return None
        except Exception as e:
            print(f"[Kalshi] Request failed: {e}")
            return None
    
    def get_exchange_status(self):
        """Check if Kalshi API is working"""
        return self._request("GET", "/exchange/status", use_auth=False)
    
    def get_markets(self, limit=100):
        """Get all markets"""
        params = {"limit": min(limit, 1000)}
        return self._request("GET", "/markets", params=params, use_auth=False)
    
    def get_btc_markets(self):
        """Find BTC-related markets"""
        markets = self.get_markets(limit=1000)
        if not markets:
            return []
        
        btc_markets = []
        for market in markets.get('markets', []):
            ticker = market.get('ticker', '').upper()
            title = market.get('title', '').upper()
            if 'BTC' in ticker or 'BITCOIN' in title:
                btc_markets.append(market)
        return btc_markets
    
    def get_market_orderbook(self, ticker):
        """Get order book for a market"""
        return self._request("GET", f"/markets/{ticker}/orderbook", use_auth=False)
    
    def get_market_trades(self, ticker=None, limit=100):
        """Get recent trades"""
        params = {"limit": min(limit, 1000)}
        if ticker:
            params["ticker"] = ticker
        return self._request("GET", "/markets/trades", params=params, use_auth=False)
    
    def get_market_sentiment(self, ticker):
        """Calculate market sentiment from order book"""
        orderbook = self.get_market_orderbook(ticker)
        if not orderbook:
            return 0.5
        
        yes_bids = sum(bid[1] for bid in orderbook.get('yes', []))
        no_bids = sum(bid[1] for bid in orderbook.get('no', []))
        total = yes_bids + no_bids
        
        return yes_bids / total if total > 0 else 0.5
    
    def get_btc_15min_line(self):
        """Get current Kalshi line for 15-minute BTC prediction"""
        try:
            btc_markets = self.get_btc_markets()
            if btc_markets:
                # Try to find a 15-min market
                for market in btc_markets:
                    ticker = market.get('ticker', '')
                    if '15MIN' in ticker or '15MIN' in market.get('title', ''):
                        sentiment = self.get_market_sentiment(ticker)
                        return sentiment * 100  # Convert to 0-100 scale
            return None
        except Exception as e:
            print(f"[Kalshi] 15-min error: {e}")
            return None
    
    def get_btc_hourly_line(self):
        """Get current Kalshi line for hourly BTC prediction"""
        try:
            btc_markets = self.get_btc_markets()
            if btc_markets:
                for market in btc_markets:
                    ticker = market.get('ticker', '')
                    if 'HOURLY' in ticker or 'HOUR' in ticker:
                        sentiment = self.get_market_sentiment(ticker)
                        return sentiment * 100
            return None
        except Exception as e:
            print(f"[Kalshi] Hourly error: {e}")
            return None
    
    def get_historical_line(self, start_time, end_time):
        """Get Kalshi data for a specific time period for verification"""
        cache_key = f"{start_time.isoformat()}_{end_time.isoformat()}"
        
        # Check cache
        if cache_key in self.cache:
            cache_age = (datetime.now() - self.cache[cache_key]['timestamp']).seconds
            if cache_age < self.cache_timeout:
                return self.cache[cache_key]['data']
        
        # For now, return None to use price verification
        # In production, you would call Kalshi's historical API here
        return None