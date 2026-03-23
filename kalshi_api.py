"""
Kalshi API Client
"""

from datetime import datetime, timedelta


class KalshiAPI:
    def __init__(self, api_key_id=None, private_key=None):
        self.api_key_id = api_key_id
        self.private_key = private_key
        self.cache = {}
        self.cache_timeout = 300
    
    def get_btc_15min_line(self):
        return None
    
    def get_btc_hourly_line(self):
        return None
    
    def get_historical_line(self, start_time, end_time):
        return None
    
    def get_market_data(self, ticker):
        return None
    
    def get_market_sentiment(self, ticker):
        return 0.5