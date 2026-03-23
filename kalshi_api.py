"""
Kalshi API Client - Automatic fetch with manual fallback
"""

import requests
import json
from datetime import datetime, timedelta
import time

class KalshiAPI:
    def __init__(self, api_key_id=None, private_key=None):
        self.api_key_id = api_key_id
        self.private_key = private_key
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes cache
        
    def get_btc_15min_line(self):
        """
        Get current Kalshi line for 15-minute BTC prediction
        Tries API first, returns None if unavailable
        """
        # Try to fetch from API
        market_data = self.get_market_data("BTC-15MIN")
        if market_data:
            return market_data.get('line')
        
        # If API fails, return None for manual entry
        return None
    
    def get_btc_hourly_line(self):
        """Get current Kalshi line for hourly BTC prediction"""
        market_data = self.get_market_data("BTC-HOURLY")
        if market_data:
            return market_data.get('line')
        return None
    
    def get_historical_line(self, start_time, end_time):
        """
        Get Kalshi line for a specific time period
        Used for automatic verification
        """
        cache_key = f"{start_time.isoformat()}_{end_time.isoformat()}"
        
        # Check cache
        if cache_key in self.cache:
            cache_age = (datetime.now() - self.cache[cache_key]['timestamp']).seconds
            if cache_age < self.cache_timeout:
                return self.cache[cache_key]['data']
        
        # Try to fetch historical data
        try:
            # This would call the Kalshi historical API
            # For now, return None to trigger manual fallback
            return None
        except Exception as e:
            print(f"[Kalshi] Historical fetch error: {e}")
            return None
    
    def get_market_data(self, ticker):
        """Fetch market data from Kalshi"""
        try:
            # This would be the actual API call
            # For now, return None to rely on manual input
            return None
        except Exception as e:
            print(f"[Kalshi] Market data error: {e}")
            return None
    
    def get_market_sentiment(self, ticker):
        """Get market sentiment for a specific market"""
        # Try API first
        market_data = self.get_market_data(ticker)
        if market_data and 'sentiment' in market_data:
            return market_data['sentiment']
        
        # Return neutral if unavailable
        return 0.5