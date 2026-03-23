"""
Whale Tracker - Simulated whale activity monitoring
"""

import numpy as np
from datetime import datetime
import random

class WhaleTracker:
    def __init__(self):
        self.history = []
        
    def track(self):
        """Get current whale activity data"""
        hour = datetime.now().hour
        
        # Simulate whale activity based on market hours
        if 14 <= hour <= 16:  # NY trading hours
            ratio = 1.3 + random.uniform(-0.1, 0.1)
            transactions = 25 + random.randint(-5, 10)
            signal = 'bullish'
        elif 8 <= hour <= 10:  # London trading hours
            ratio = 1.2 + random.uniform(-0.1, 0.1)
            transactions = 20 + random.randint(-5, 8)
            signal = 'bullish'
        elif 20 <= hour <= 22:  # Asia trading hours
            ratio = 0.9 + random.uniform(-0.1, 0.1)
            transactions = 18 + random.randint(-5, 8)
            signal = 'neutral'
        else:
            ratio = 1.0 + random.uniform(-0.15, 0.15)
            transactions = 12 + random.randint(-4, 6)
            signal = 'neutral' if 0.9 < ratio < 1.1 else 'bearish' if ratio < 0.9 else 'bullish'
        
        data = {
            'ratio': ratio,
            'transactions': transactions,
            'timestamp': datetime.now().isoformat(),
            'buy_volume': random.randint(50, 200) * 1000,
            'sell_volume': random.randint(50, 200) * 1000
        }
        
        self.history.append(data)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        return data
    
    def get_signal(self):
        """Get whale signal (bullish/bearish/neutral)"""
        if not self.history:
            return 'neutral'
        
        recent = self.history[-10:]
        avg_ratio = sum(h['ratio'] for h in recent) / len(recent)
        
        if avg_ratio > 1.15:
            return 'bullish'
        elif avg_ratio < 0.85:
            return 'bearish'
        else:
            return 'neutral'