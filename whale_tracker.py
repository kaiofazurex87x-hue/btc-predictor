import pandas as pd
from collections import deque
from datetime import datetime
import ccxt


class WhaleTracker:
    def __init__(self):
        self.exchange = ccxt.kraken({'enableRateLimit': True})
        self.trades   = deque(maxlen=1000)

    def track(self):
        try:
            ob   = self.exchange.fetch_order_book('BTC/USD', 100)
            bids = pd.DataFrame(ob['bids'], columns=['price','volume'])
            asks = pd.DataFrame(ob['asks'], columns=['price','volume'])
            bids['value'] = bids['price'] * bids['volume']
            asks['value'] = asks['price'] * asks['volume']
            bids['type']  = 'buy'
            asks['type']  = 'sell'
            orders = pd.concat([bids, asks])
            whales = orders[orders['value'] >= 500_000]
            for _, row in whales.iterrows():
                self.trades.append({
                    'ts':    datetime.now(),
                    'type':  row['type'],
                    'value': row['value']
                })
            if whales.empty:
                return {
                    'active': False, 'sentiment': 0,
                    'buy_vol': 0, 'sell_vol': 0,
                    'accumulating': False
                }
            buy_v  = whales[whales['type']=='buy']['value'].sum()
            sell_v = whales[whales['type']=='sell']['value'].sum()
            total  = buy_v + sell_v
            sent   = round(
                (buy_v - sell_v) / total * 100, 1) if total else 0
            return {
                'active':       True,
                'sentiment':    sent,
                'buy_vol':      round(buy_v  / 1_000_000, 2),
                'sell_vol':     round(sell_v / 1_000_000, 2),
                'accumulating': sent > 30
            }
        except Exception:
            return {
                'active': False, 'sentiment': 0,
                'buy_vol': 0, 'sell_vol': 0,
                'accumulating': False
            }

    def get_signal(self):
        if not self.trades:
            return 0
        recent = list(self.trades)[-50:]
        buy    = sum(t['value'] for t in recent if t['type'] == 'buy')
        sell   = sum(t['value'] for t in recent if t['type'] == 'sell')
        total  = buy + sell
        return round((buy - sell) / total * 100, 1) if total else 0
