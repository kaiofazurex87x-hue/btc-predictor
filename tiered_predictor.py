"""
Hourly BTC Predictor
"""

import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from collections import deque


class TieredHourlyPredictor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.is_trained = False
        
        self.predictions = deque(maxlen=2000)
        self.pending_predictions = {}
        self.accuracy_history = deque(maxlen=200)
        
        self.total_predictions = 0
        self.correct_predictions = 0
        self.last_price = None
        self.current_price = 68000
        
        os.makedirs(data_dir, exist_ok=True)
        self.load_data()
    
    def train(self):
        print("📊 Training hourly predictor...")
        if len(self.predictions) > 0:
            acc = self.get_accuracy()['recent']
            print(f"   Recent accuracy: {acc:.1f}%")
        self.is_trained = True
        self.save_data()
        return True
    
    def update_price(self, price):
        self.current_price = price
        self.last_price = price
        return price
    
    def get_current_price(self):
        return self.current_price if self.current_price else 68000
    
    def get_price_at_time(self, target_time):
        return self.current_price if self.current_price else 68000
    
    def calculate_expected_move(self):
        momentum = 0
        if self.last_price and self.current_price:
            momentum = (self.current_price - self.last_price) / self.last_price
        acc = self.get_accuracy()['overall'] / 100
        return momentum * (0.5 + acc * 0.5)
    
    def predict(self):
        now = datetime.now()
        m = now.minute
        s = now.second
        
        if m == 0 and s < 30:
            price = self.get_current_price()
            if price is None:
                return {'error': 'No price data'}
            
            expected_move = self.calculate_expected_move()
            
            safe = round((price * (1 + expected_move * 0.5) - 500) / 100) * 100
            modest = round((price * (1 + expected_move)) / 100) * 100
            aggressive = round((price * (1 + expected_move * 1.5) + 500) / 100) * 100
            
            acc = self.get_accuracy()['overall'] / 100
            base_conf = 50 + abs(expected_move) * 2500
            confidence = 50 + (base_conf - 50) * (0.5 + acc * 0.5)
            
            block_start = now.replace(minute=0, second=0, microsecond=0)
            key = block_start.isoformat()
            
            if key not in self.pending_predictions:
                pred = {
                    'id': key,
                    'start_time': block_start,
                    'end_time': block_start + timedelta(hours=1),
                    'start_price': price,
                    'tiers': {
                        'safe': {'price': safe, 'confidence': min(80, confidence * 0.8)},
                        'modest': {'price': modest, 'confidence': min(90, confidence)},
                        'aggressive': {'price': aggressive, 'confidence': min(70, confidence * 0.7)}
                    },
                    'expected_move': expected_move,
                    'resolved': False,
                    'correct': None
                }
                self.pending_predictions[key] = pred
                self.save_data()
                
                print(f"⏰ Hourly: {block_start.strftime('%H:%M')} → ${modest:,}")
                
                return {
                    'predicted_price': modest,
                    'tiers': {
                        'safe': {'price': safe, 'formatted': f"${safe:,}", 'confidence': pred['tiers']['safe']['confidence']},
                        'modest': {'price': modest, 'formatted': f"${modest:,}", 'confidence': pred['tiers']['modest']['confidence']},
                        'aggressive': {'price': aggressive, 'formatted': f"${aggressive:,}", 'confidence': pred['tiers']['aggressive']['confidence']}
                    }
                }
        
        key = now.replace(minute=0, second=0, microsecond=0).isoformat()
        if key in self.pending_predictions:
            p = self.pending_predictions[key]
            return {
                'predicted_price': p['tiers']['modest']['price'],
                'tiers': {
                    'safe': {'price': p['tiers']['safe']['price'], 'formatted': f"${p['tiers']['safe']['price']:,}", 'confidence': p['tiers']['safe']['confidence']},
                    'modest': {'price': p['tiers']['modest']['price'], 'formatted': f"${p['tiers']['modest']['price']:,}", 'confidence': p['tiers']['modest']['confidence']},
                    'aggressive': {'price': p['tiers']['aggressive']['price'], 'formatted': f"${p['tiers']['aggressive']['price']:,}", 'confidence': p['tiers']['aggressive']['confidence']}
                }
            }
        
        return {'message': 'Waiting for next hour'}
    
    def verify_pending(self):
        now = datetime.now()
        verified = 0
        
        for pid, pred in list(self.pending_predictions.items()):
            if now >= pred['end_time'] + timedelta(minutes=1):
                end_price = self.get_price_at_time(pred['end_time'])
                
                if end_price:
                    rounded = round(end_price / 100) * 100
                    tiers = pred['tiers']
                    distances = {
                        'safe': abs(tiers['safe']['price'] - rounded),
                        'modest': abs(tiers['modest']['price'] - rounded),
                        'aggressive': abs(tiers['aggressive']['price'] - rounded)
                    }
                    pred['closest_tier'] = min(distances, key=distances.get)
                    
                    actual_dir = 'UP' if rounded > pred['start_price'] else 'DOWN'
                    pred_dir = 'UP' if pred['tiers']['modest']['price'] > pred['start_price'] else 'DOWN'
                    pred['correct'] = (pred_dir == actual_dir)
                    pred['resolved'] = True
                    
                    self.total_predictions += 1
                    if pred['correct']:
                        self.correct_predictions += 1
                    self.accuracy_history.append(pred['correct'])
                    self.predictions.append(pred)
                    del self.pending_predictions[pid]
                    verified += 1
                    
                    r = "✅" if pred['correct'] else "❌"
                    print(f"📊 Hourly {pred['start_time'].strftime('%H:%M')}: {r} (closest to {pred['closest_tier']})")
                    self.save_data()
        
        return verified
    
    def get_accuracy(self):
        if self.total_predictions == 0:
            return {'overall': 0, 'recent': 0, 'total': 0, 'correct': 0}
        recent = list(self.accuracy_history)[-20:]
        recent_acc = sum(recent) / len(recent) * 100 if recent else 0
        return {
            'overall': (self.correct_predictions / self.total_predictions) * 100,
            'recent': recent_acc,
            'total': self.total_predictions,
            'correct': self.correct_predictions
        }
    
    def get_history(self, limit=10):
        recent = list(self.predictions)[-limit:]
        return [{
            'start_time': p['start_time'].isoformat(),
            'predicted': p['tiers']['modest']['price'],
            'actual': p.get('end_price'),
            'closest_tier': p.get('closest_tier'),
            'correct': p.get('correct')
        } for p in recent]
    
    def get_pending(self):
        return {
            pid: {
                'start_time': p['start_time'].isoformat(),
                'end_time': p['end_time'].isoformat(),
                'tiers': p['tiers'],
                'start_price': p['start_price']
            }
            for pid, p in self.pending_predictions.items()
        }
    
    def prune_old_data(self):
        cutoff = datetime.now() - timedelta(days=30)
        self.predictions = deque([p for p in self.predictions if p['start_time'] > cutoff], maxlen=2000)
        old_cutoff = datetime.now() - timedelta(hours=24)
        for pid, p in list(self.pending_predictions.items()):
            if p['start_time'] < old_cutoff:
                del self.pending_predictions[pid]
        self.save_data()
    
    def load_data(self):
        try:
            file = os.path.join(self.data_dir, 'hourly_data.pkl')
            if os.path.exists(file):
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    self.predictions = data.get('predictions', deque(maxlen=2000))
                    self.pending_predictions = data.get('pending', {})
                    self.total_predictions = data.get('total', 0)
                    self.correct_predictions = data.get('correct', 0)
                    self.accuracy_history = data.get('history', deque(maxlen=200))
                print(f"📚 Loaded {len(self.predictions)} hourly predictions")
                print(f"   Hourly accuracy: {self.get_accuracy()['overall']:.1f}%")
        except Exception as e:
            print(f"⚠️ Load error: {e}")
    
    def save_data(self):
        try:
            file = os.path.join(self.data_dir, 'hourly_data.pkl')
            data = {
                'predictions': self.predictions,
                'pending': self.pending_predictions,
                'total': self.total_predictions,
                'correct': self.correct_predictions,
                'history': self.accuracy_history
            }
            with open(file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ Save error: {e}")