"""
15-Minute BTC Predictor - No XGBoost
"""

import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from collections import deque


class BTCPredictor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.is_trained = False
        
        self.predictions = deque(maxlen=5000)
        self.pending_predictions = {}
        self.accuracy_history = deque(maxlen=500)
        
        self.total_predictions = 0
        self.correct_predictions = 0
        self.last_price = None
        self.current_price = 68000
        self.kalshi_api = None
        
        os.makedirs(data_dir, exist_ok=True)
        self.load_data()
    
    def set_kalshi_api(self, kalshi):
        self.kalshi_api = kalshi
    
    def train(self):
        print("📊 Training 15-min predictor...")
        self.is_trained = True
        self.save_data()
        return True
    
    def get_current_price(self):
        return self.current_price if self.current_price else 68000
    
    def update_price(self, price):
        self.current_price = price
        self.last_price = price
        return price
    
    def get_price_at_time(self, target_time):
        return self.current_price if self.current_price else 68000
    
    def get_next_prediction_time(self):
        now = datetime.now()
        m = now.minute
        if m < 15:
            next_m = 15
        elif m < 30:
            next_m = 30
        elif m < 45:
            next_m = 45
        else:
            next_m = 0
            now = now + timedelta(hours=1)
        return now.replace(minute=next_m, second=0, microsecond=0).isoformat()
    
    def predict(self, kalshi_line=None):
        now = datetime.now()
        m = now.minute
        s = now.second
        block_start_min = (m // 15) * 15
        block_start = now.replace(minute=block_start_min, second=0, microsecond=0)
        
        if s < 5 and m % 15 == 0:
            key = block_start.isoformat()
            if key not in self.pending_predictions:
                price = self.get_current_price()
                if price is None:
                    return {'error': 'No price data'}
                
                if kalshi_line:
                    signal = kalshi_line / 100
                else:
                    signal = 0.5
                    if self.last_price and price:
                        momentum = (price - self.last_price) / self.last_price
                        signal = 0.5 + momentum * 5
                        signal = max(0.1, min(0.9, signal))
                
                direction = 'UP' if signal > 0.5 else 'DOWN'
                confidence = 50 + abs(signal - 0.5) * 80
                confidence = min(90, max(50, confidence))
                
                pred = {
                    'id': key,
                    'start_time': block_start,
                    'end_time': block_start + timedelta(minutes=15),
                    'start_price': price,
                    'prediction': direction,
                    'signal': signal,
                    'confidence': confidence,
                    'resolved': False,
                    'correct': None,
                    'verified_by': None
                }
                self.pending_predictions[key] = pred
                self.save_data()
                
                print(f"🔮 15-min: {block_start.strftime('%H:%M')} → {direction} {confidence:.0f}%")
                
                return {
                    'direction': direction,
                    'confidence': confidence,
                    'start_time': block_start.isoformat(),
                    'end_time': (block_start + timedelta(minutes=15)).isoformat(),
                    'start_price': price,
                    'prediction_id': key
                }
        
        key = block_start.isoformat()
        if key in self.pending_predictions:
            p = self.pending_predictions[key]
            return {
                'direction': p['prediction'],
                'confidence': p['confidence'],
                'start_time': p['start_time'].isoformat(),
                'end_time': p['end_time'].isoformat(),
                'start_price': p['start_price'],
                'prediction_id': key
            }
        
        return {'message': 'No prediction', 'next_at': self.get_next_prediction_time()}
    
    def auto_verify_from_kalshi(self, pred):
        if not self.kalshi_api:
            return False, None, None
        try:
            data = self.kalshi_api.get_historical_line(pred['start_time'], pred['end_time'])
            if data and 'yes_price' in data and 'no_price' in data:
                actual = 'UP' if data['yes_price'] > data['no_price'] else 'DOWN'
                correct = (pred['prediction'] == actual)
                return True, correct, actual
        except:
            pass
        return False, None, None
    
    def verify_by_price(self, pred):
        end_price = self.get_price_at_time(pred['end_time'])
        if end_price:
            actual = 'UP' if end_price > pred['start_price'] else 'DOWN'
            correct = (pred['prediction'] == actual)
            return True, correct, actual, end_price
        return False, None, None, None
    
    def verify_pending(self):
        now = datetime.now()
        verified = 0
        
        for pid, pred in list(self.pending_predictions.items()):
            if now >= pred['end_time'] + timedelta(seconds=30):
                k_ok, k_correct, k_actual = self.auto_verify_from_kalshi(pred)
                
                if k_ok:
                    pred['resolved'] = True
                    pred['correct'] = k_correct
                    pred['actual_direction'] = k_actual
                    pred['verified_by'] = 'kalshi_auto'
                    self.total_predictions += 1
                    if k_correct:
                        self.correct_predictions += 1
                    self.accuracy_history.append(k_correct)
                    self.predictions.append(pred)
                    del self.pending_predictions[pid]
                    verified += 1
                    r = "✅" if k_correct else "❌"
                    print(f"📊 15-min {pred['start_time'].strftime('%H:%M')}: {r} (Kalshi)")
                    self.save_data()
                    continue
                
                p_ok, p_correct, p_actual, p_price = self.verify_by_price(pred)
                
                if p_ok:
                    pred['end_price'] = p_price
                    pred['resolved'] = True
                    pred['correct'] = p_correct
                    pred['actual_direction'] = p_actual
                    pred['verified_by'] = 'price_auto'
                    self.total_predictions += 1
                    if p_correct:
                        self.correct_predictions += 1
                    self.accuracy_history.append(p_correct)
                    self.predictions.append(pred)
                    del self.pending_predictions[pid]
                    verified += 1
                    r = "✅" if p_correct else "❌"
                    print(f"📊 15-min {pred['start_time'].strftime('%H:%M')}: {r} (Price)")
                    self.save_data()
                    continue
                
                if now > pred['end_time'] + timedelta(hours=1):
                    pred['resolved'] = True
                    pred['verified_by'] = 'missed'
                    del self.pending_predictions[pid]
                    verified += 1
        
        return verified
    
    def manual_verify(self, pred_id, actual_direction):
        if pred_id in self.pending_predictions:
            pred = self.pending_predictions[pred_id]
            pred['resolved'] = True
            pred['correct'] = (pred['prediction'] == actual_direction)
            pred['actual_direction'] = actual_direction
            pred['verified_by'] = 'manual'
            self.total_predictions += 1
            if pred['correct']:
                self.correct_predictions += 1
            self.accuracy_history.append(pred['correct'])
            self.predictions.append(pred)
            del self.pending_predictions[pred_id]
            self.save_data()
            return True, pred['correct']
        return False, None
    
    def kalshi_check(self, pred_id, up_prob, kalshi_yes=None, kalshi_no=None):
        if pred_id in self.pending_predictions:
            pred = self.pending_predictions[pred_id]
            if kalshi_yes is not None and kalshi_no is not None:
                actual = 'UP' if kalshi_yes > kalshi_no else 'DOWN'
            else:
                actual = 'UP' if up_prob > 0.5 else 'DOWN'
            
            pred['resolved'] = True
            pred['correct'] = (pred['prediction'] == actual)
            pred['actual_direction'] = actual
            pred['verified_by'] = 'kalshi_manual'
            self.total_predictions += 1
            if pred['correct']:
                self.correct_predictions += 1
            self.accuracy_history.append(pred['correct'])
            self.predictions.append(pred)
            del self.pending_predictions[pred_id]
            self.save_data()
            return {'success': True, 'correct': pred['correct']}
        return {'error': 'Prediction not found'}
    
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
    
    def get_history(self, limit=30):
        recent = list(self.predictions)[-limit:]
        return [{
            'start_time': p['start_time'].isoformat(),
            'prediction': p['prediction'],
            'correct': p.get('correct'),
            'start_price': p.get('start_price'),
            'end_price': p.get('end_price'),
            'confidence': p.get('confidence'),
            'verified_by': p.get('verified_by')
        } for p in recent]
    
    def get_pending(self):
        return {
            pid: {
                'start_time': p['start_time'].isoformat(),
                'end_time': p['end_time'].isoformat(),
                'prediction': p['prediction'],
                'start_price': p['start_price'],
                'confidence': p['confidence']
            }
            for pid, p in self.pending_predictions.items()
        }
    
    def prune_old_data(self):
        cutoff = datetime.now() - timedelta(days=30)
        self.predictions = deque([p for p in self.predictions if p['start_time'] > cutoff], maxlen=5000)
        old_cutoff = datetime.now() - timedelta(hours=24)
        for pid, p in list(self.pending_predictions.items()):
            if p['start_time'] < old_cutoff:
                del self.pending_predictions[pid]
        self.save_data()
    
    def load_data(self):
        try:
            file = os.path.join(self.data_dir, 'predictor_data.pkl')
            if os.path.exists(file):
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    self.predictions = data.get('predictions', deque(maxlen=5000))
                    self.pending_predictions = data.get('pending', {})
                    self.total_predictions = data.get('total', 0)
                    self.correct_predictions = data.get('correct', 0)
                    self.accuracy_history = data.get('history', deque(maxlen=500))
                print(f"📚 Loaded {len(self.predictions)} predictions, {len(self.pending_predictions)} pending")
        except Exception as e:
            print(f"⚠️ Load error: {e}")
    
    def save_data(self):
        try:
            file = os.path.join(self.data_dir, 'predictor_data.pkl')
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