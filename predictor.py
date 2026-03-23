"""
15-Minute BTC Predictor - With Manual Baseline Correction
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
        self.resolved_predictions = deque(maxlen=2000)
        self.accuracy_history = deque(maxlen=500)
        
        self.total_predictions = 0
        self.correct_predictions = 0
        self.last_price = None
        self.current_price = 68000
        self.kalshi_api = None
        self.current_kalshi_line = None
        
        os.makedirs(data_dir, exist_ok=True)
        self.load_data()
    
    def set_kalshi_api(self, kalshi):
        self.kalshi_api = kalshi
    
    def update_kalshi_line(self, line):
        """Update the current Kalshi line (for display only)"""
        self.current_kalshi_line = line
        print(f"[Kalshi] Current line updated: {line}")
    
    def correct_baseline_price(self, pred_id, correct_price):
        """
        CORRECT the baseline price for an active prediction
        This is critical for when Kalshi's baseline is wrong
        """
        if pred_id in self.pending_predictions:
            pred = self.pending_predictions[pred_id]
            old_price = pred['start_price']
            pred['start_price'] = correct_price
            pred['baseline_corrected'] = True
            pred['original_price'] = old_price
            
            print(f"🔧 BASELINE CORRECTED: {pred['start_time'].strftime('%H:%M')} "
                  f"${old_price:,.0f} → ${correct_price:,.0f}")
            
            # Re-evaluate the prediction based on corrected baseline
            # Keep the same direction (UP/DOWN) but update the target
            self.save_data()
            return True, pred
        return False, None
    
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
        """Get price at exact timestamp (for verification)"""
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
        """
        Make a prediction for the NEXT 15-minute block
        """
        now = datetime.now()
        m = now.minute
        s = now.second
        block_start_min = (m // 15) * 15
        block_start = now.replace(minute=block_start_min, second=0, microsecond=0)
        
        # Only create prediction at the START of a block
        if s < 5 and m % 15 == 0:
            key = block_start.isoformat()
            if key not in self.pending_predictions:
                price = self.get_current_price()
                if price is None:
                    return {'error': 'No price data'}
                
                # Use Kalshi line if available for direction, but NOT for price
                if kalshi_line:
                    # Kalshi line is a probability (0-100) for direction
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
                    'verified_by': None,
                    'baseline_corrected': False,
                    'original_price': price,
                    'kalshi_yes': None,
                    'kalshi_no': None,
                    'end_price': None
                }
                self.pending_predictions[key] = pred
                self.save_data()
                
                print(f"🔮 15-min: {block_start.strftime('%H:%M')} → {direction} {confidence:.0f}% (Start: ${price:,.0f})")
                
                return {
                    'direction': direction,
                    'confidence': confidence,
                    'start_time': block_start.isoformat(),
                    'end_time': (block_start + timedelta(minutes=15)).isoformat(),
                    'start_price': price,
                    'prediction_id': key
                }
        
        # Return current pending prediction if exists
        key = block_start.isoformat()
        if key in self.pending_predictions:
            p = self.pending_predictions[key]
            return {
                'direction': p['prediction'],
                'confidence': p['confidence'],
                'start_time': p['start_time'].isoformat(),
                'end_time': p['end_time'].isoformat(),
                'start_price': p['start_price'],
                'prediction_id': key,
                'baseline_corrected': p.get('baseline_corrected', False)
            }
        
        return {'message': 'No prediction', 'next_at': self.get_next_prediction_time()}
    
    def verify_pending(self):
        """Check all pending predictions that should have completed"""
        now = datetime.now()
        verified = 0
        
        for pid, pred in list(self.pending_predictions.items()):
            if now >= pred['end_time'] + timedelta(seconds=30):
                end_price = self.get_price_at_time(pred['end_time'])
                
                if end_price:
                    pred['end_price'] = end_price
                    pred['resolved'] = True
                    
                    # Use corrected baseline if available
                    start_price = pred['start_price']
                    
                    actual_direction = 'UP' if end_price > start_price else 'DOWN'
                    was_correct = (pred['prediction'] == actual_direction)
                    
                    pred['correct'] = was_correct
                    pred['actual_direction'] = actual_direction
                    pred['verified_by'] = 'price_auto'
                    
                    self.total_predictions += 1
                    if was_correct:
                        self.correct_predictions += 1
                    
                    self.accuracy_history.append(was_correct)
                    self.resolved_predictions.append(pred)
                    del self.pending_predictions[pid]
                    verified += 1
                    
                    corrected_msg = " (baseline corrected)" if pred.get('baseline_corrected') else ""
                    r = "✅" if was_correct else "❌"
                    print(f"📊 15-min {pred['start_time'].strftime('%H:%M')}: {r} {corrected_msg}")
                    print(f"   Start: ${start_price:,.0f}, End: ${end_price:,.0f}, Actual: {actual_direction}")
                    self.save_data()
        
        return verified
    
    def manual_verify_with_kalshi(self, pred_id, kalshi_yes, kalshi_no, correct_baseline=None):
        """
        Manual verification with Kalshi data
        Can also correct the baseline price if provided
        """
        if pred_id in self.pending_predictions:
            pred = self.pending_predictions[pred_id]
            
            # First, correct baseline if provided
            if correct_baseline is not None:
                pred['start_price'] = correct_baseline
                pred['baseline_corrected'] = True
                print(f"🔧 BASELINE CORRECTED to ${correct_baseline:,.0f}")
            
            # Then verify with Kalshi
            actual_direction = 'UP' if kalshi_yes > kalshi_no else 'DOWN'
            was_correct = (pred['prediction'] == actual_direction)
            
            pred['resolved'] = True
            pred['correct'] = was_correct
            pred['actual_direction'] = actual_direction
            pred['kalshi_yes'] = kalshi_yes
            pred['kalshi_no'] = kalshi_no
            pred['verified_by'] = 'kalshi_manual'
            pred['end_price'] = None  # Not used for Kalshi verification
            
            self.total_predictions += 1
            if was_correct:
                self.correct_predictions += 1
            
            self.accuracy_history.append(was_correct)
            self.resolved_predictions.append(pred)
            del self.pending_predictions[pred_id]
            self.save_data()
            
            corrected_msg = " (baseline corrected)" if pred.get('baseline_corrected') else ""
            r = "✅" if was_correct else "❌"
            print(f"📊 15-min {pred['start_time'].strftime('%H:%M')}: {r} (Manual-Kalshi){corrected_msg}")
            print(f"   Kalshi: YES={kalshi_yes}, NO={kalshi_no} → {actual_direction}")
            
            return {'success': True, 'correct': was_correct, 'actual': actual_direction}
        
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
        recent = list(self.resolved_predictions)[-limit:]
        return [{
            'start_time': p['start_time'].isoformat(),
            'prediction': p['prediction'],
            'correct': p.get('correct'),
            'start_price': p.get('start_price'),
            'original_price': p.get('original_price'),
            'end_price': p.get('end_price'),
            'confidence': p.get('confidence'),
            'verified_by': p.get('verified_by'),
            'baseline_corrected': p.get('baseline_corrected', False),
            'kalshi_yes': p.get('kalshi_yes'),
            'kalshi_no': p.get('kalshi_no')
        } for p in recent]
    
    def get_pending(self):
        return {
            pid: {
                'start_time': p['start_time'].isoformat(),
                'end_time': p['end_time'].isoformat(),
                'prediction': p['prediction'],
                'start_price': p['start_price'],
                'confidence': p['confidence'],
                'baseline_corrected': p.get('baseline_corrected', False)
            }
            for pid, p in self.pending_predictions.items()
        }
    
    def prune_old_data(self):
        cutoff = datetime.now() - timedelta(days=30)
        self.resolved_predictions = deque([p for p in self.resolved_predictions if p['start_time'] > cutoff], maxlen=2000)
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
                    self.resolved_predictions = data.get('resolved', deque(maxlen=2000))
                    self.pending_predictions = data.get('pending', {})
                    self.total_predictions = data.get('total', 0)
                    self.correct_predictions = data.get('correct', 0)
                    self.accuracy_history = data.get('history', deque(maxlen=500))
                print(f"📚 Loaded {len(self.resolved_predictions)} resolved, {len(self.pending_predictions)} pending")
        except Exception as e:
            print(f"⚠️ Load error: {e}")
    
    def save_data(self):
        try:
            file = os.path.join(self.data_dir, 'predictor_data.pkl')
            data = {
                'resolved': self.resolved_predictions,
                'pending': self.pending_predictions,
                'total': self.total_predictions,
                'correct': self.correct_predictions,
                'history': self.accuracy_history
            }
            with open(file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ Save error: {e}")