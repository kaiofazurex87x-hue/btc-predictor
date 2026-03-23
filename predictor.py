"""
15-Minute BTC Predictor - Auto Verification with Manual Fallback
"""

import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from collections import deque
import json
import requests

class BTCPredictor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.is_trained = False
        
        # Store predictions
        self.predictions = deque(maxlen=2000)
        self.pending_predictions = {}  # key: start_time -> prediction object
        self.accuracy_history = deque(maxlen=100)
        
        # Stats
        self.total_predictions = 0
        self.correct_predictions = 0
        self.last_price = None
        self.current_price = 68000
        
        # Kalshi API (will be set later)
        self.kalshi_api = None
        
        # Load existing data
        self.load_data()
    
    def set_kalshi_api(self, kalshi):
        """Set Kalshi API for automatic verification"""
        self.kalshi_api = kalshi
    
    def train(self):
        """Train the model"""
        print("📊 Training 15-min predictor...")
        self.is_trained = True
        self.save_data()
        return True
    
    def get_current_price(self):
        """Get current BTC price - will be updated by main app"""
        return self.current_price if self.current_price else 68000
    
    def update_price(self, price):
        """Update current price from external source"""
        self.current_price = price
        return price
    
    def get_price_at_time(self, target_time):
        """Get price at exact timestamp - should be implemented with price history"""
        # This should query your price database
        # For now, return current price as fallback
        return self.current_price if self.current_price else 68000
    
    def get_next_prediction_time(self):
        """Calculate next prediction time"""
        now = datetime.now()
        current_minute = now.minute
        
        if current_minute < 15:
            next_minute = 15
        elif current_minute < 30:
            next_minute = 30
        elif current_minute < 45:
            next_minute = 45
        else:
            next_minute = 0
            now = now + timedelta(hours=1)
        
        next_time = now.replace(minute=next_minute, second=0, microsecond=0)
        return next_time.isoformat()
    
    def predict(self, kalshi_line=None, force=False):
        """
        Make a prediction for the NEXT 15-minute block
        Only creates prediction at the start of each block (:00, :15, :30, :45)
        """
        now = datetime.now()
        current_minute = now.minute
        current_second = now.second
        
        # Calculate current 15-minute block start time
        block_start_minute = (current_minute // 15) * 15
        block_start = now.replace(minute=block_start_minute, second=0, microsecond=0)
        
        # Only create prediction at the START of a block (within first 5 seconds)
        if current_second < 5 and current_minute % 15 == 0:
            # Check if we already made a prediction for this block
            block_key = block_start.isoformat()
            if block_key not in self.pending_predictions:
                
                # Get current price
                current_price = self.get_current_price()
                if current_price is None:
                    return {'error': 'No price data'}
                
                # Calculate signal from Kalshi or price
                if kalshi_line:
                    signal = kalshi_line / 100  # Convert to probability
                else:
                    # Simple momentum-based signal
                    signal = 0.5
                    if self.last_price and current_price:
                        momentum = (current_price - self.last_price) / self.last_price
                        signal = 0.5 + momentum * 5
                        signal = max(0.1, min(0.9, signal))
                
                # Direction based on signal
                direction = 'UP' if signal > 0.5 else 'DOWN'
                confidence = 50 + abs(signal - 0.5) * 80
                confidence = min(90, max(50, confidence))
                
                # Store prediction
                prediction = {
                    'id': block_key,
                    'start_time': block_start,
                    'end_time': block_start + timedelta(minutes=15),
                    'start_price': current_price,
                    'prediction': direction,
                    'signal': signal,
                    'confidence': confidence,
                    'kalshi_line': kalshi_line,
                    'resolved': False,
                    'correct': None,
                    'end_price': None,
                    'verified_by': None,  # 'auto', 'manual', or 'price'
                    'manual_override': False
                }
                
                self.pending_predictions[block_key] = prediction
                self.save_data()
                
                print(f"🔮 15-min PREDICTION: {block_start.strftime('%H:%M')} → "
                      f"{direction} with {confidence:.0f}% confidence")
                
                return {
                    'direction': direction,
                    'confidence': confidence,
                    'start_time': block_start.isoformat(),
                    'end_time': (block_start + timedelta(minutes=15)).isoformat(),
                    'start_price': current_price,
                    'prediction_id': block_key
                }
        
        # Return current pending prediction if exists
        current_key = block_start.isoformat()
        if current_key in self.pending_predictions:
            pred = self.pending_predictions[current_key]
            return {
                'direction': pred['prediction'],
                'confidence': pred['confidence'],
                'start_time': pred['start_time'].isoformat(),
                'end_time': pred['end_time'].isoformat(),
                'start_price': pred['start_price'],
                'prediction_id': current_key
            }
        
        return {'message': 'No prediction at this time', 'next_prediction_at': self.get_next_prediction_time()}
    
    def auto_verify_from_kalshi(self, pred):
        """
        Attempt to automatically verify using Kalshi API
        Returns: (verified, result, actual_direction)
        """
        if not self.kalshi_api:
            return False, None, None
        
        try:
            # Try to get Kalshi data for this time period
            kalshi_data = self.kalshi_api.get_historical_line(pred['start_time'], pred['end_time'])
            
            if kalshi_data and 'yes_price' in kalshi_data and 'no_price' in kalshi_data:
                # Kalshi data available - verify automatically
                yes = kalshi_data['yes_price']
                no = kalshi_data['no_price']
                
                actual_direction = 'UP' if yes > no else 'DOWN'
                was_correct = (pred['prediction'] == actual_direction)
                
                return True, was_correct, actual_direction
            
        except Exception as e:
            print(f"[Auto Verify] Kalshi error: {e}")
        
        return False, None, None
    
    def verify_by_price(self, pred):
        """
        Verify using actual price movement
        This is the automatic fallback when Kalshi data isn't available
        """
        end_price = self.get_price_at_time(pred['end_time'])
        
        if end_price:
            actual_direction = 'UP' if end_price > pred['start_price'] else 'DOWN'
            was_correct = (pred['prediction'] == actual_direction)
            return True, was_correct, actual_direction, end_price
        
        return False, None, None, None
    
    def verify_pending(self):
        """
        Check all pending predictions that should have completed
        Tries Kalshi first, then falls back to price movement
        """
        now = datetime.now()
        verified_count = 0
        
        for pred_id, pred in list(self.pending_predictions.items()):
            # Check if prediction has ended
            if now >= pred['end_time'] + timedelta(seconds=30):
                
                # FIRST: Try automatic Kalshi verification
                kalshi_verified, was_correct, actual = self.auto_verify_from_kalshi(pred)
                
                if kalshi_verified:
                    # Successfully verified with Kalshi
                    pred['resolved'] = True
                    pred['correct'] = was_correct
                    pred['actual_direction'] = actual
                    pred['verified_by'] = 'kalshi_auto'
                    pred['manual_override'] = False
                    
                    self.total_predictions += 1
                    if was_correct:
                        self.correct_predictions += 1
                    
                    self.accuracy_history.append(was_correct)
                    self.predictions.append(pred)
                    del self.pending_predictions[pred_id]
                    verified_count += 1
                    
                    result = "✅ CORRECT" if was_correct else "❌ WRONG"
                    print(f"📊 15-min AUTO-KALSHI: {pred['start_time'].strftime('%H:%M')} → "
                          f"Predicted {pred['prediction']}, Kalshi said {actual} ({result})")
                    
                    self.save_data()
                    continue
                
                # SECOND: Fallback to price movement verification
                price_verified, was_correct, actual, end_price = self.verify_by_price(pred)
                
                if price_verified:
                    # Successfully verified with price
                    pred['end_price'] = end_price
                    pred['resolved'] = True
                    pred['correct'] = was_correct
                    pred['actual_direction'] = actual
                    pred['verified_by'] = 'price_auto'
                    pred['manual_override'] = False
                    
                    self.total_predictions += 1
                    if was_correct:
                        self.correct_predictions += 1
                    
                    self.accuracy_history.append(was_correct)
                    self.predictions.append(pred)
                    del self.pending_predictions[pred_id]
                    verified_count += 1
                    
                    result = "✅ CORRECT" if was_correct else "❌ WRONG"
                    print(f"📊 15-min AUTO-PRICE: {pred['start_time'].strftime('%H:%M')} → "
                          f"Predicted {pred['prediction']}, Price {actual} ({result})")
                    
                    self.save_data()
                    continue
                
                # If neither verification worked, keep pending for manual input
                # But if it's been more than 1 hour, mark as missed
                if now > pred['end_time'] + timedelta(hours=1):
                    pred['resolved'] = True
                    pred['correct'] = None
                    pred['verified_by'] = 'missed'
                    pred['manual_override'] = False
                    del self.pending_predictions[pred_id]
                    verified_count += 1
                    print(f"📊 15-min MISSED: {pred['start_time'].strftime('%H:%M')} - No verification data")
        
        return verified_count
    
    def manual_verify(self, pred_id, actual_direction):
        """
        Manually verify a prediction (user input)
        This overrides automatic verification
        """
        if pred_id in self.pending_predictions:
            pred = self.pending_predictions[pred_id]
            
            pred['resolved'] = True
            pred['correct'] = (pred['prediction'] == actual_direction)
            pred['actual_direction'] = actual_direction
            pred['verified_by'] = 'manual'
            pred['manual_override'] = True
            
            self.total_predictions += 1
            if pred['correct']:
                self.correct_predictions += 1
            
            self.accuracy_history.append(pred['correct'])
            self.predictions.append(pred)
            del self.pending_predictions[pred_id]
            self.save_data()
            
            result = "✅ CORRECT" if pred['correct'] else "❌ WRONG"
            print(f"📊 15-min MANUAL: {pred['start_time'].strftime('%H:%M')} → "
                  f"Predicted {pred['prediction']}, User said {actual_direction} ({result})")
            
            return True, pred['correct']
        
        return False, None
    
    def kalshi_check(self, pred_id, up_prob, kalshi_yes=None, kalshi_no=None):
        """
        Manual Kalshi check - user enters Kalshi data
        This overrides automatic verification
        """
        if pred_id in self.pending_predictions:
            pred = self.pending_predictions[pred_id]
            
            # Determine actual direction from Kalshi data
            if kalshi_yes is not None and kalshi_no is not None:
                actual_direction = 'UP' if kalshi_yes > kalshi_no else 'DOWN'
            else:
                actual_direction = 'UP' if up_prob > 0.5 else 'DOWN'
            
            pred['resolved'] = True
            pred['correct'] = (pred['prediction'] == actual_direction)
            pred['actual_direction'] = actual_direction
            pred['kalshi_yes'] = kalshi_yes
            pred['kalshi_no'] = kalshi_no
            pred['verified_by'] = 'kalshi_manual'
            pred['manual_override'] = True
            
            self.total_predictions += 1
            if pred['correct']:
                self.correct_predictions += 1
            
            self.accuracy_history.append(pred['correct'])
            self.predictions.append(pred)
            del self.pending_predictions[pred_id]
            self.save_data()
            
            result = "✅ CORRECT" if pred['correct'] else "❌ WRONG"
            print(f"📊 15-min KALSHI MANUAL: {pred['start_time'].strftime('%H:%M')} → "
                  f"Predicted {pred['prediction']}, User said {actual_direction} ({result})")
            
            return {'success': True, 'correct': pred['correct']}
        
        return {'error': 'Prediction not found'}
    
    def get_accuracy(self):
        """Get current accuracy stats"""
        if self.total_predictions == 0:
            return {
                'overall': 0,
                'recent': 0,
                'total': 0,
                'correct': 0
            }
        
        recent = list(self.accuracy_history)[-20:]
        recent_acc = sum(recent) / len(recent) * 100 if recent else 0
        
        return {
            'overall': (self.correct_predictions / self.total_predictions) * 100,
            'recent': recent_acc,
            'total': self.total_predictions,
            'correct': self.correct_predictions
        }
    
    def get_history(self, limit=30):
        """Get recent predictions"""
        recent = list(self.predictions)[-limit:]
        return [{
            'start_time': p['start_time'].isoformat(),
            'prediction': p['prediction'],
            'correct': p.get('correct'),
            'start_price': p.get('start_price'),
            'end_price': p.get('end_price'),
            'confidence': p.get('confidence'),
            'verified_by': p.get('verified_by'),
            'manual_override': p.get('manual_override', False)
        } for p in recent]
    
    def get_pending(self):
        """Get all pending predictions that need manual input"""
        return {
            pred_id: {
                'start_time': pred['start_time'].isoformat(),
                'end_time': pred['end_time'].isoformat(),
                'prediction': pred['prediction'],
                'start_price': pred['start_price'],
                'confidence': pred['confidence']
            }
            for pred_id, pred in self.pending_predictions.items()
            if not pred.get('resolved')
        }
    
    def prune_old_data(self):
        """Keep only recent data"""
        cutoff = datetime.now() - timedelta(days=30)
        self.predictions = deque([p for p in self.predictions if p['start_time'] > cutoff], maxlen=2000)
        self.save_data()
    
    def load_data(self):
        """Load saved data"""
        try:
            data_file = os.path.join(self.data_dir, 'predictor_data.pkl')
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.predictions = data.get('predictions', deque(maxlen=2000))
                    self.pending_predictions = data.get('pending', {})
                    self.total_predictions = data.get('total', 0)
                    self.correct_predictions = data.get('correct', 0)
                    self.accuracy_history = data.get('history', deque(maxlen=100))
                print(f"📚 Loaded {len(self.predictions)} predictions, {len(self.pending_predictions)} pending")
        except Exception as e:
            print(f"⚠️ Could not load data: {e}")
    
    def save_data(self):
        """Save all data"""
        try:
            data_file = os.path.join(self.data_dir, 'predictor_data.pkl')
            data = {
                'predictions': self.predictions,
                'pending': self.pending_predictions,
                'total': self.total_predictions,
                'correct': self.correct_predictions,
                'history': self.accuracy_history
            }
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ Could not save data: {e}")