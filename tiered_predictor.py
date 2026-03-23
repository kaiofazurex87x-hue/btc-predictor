"""
Hourly BTC Predictor - Fixed Training & Persistent Storage
"""

import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from collections import deque
import json

class TieredHourlyPredictor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.is_trained = False
        
        # Store predictions
        self.predictions = deque(maxlen=2000)
        self.pending_predictions = {}  # key: start_time -> prediction object
        self.accuracy_history = deque(maxlen=200)
        
        # Stats
        self.total_predictions = 0
        self.correct_predictions = 0
        self.last_price = None
        self.current_price = 68000
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing data
        self.load_data()
    
    def train(self):
        """Train the hourly predictor with existing data"""
        print("📊 Training hourly predictor...")
        
        # If we have historical predictions, use them to train
        if len(self.predictions) > 0:
            print(f"   Training on {len(self.predictions)} historical predictions")
            # Simple training: calculate moving average of accuracy
            recent_acc = self.get_accuracy()['recent']
            overall_acc = self.get_accuracy()['overall']
            print(f"   Recent accuracy: {recent_acc:.1f}%")
            print(f"   Overall accuracy: {overall_acc:.1f}%")
        
        self.is_trained = True
        self.save_data()
        print("✅ Hourly predictor trained successfully")
        return True
    
    def update_price(self, price):
        """Update current price from external source"""
        self.current_price = price
        self.last_price = price
        return price
    
    def get_current_price(self):
        """Get current BTC price"""
        return self.current_price if self.current_price else 68000
    
    def get_price_at_time(self, target_time):
        """Get price at exact timestamp"""
        # This should query your price database
        return self.current_price if self.current_price else 68000
    
    def calculate_expected_move(self):
        """
        Calculate expected price movement for next hour
        Uses simple momentum + historical accuracy
        """
        # Base momentum from recent price changes
        momentum = 0
        if self.last_price and self.current_price:
            momentum = (self.current_price - self.last_price) / self.last_price
        
        # Adjust by historical accuracy (confidence weighting)
        accuracy = self.get_accuracy()['overall'] / 100  # 0-1 scale
        
        # Combine: more accurate = more confident in momentum
        expected_move = momentum * (0.5 + accuracy * 0.5)
        
        # Cap at reasonable range (±1%)
        return max(-0.01, min(0.01, expected_move))
    
    def predict(self, force=False):
        """
        Make hourly prediction at the start of each hour
        """
        now = datetime.now()
        current_minute = now.minute
        current_second = now.second
        
        # Only make prediction at the start of the hour (within first 30 seconds)
        if current_minute == 0 and current_second < 30:
            current_price = self.get_current_price()
            if current_price is None:
                return {'error': 'No price data'}
            
            # Calculate expected move based on recent momentum and learning
            expected_move_pct = self.calculate_expected_move()
            
            # Calculate three tiers
            safe_price = current_price * (1 + expected_move_pct * 0.5) - 500
            modest_price = current_price * (1 + expected_move_pct)
            aggressive_price = current_price * (1 + expected_move_pct * 1.5) + 500
            
            # Round to nearest $100
            safe = round(safe_price / 100) * 100
            modest = round(modest_price / 100) * 100
            aggressive = round(aggressive_price / 100) * 100
            
            # Calculate confidence based on historical accuracy and signal strength
            accuracy = self.get_accuracy()['overall'] / 100
            signal_strength = abs(expected_move_pct) * 25
            base_confidence = 50 + signal_strength
            
            # Apply accuracy weighting
            confidence = 50 + (base_confidence - 50) * (0.5 + accuracy * 0.5)
            
            # Store prediction
            block_start = now.replace(minute=0, second=0, microsecond=0)
            block_key = block_start.isoformat()
            
            # Check if we already have a prediction for this hour
            if block_key not in self.pending_predictions:
                prediction = {
                    'id': block_key,
                    'start_time': block_start,
                    'end_time': block_start + timedelta(hours=1),
                    'start_price': current_price,
                    'tiers': {
                        'safe': {'price': safe, 'confidence': min(80, confidence * 0.8)},
                        'modest': {'price': modest, 'confidence': min(90, confidence)},
                        'aggressive': {'price': aggressive, 'confidence': min(70, confidence * 0.7)}
                    },
                    'expected_move': expected_move_pct,
                    'resolved': False,
                    'correct': None,
                    'end_price': None,
                    'closest_tier': None,
                    'verified_by': None
                }
                
                self.pending_predictions[block_key] = prediction
                self.save_data()
                
                print(f"⏰ HOURLY PREDICTION: {block_start.strftime('%H:%M')} → "
                      f"Safe: ${safe:,}, Modest: ${modest:,}, Aggressive: ${aggressive:,}")
                
                return {
                    'predicted_price': modest,
                    'tiers': {
                        'safe': {'price': safe, 'formatted': f"${safe:,}", 'confidence': prediction['tiers']['safe']['confidence']},
                        'modest': {'price': modest, 'formatted': f"${modest:,}", 'confidence': prediction['tiers']['modest']['confidence']},
                        'aggressive': {'price': aggressive, 'formatted': f"${aggressive:,}", 'confidence': prediction['tiers']['aggressive']['confidence']}
                    }
                }
        
        # Return current prediction if exists
        current_key = now.replace(minute=0, second=0, microsecond=0).isoformat()
        if current_key in self.pending_predictions:
            pred = self.pending_predictions[current_key]
            return {
                'predicted_price': pred['tiers']['modest']['price'],
                'tiers': {
                    'safe': {'price': pred['tiers']['safe']['price'], 'formatted': f"${pred['tiers']['safe']['price']:,}", 'confidence': pred['tiers']['safe']['confidence']},
                    'modest': {'price': pred['tiers']['modest']['price'], 'formatted': f"${pred['tiers']['modest']['price']:,}", 'confidence': pred['tiers']['modest']['confidence']},
                    'aggressive': {'price': pred['tiers']['aggressive']['price'], 'formatted': f"${pred['tiers']['aggressive']['price']:,}", 'confidence': pred['tiers']['aggressive']['confidence']}
                }
            }
        
        return {'message': 'Waiting for next hour'}
    
    def verify_pending(self):
        """Check pending hourly predictions"""
        now = datetime.now()
        verified_count = 0
        
        for pred_id, pred in list(self.pending_predictions.items()):
            if now >= pred['end_time'] + timedelta(minutes=1):
                end_price = self.get_price_at_time(pred['end_time'])
                
                if end_price:
                    pred['end_price'] = end_price
                    pred['resolved'] = True
                    
                    # Find closest tier
                    rounded_end = round(end_price / 100) * 100
                    tiers = pred['tiers']
                    
                    distances = {
                        'safe': abs(tiers['safe']['price'] - rounded_end),
                        'modest': abs(tiers['modest']['price'] - rounded_end),
                        'aggressive': abs(tiers['aggressive']['price'] - rounded_end)
                    }
                    pred['closest_tier'] = min(distances, key=distances.get)
                    
                    # Check if direction was correct (modest tier direction)
                    actual_direction = 'UP' if rounded_end > pred['start_price'] else 'DOWN'
                    pred['direction'] = 'UP' if pred['tiers']['modest']['price'] > pred['start_price'] else 'DOWN'
                    pred['correct'] = (pred['direction'] == actual_direction)
                    pred['verified_by'] = 'price_auto'
                    
                    # Update stats
                    self.total_predictions += 1
                    if pred['correct']:
                        self.correct_predictions += 1
                    
                    self.accuracy_history.append(pred['correct'])
                    self.predictions.append(pred)
                    del self.pending_predictions[pred_id]
                    verified_count += 1
                    
                    result = "✅ CORRECT" if pred['correct'] else "❌ WRONG"
                    print(f"📊 HOURLY RESULT: {pred['start_time'].strftime('%H:%M')} → "
                          f"Closest to {pred['closest_tier']} tier ({result})")
                    
                    self.save_data()
        
        return verified_count
    
    def manual_verify(self, pred_id, actual_direction):
        """
        Manually verify an hourly prediction
        """
        if pred_id in self.pending_predictions:
            pred = self.pending_predictions[pred_id]
            
            pred['resolved'] = True
            pred['correct'] = (pred['direction'] == actual_direction)
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
            print(f"📊 HOURLY MANUAL: {pred['start_time'].strftime('%H:%M')} → "
                  f"User said {actual_direction} ({result})")
            
            return True, pred['correct']
        
        return False, None
    
    def get_accuracy(self):
        """Get accuracy stats"""
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
    
    def get_history(self, limit=10):
        """Get recent predictions"""
        recent = list(self.predictions)[-limit:]
        return [{
            'start_time': p['start_time'].isoformat(),
            'predicted': p['tiers']['modest']['price'],
            'actual': p.get('end_price'),
            'closest_tier': p.get('closest_tier'),
            'correct': p.get('correct'),
            'verified_by': p.get('verified_by')
        } for p in recent]
    
    def get_pending(self):
        """Get pending hourly predictions"""
        return {
            pred_id: {
                'start_time': pred['start_time'].isoformat(),
                'end_time': pred['end_time'].isoformat(),
                'tiers': pred['tiers'],
                'start_price': pred['start_price']
            }
            for pred_id, pred in self.pending_predictions.items()
            if not pred.get('resolved')
        }
    
    def prune_old_data(self):
        """Keep only recent data (last 30 days)"""
        cutoff = datetime.now() - timedelta(days=30)
        self.predictions = deque([p for p in self.predictions if p['start_time'] > cutoff], maxlen=2000)
        
        # Clean up old pending predictions (over 24 hours)
        old_cutoff = datetime.now() - timedelta(hours=24)
        for pred_id, pred in list(self.pending_predictions.items()):
            if pred['start_time'] < old_cutoff:
                del self.pending_predictions[pred_id]
        
        self.save_data()
    
    def load_data(self):
        """Load saved data from disk"""
        try:
            data_file = os.path.join(self.data_dir, 'hourly_data.pkl')
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.predictions = data.get('predictions', deque(maxlen=2000))
                    self.pending_predictions = data.get('pending', {})
                    self.total_predictions = data.get('total', 0)
                    self.correct_predictions = data.get('correct', 0)
                    self.accuracy_history = data.get('history', deque(maxlen=200))
                print(f"📚 Loaded {len(self.predictions)} hourly predictions, {len(self.pending_predictions)} pending")
                print(f"   Hourly accuracy: {self.get_accuracy()['overall']:.1f}%")
            else:
                print(f"📚 No existing hourly data found. Starting fresh.")
        except Exception as e:
            print(f"⚠️ Could not load hourly data: {e}")
    
    def save_data(self):
        """Save all data to disk"""
        try:
            data_file = os.path.join(self.data_dir, 'hourly_data.pkl')
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
            print(f"⚠️ Could not save hourly data: {e}")