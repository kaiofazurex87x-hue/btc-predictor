"""
BTC Predictor - Main Application
"""

import os
import threading
import time
import bcrypt
import pytz
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user

import config
from predictor import BTCPredictor
from tiered_predictor import TieredHourlyPredictor
from whale_tracker import WhaleTracker
from kalshi_api import KalshiAPI

# Initialize Flask app
app = Flask(__name__,
            template_folder=os.path.join(config.BASE_DIR, 'templates'))
app.config['SECRET_KEY'] = config.SECRET_KEY

# Initialize components
predictor = BTCPredictor(data_dir=config.DATA_DIR)
tiered = TieredHourlyPredictor(data_dir=config.DATA_DIR)
whale = WhaleTracker()
kalshi = KalshiAPI()

# Connect Kalshi to predictor
predictor.set_kalshi_api(kalshi)

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):
    def __init__(self, id):
        self.id = id


@login_manager.user_loader
def load_user(user_id):
    return User(user_id) if user_id == '1' else None


def verify_password(password):
    """Verify admin password"""
    if not config.ADMIN_PASSWORD_HASH:
        return False
    try:
        return bcrypt.checkpw(password.encode(), config.ADMIN_PASSWORD_HASH.encode())
    except Exception:
        return False


def get_kalshi_15min():
    """Get Kalshi 15-min line"""
    try:
        return kalshi.get_btc_15min_line()
    except Exception as e:
        print(f"[Kalshi] Error: {e}")
    return None


def get_kalshi_hourly():
    """Get Kalshi hourly line"""
    try:
        return kalshi.get_btc_hourly_line()
    except Exception as e:
        print(f"[Kalshi] Error: {e}")
    return None


# ──────────────────────────────────────────────────────────────
# BACKGROUND THREADS
# ──────────────────────────────────────────────────────────────


def auto_predict_loop():
    """Auto-predict every 15 minutes at :00, :15, :30, :45"""
    TZ = pytz.timezone(config.TIMEZONE)
    
    while not predictor.is_trained:
        time.sleep(30)
    
    while True:
        now = datetime.now(pytz.utc).astimezone(TZ)
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
        
        next_run = now.replace(minute=next_minute, second=5, microsecond=0)
        wait = (next_run - datetime.now(pytz.utc).astimezone(TZ)).total_seconds()
        
        time.sleep(max(wait, 1))
        
        try:
            line = get_kalshi_15min()
            result = predictor.predict(kalshi_line=line)
            if 'direction' in result:
                print(f"[15-min] {result['direction']} {result['confidence']:.0f}%")
        except Exception as e:
            print(f"[Auto] Error: {e}")


def auto_hourly_loop():
    """Auto-predict every hour"""
    TZ = pytz.timezone(config.TIMEZONE)
    
    while not tiered.is_trained:
        time.sleep(30)
    
    while True:
        now = datetime.now(pytz.utc).astimezone(TZ)
        next_run = now.replace(minute=0, second=10, microsecond=0) + timedelta(hours=1)
        wait = (next_run - datetime.now(pytz.utc).astimezone(TZ)).total_seconds()
        
        time.sleep(max(wait, 1))
        
        try:
            result = tiered.predict()
            if 'tiers' in result:
                print(f"[Hourly] Targets: {result['tiers']['modest']['formatted']}")
        except Exception as e:
            print(f"[Hourly] Error: {e}")


def verify_loop():
    """Verify predictions at the end of each block"""
    TZ = pytz.timezone(config.TIMEZONE)
    
    while True:
        now = datetime.now(pytz.utc).astimezone(TZ)
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
        
        next_verify = now.replace(minute=next_minute, second=10, microsecond=0)
        wait = (next_verify - datetime.now(pytz.utc).astimezone(TZ)).total_seconds()
        
        time.sleep(max(wait, 1))
        
        try:
            n15 = predictor.verify_pending()
            nhr = tiered.verify_pending()
            if n15 or nhr:
                print(f"[Verify] 15-min: {n15}, Hourly: {nhr}")
        except Exception as e:
            print(f"[Verify] Error: {e}")


def price_update_loop():
    """Update BTC price every minute"""
    import requests
    
    while True:
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                price = data['bitcoin']['usd']
                predictor.update_price(price)
                tiered.update_price(price)
                print(f"[Price] BTC: ${price:,.2f}")
        except Exception as e:
            print(f"[Price] Error: {e}")
        
        time.sleep(60)


def prune_loop():
    """Prune old data every 6 hours"""
    while True:
        time.sleep(6 * 3600)
        try:
            predictor.prune_old_data()
            tiered.prune_old_data()
            print("[Prune] Old data pruned")
        except Exception as e:
            print(f"[Prune] Error: {e}")


# Start background threads
threading.Thread(target=auto_predict_loop, daemon=True).start()
threading.Thread(target=auto_hourly_loop, daemon=True).start()
threading.Thread(target=verify_loop, daemon=True).start()
threading.Thread(target=price_update_loop, daemon=True).start()
threading.Thread(target=prune_loop, daemon=True).start()


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if verify_password(request.form.get('password', '')):
            login_user(User('1'))
            return redirect(url_for('dashboard'))
        flash('Invalid password', 'error')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    try:
        body = request.get_json(silent=True) or {}
        return jsonify(predictor.predict(kalshi_line=body.get('kalshi_line')))
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/kalshi_check', methods=['POST'])
@login_required
def api_kalshi_check():
    try:
        body = request.get_json(silent=True) or {}
        pred_id = body.get('pred_id')
        up_prob = float(body.get('up_prob', 0.5))
        yes = body.get('kalshi_yes')
        no = body.get('kalshi_no')
        if yes is not None:
            yes = float(yes)
        if no is not None:
            no = float(no)
        return jsonify(predictor.kalshi_check(pred_id, up_prob, kalshi_yes=yes, kalshi_no=no))
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/manual_verify', methods=['POST'])
@login_required
def api_manual_verify():
    try:
        body = request.get_json(silent=True) or {}
        pred_id = body.get('pred_id')
        actual_direction = body.get('actual_direction')
        if not pred_id or actual_direction not in ['UP', 'DOWN']:
            return jsonify({'error': 'Invalid input'})
        success, was_correct = predictor.manual_verify(pred_id, actual_direction)
        return jsonify({'success': success, 'correct': was_correct if success else None})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/hourly', methods=['POST'])
@login_required
def api_hourly():
    try:
        return jsonify(tiered.predict())
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/kalshi_line')
@login_required
def api_kalshi_line():
    return jsonify({
        'line_15min': get_kalshi_15min(),
        'line_hourly': get_kalshi_hourly()
    })


@app.route('/api/latest')
@login_required
def api_latest():
    return jsonify({
        'price': predictor.current_price,
        'min15': (predictor.get_history(1) or [None])[0],
        'hourly': (tiered.get_history(1) or [None])[0]
    })


@app.route('/api/accuracy')
@login_required
def api_accuracy():
    return jsonify({
        'min15': predictor.get_accuracy(),
        'hourly': tiered.get_accuracy()
    })


@app.route('/api/history')
@login_required
def api_history():
    return jsonify({
        'min15': predictor.get_history(30),
        'hourly': tiered.get_history(10)
    })


@app.route('/api/pending')
@login_required
def api_pending():
    return jsonify({
        '15min': predictor.get_pending(),
        'hourly': tiered.get_pending()
    })


@app.route('/api/whale')
@login_required
def api_whale():
    try:
        return jsonify({
            'data': whale.track(),
            'signal': whale.get_signal()
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/retrain', methods=['POST'])
@login_required
def api_retrain():
    def run():
        predictor.train()
        tiered.train()
    threading.Thread(target=run, daemon=True).start()
    return jsonify({'success': True, 'message': 'Training started'})


@app.route('/api/train_status')
@login_required
def train_status():
    return jsonify({
        'min15_trained': predictor.is_trained,
        'hourly_trained': tiered.is_trained
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    if not predictor.is_trained:
        predictor.train()
    if not tiered.is_trained:
        tiered.train()
    app.run(host='0.0.0.0', port=5000, debug=False)