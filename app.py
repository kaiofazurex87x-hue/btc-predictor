from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import bcrypt
import threading
import time
import os
import config

from predictor import BTCPredictor
from tiered_predictor import TieredHourlyPredictor
from whale_tracker import WhaleTracker
from kalshi_api import KalshiAPI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

for d in ['data', 'models/predictor', 'models/tiered', 'templates']:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'))
app.config['SECRET_KEY'] = config.SECRET_KEY

predictor = BTCPredictor()
tiered    = TieredHourlyPredictor()
whale     = WhaleTracker()
kalshi    = KalshiAPI()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):
    def __init__(self, id): self.id = id


@login_manager.user_loader
def load_user(uid):
    return User(uid) if uid == '1' else None


def verify_pw(pw):
    h = config.ADMIN_PASSWORD_HASH
    if h == "REPLACE_ME":
        return False
    try:
        return bcrypt.checkpw(pw.encode(), h.encode())
    except Exception:
        return False


def auto_predict_loop():
    import pytz
    from datetime import datetime, timedelta
    TZ = pytz.timezone(config.TIMEZONE)
    while not predictor.is_trained:
        time.sleep(30)
    while True:
        now      = datetime.now(pytz.utc).astimezone(TZ)
        next_min = ((now.minute // 15) + 1) * 15
        if next_min >= 60:
            next_run = now.replace(
                minute=0, second=2, microsecond=0) + timedelta(hours=1)
        else:
            next_run = now.replace(
                minute=next_min, second=2, microsecond=0)
        wait = (next_run - now).total_seconds()
        time.sleep(max(wait, 1))
        try:
            line   = kalshi.get_btc_15min_line()
            result = predictor.predict(kalshi_line=line, force=True)
            print(f"[Auto 15-min] {result['direction']} "
                  f"{result['confidence']}% | "
                  f"ref: ${result['reference_price']}")
        except Exception as e:
            print(f"[Auto 15-min] Error: {e}")


def auto_hourly_loop():
    import pytz
    from datetime import datetime, timedelta
    TZ = pytz.timezone(config.TIMEZONE)
    while not tiered.is_trained:
        time.sleep(30)
    while True:
        now      = datetime.now(pytz.utc).astimezone(TZ)
        next_run = now.replace(
            minute=0, second=5, microsecond=0) + timedelta(hours=1)
        wait = (next_run - now).total_seconds()
        time.sleep(max(wait, 1))
        try:
            line   = kalshi.get_btc_hourly_line()
            result = tiered.predict(override_price=line)
            print(f"[Auto Hourly] "
                  f"Predicted: ${result['predicted_price']} | "
                  f"Safe: {result['tiers']['safe']['formatted']} | "
                  f"Modest: {result['tiers']['modest']['formatted']} | "
                  f"Aggressive: {result['tiers']['aggressive']['formatted']}")
        except Exception as e:
            print(f"[Auto Hourly] Error: {e}")


def verify_loop():
    while True:
        time.sleep(120)
        try:
            n15 = predictor.verify_pending()
            nhr = tiered.verify_pending()
            if n15 or nhr:
                print(f"[Verify] 15-min: {n15}  Hourly: {nhr}")
        except Exception as e:
            print(f"[Verify] Error: {e}")


def prune_loop():
    while True:
        time.sleep(6 * 3600)
        try:
            predictor.prune_old_data()
        except Exception as e:
            print(f"[Prune] Error: {e}")


threading.Thread(target=auto_predict_loop, daemon=True).start()
threading.Thread(target=auto_hourly_loop,  daemon=True).start()
threading.Thread(target=verify_loop,       daemon=True).start()
threading.Thread(target=prune_loop,        daemon=True).start()


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if verify_pw(request.form.get('password', '')):
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
        return jsonify(predictor.predict(
            kalshi_line=body.get('kalshi_line')))
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/kalshi_check', methods=['POST'])
@login_required
def api_kalshi_check():
    try:
        body    = request.get_json(silent=True) or {}
        pred_id = body.get('pred_id')
        up_prob = float(body.get('up_prob'))
        yes     = body.get('kalshi_yes')
        no      = body.get('kalshi_no')
        if yes is not None: yes = float(yes)
        if no  is not None: no  = float(no)
        return jsonify(predictor.kalshi_check(
            pred_id, up_prob, kalshi_yes=yes, kalshi_no=no))
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/hourly', methods=['POST'])
@login_required
def api_hourly():
    try:
        body = request.get_json(silent=True) or {}
        return jsonify(tiered.predict(
            override_price=body.get('kalshi_line')))
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/kalshi_line')
@login_required
def api_kalshi_line():
    try:
        return jsonify({
            'line_15min':  kalshi.get_btc_15min_line(),
            'line_hourly': kalshi.get_btc_hourly_line()
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/latest')
@login_required
def api_latest():
    try:
        return jsonify({
            'min15':  (predictor.get_history(1) or [None])[0],
            'hourly': (tiered.get_history(1)    or [None])[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/accuracy')
@login_required
def api_accuracy():
    return jsonify({
        'min15':  predictor.get_accuracy(),
        'hourly': tiered.get_accuracy()
    })


@app.route('/api/history')
@login_required
def api_history():
    return jsonify({
        'min15':  predictor.get_history(30),
        'hourly': tiered.get_history(10)
    })


@app.route('/api/whale')
@login_required
def api_whale():
    try:
        return jsonify({
            'data':   whale.track(),
            'signal': whale.get_signal()
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/retrain', methods=['POST'])
@login_required
def api_retrain():
    try:
        def run():
            predictor.train()
            tiered.train()
        threading.Thread(target=run, daemon=True).start()
        return jsonify({'success': True,
                        'message': 'Training started. Takes 3-5 minutes.'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/train_status')
@login_required
def train_status():
    return jsonify({
        'min15_trained':  predictor.is_trained,
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
