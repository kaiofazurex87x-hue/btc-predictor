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

# ── Fix working directory for Render ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

for d in ['data', 'models/predictor', 'models/tiered', 'templates']:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
app.config['SECRET_KEY'] = config.SECRET_KEY

predictor = BTCPredictor()
tiered    = TieredHourlyPredictor()
whale     = WhaleTracker()

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
    if h == "REPLACE_ME": return False
    try:
        return bcrypt.checkpw(pw.encode(), h.encode())
    except Exception:
        return False

# ── Background loops ───────────────────────────────────────────
def verify_loop():
    while True:
        time.sleep(120)
        try:
            n = predictor.verify_pending()
            if n: print(f"Verified {n} predictions")
        except Exception as e:
            print(f"Verify error: {e}")

def prune_loop():
    while True:
        time.sleep(6 * 3600)
        try:
            predictor.prune_old_data()
        except Exception as e:
            print(f"Prune error: {e}")

threading.Thread(target=verify_loop, daemon=True).start()
threading.Thread(target=prune_loop,  daemon=True).start()

# ── Auth ────────────────────────────────────────────────────────
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        if verify_pw(request.form.get('password','')):
            login_user(User('1'))
            return redirect(url_for('dashboard'))
        flash('Invalid password','error')
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

# ── API: predict ────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    try:
        body     = request.get_json(silent=True) or {}
        override = body.get('override_price')
        return jsonify(predictor.predict(override_price=override))
    except Exception as e:
        return jsonify({'error': str(e)})

# ── API: Kalshi check (fully optional) ─────────────────────────
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
        return jsonify(predictor.kalshi_check(pred_id, up_prob, kalshi_yes=yes, kalshi_no=no))
    except Exception as e:
        return jsonify({'error': str(e)})

# ── API: hourly tiered ──────────────────────────────────────────
@app.route('/api/hourly', methods=['POST'])
@login_required
def api_hourly():
    try:
        body     = request.get_json(silent=True) or {}
        override = body.get('override_price')
        return jsonify(tiered.predict(override_price=override))
    except Exception as e:
        return jsonify({'error': str(e)})

# ── API: accuracy ───────────────────────────────────────────────
@app.route('/api/accuracy')
@login_required
def api_accuracy():
    return jsonify({
        'min15':  predictor.get_accuracy(),
        'hourly': tiered.get_accuracy()
    })

# ── API: history ────────────────────────────────────────────────
@app.route('/api/history')
@login_required
def api_history():
    return jsonify({
        'min15':  predictor.get_history(30),
        'hourly': tiered.get_history(10)
    })

# ── API: whale ──────────────────────────────────────────────────
@app.route('/api/whale')
@login_required
def api_whale():
    try:
        return jsonify({'data': whale.track(), 'signal': whale.get_signal()})
    except Exception as e:
        return jsonify({'error': str(e)})

# ── API: manual retrain ─────────────────────────────────────────
@app.route('/api/retrain', methods=['POST'])
@login_required
def api_retrain():
    try:
        predictor.train()
        tiered.train()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

# ── Health check (keeps UptimeRobot happy) ──────────────────────
@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("Starting BTC Predictor...")
    if not predictor.is_trained:
        print("Training 15-min model (first run)...")
        predictor.train()
    if not tiered.is_trained:
        print("Training hourly model (first run)...")
        tiered.train()
    print("Ready → http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
