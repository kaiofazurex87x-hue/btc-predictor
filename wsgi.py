import sys
import os

path = '/home/btcpredictor87/btc-predictor'
if path not in sys.path:
    sys.path.insert(0, path)
os.chdir(path)

for d in ['data', 'models/predictor', 'models/tiered']:
    os.makedirs(os.path.join(path, d), exist_ok=True)

from app import app as application
