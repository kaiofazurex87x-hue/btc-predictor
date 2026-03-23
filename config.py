"""
Configuration file - Reads from environment variables (fly.io secrets)
"""

import os
import secrets

# Read from environment variables (set via fly secrets)
SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Admin password hash - set via fly secrets
ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH', '')

# Kalshi API Keys - set via fly secrets
KALSHI_KEY_ID = os.environ.get('KALSHI_KEY_ID', '')
KALSHI_PRIVATE_KEY = os.environ.get('KALSHI_PRIVATE_KEY', '')

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Timezone
TIMEZONE = 'US/Eastern'

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"[Config] Loaded - Kalshi Key ID: {KALSHI_KEY_ID[:8] if KALSHI_KEY_ID else 'Not set'}...")