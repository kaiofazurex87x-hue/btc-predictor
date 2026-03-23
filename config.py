"""
Configuration file
"""

import os
import secrets

# Generate a secure secret key
SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Admin password - set via fly secrets
ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH', '')

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Timezone
TIMEZONE = 'US/Eastern'

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)