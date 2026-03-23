"""
Configuration file - Change these values
"""

import os
import secrets

# Generate a secure secret key (change this!)
SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Admin password hash - generate with:
# python -c "import bcrypt; print(bcrypt.hashpw(b'your_password', bcrypt.gensalt()).decode())"
ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH', 'REPLACE_ME')

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Timezone
TIMEZONE = 'US/Eastern'