import os

# ── Auth ──────────────────────────────────────────────────────
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "REPLACE_ME")
SECRET_KEY          = os.environ.get("SECRET_KEY", "change-me-in-production")

# ── Timezone ──────────────────────────────────────────────────
TIMEZONE = "America/Chicago"

# ── Kalshi ────────────────────────────────────────────────────
KALSHI_API_KEY     = os.environ.get("KALSHI_API_KEY", "")
KALSHI_PRIVATE_KEY = os.environ.get("KALSHI_PRIVATE_KEY", "")

# ── Kalshi edge threshold ─────────────────────────────────────
MIN_EDGE_CENTS = 10

# ── Self-learning ─────────────────────────────────────────────
RETRAIN_AFTER_N_VERIFIED = 50
AUTO_PRUNE_DAYS          = 90
PRUNE_MIN_ACCURACY       = 70

# ── Hourly tier offsets ───────────────────────────────────────
ROUND_TO          = 100
SAFE_OFFSET       = 800
MODEST_OFFSET     = 500
AGGRESSIVE_OFFSET = 100

# ── Persistent storage ────────────────────────────────────────
# Points to Fly.io mounted volume so everything
# survives app restarts and redeploys forever
DATA_DIR   = os.environ.get("DATA_DIR", "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
