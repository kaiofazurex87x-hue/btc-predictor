import os

# ── Auth ──────────────────────────────────────────────────────
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "REPLACE_ME")
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")

# ── Timezone ──────────────────────────────────────────────────
TIMEZONE = "America/Chicago"  # US Central

# ── Kalshi edge threshold ─────────────────────────────────────
# App says BUY only when our probability beats Kalshi price by this many cents
MIN_EDGE_CENTS = 10

# ── Self-learning ─────────────────────────────────────────────
RETRAIN_AFTER_N_VERIFIED = 50   # auto-retrain every 50 verified predictions
AUTO_PRUNE_DAYS = 90            # delete records older than 90 days
PRUNE_MIN_ACCURACY = 70         # only prune when model accuracy is above this

# ── Hourly tiers ──────────────────────────────────────────────
ROUND_TO = 100
