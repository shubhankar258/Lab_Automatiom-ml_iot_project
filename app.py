# ============================================================
#  OLAS — Optimised Lab Automation System
#  Flask Scheduler + Live State Fetch | Deploy on Render.com
#  Year: 2026
# ============================================================

from flask import Flask, jsonify, render_template_string
import pickle
import pandas as pd
import numpy as np
import requests
import datetime
import schedule
import threading
import time
import os
import logging
import pytz

# Set timezone to IST
IST = pytz.timezone('Asia/Kolkata')

# ── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("OLAS")

app = Flask(__name__)

# ── Config — set these as Environment Variables on Render ────
NODE_ID  = os.environ.get("RAINMAKER_NODE_ID",  "aHjSGbCmWDjvmETWDMrupL")
RM_EMAIL = os.environ.get("RAINMAKER_EMAIL",    "sdhumal197@gmail.com")
RM_PASS  = os.environ.get("RAINMAKER_PASSWORD", "Pass@123")

SWITCHES  = ["Switch1", "Switch2", "Switch3", "Switch4"]
FEATURES  = [
    "hour", "minute", "day_of_week", "is_weekend", "time_block",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "dow_sin",  "dow_cos"
]
API_URL   = "https://api.rainmaker.espressif.com/v1/user/nodes/params"
LOGIN_URL = "https://api.rainmaker.espressif.com/v1/login2"

# ── Token cache ───────────────────────────────────────────────
_token_cache = {
    "access_token" : None,
    "refresh_token": None,
    "fetched_at"   : 0
}

# Global variable to store actual states fetched from RainMaker
actual_states = {sw: False for sw in SWITCHES}

def login_and_get_tokens():
    """
    Full login using email + password via /v1/login2.
    Stores both access_token and refresh_token in cache.
    """
    resp = requests.post(
        LOGIN_URL,
        json={"user_name": RM_EMAIL, "password": RM_PASS},
        timeout=10
    )
    if resp.status_code == 200:
        data = resp.json()
        _token_cache["access_token"]  = data["accesstoken"]
        _token_cache["refresh_token"] = data["refreshtoken"]
        _token_cache["fetched_at"]    = time.time()
        log.info("RainMaker login successful — tokens obtained")
    else:
        raise RuntimeError(
            f"RainMaker login failed: {resp.status_code}  {resp.text[:200]}"
        )

def get_access_token() -> str:
    """
    Returns a valid access token.
    """
    now = time.time()
    age = now - _token_cache["fetched_at"]

    if _token_cache["access_token"] is None or age > 3000:
        if _token_cache["refresh_token"]:
            try:
                resp = requests.post(
                    "https://cognito-idp.us-east-1.amazonaws.com/",
                    headers={
                        "Content-Type": "application/x-amz-json-1.1",
                        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth"
                    },
                    json={
                        "AuthFlow"      : "REFRESH_TOKEN_AUTH",
                        "ClientId"      : "1p3enpe49h9v0lqd7i4s5bub",
                        "AuthParameters": {
                            "REFRESH_TOKEN": _token_cache["refresh_token"]
                        }
                    },
                    timeout=10
                )
                new_token = resp.json()["AuthenticationResult"]["AccessToken"]
                _token_cache["access_token"] = new_token
                _token_cache["fetched_at"]   = now
                log.info("Access token refreshed silently via refresh_token")
            except Exception as e:
                log.warning(f"Silent refresh failed ({e}) — doing full re-login")
                login_and_get_tokens()
        else:
            login_and_get_tokens()

    return _token_cache["access_token"]

# ── Fetch actual states from RainMaker ───────────────────────
def fetch_actual_states():
    """Fetch current switch states from RainMaker cloud"""
    global actual_states
    try:
        headers = {"Authorization": f"Bearer {get_access_token()}"}
        resp = requests.get(
            f"{API_URL}?node_id={NODE_ID}",
            headers=headers,
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            for sw in SWITCHES:
                if sw in data:
                    state = data[sw].get("Power", data[sw].get("output", False))
                    actual_states[sw] = bool(state)
            log.info(f"Actual states fetched: {actual_states}")
            return True
        else:
            log.warning(f"Failed to fetch states: {resp.status_code}")
            return False
    except Exception as e:
        log.error(f"Error fetching states: {e}")
        return False

# ── Login on startup ──────────────────────────────────────────
if RM_EMAIL != "your@email.com":
    try:
        login_and_get_tokens()
        fetch_actual_states()
    except Exception as e:
        log.error(f"Startup login failed: {e}")

# ── Load model ───────────────────────────────────────────────
log.info("Loading lab_model.pkl ...")
with open("lab_model.pkl", "rb") as f:
    models = pickle.load(f)
log.info(f"Model loaded — classifiers: {list(models.keys())}")

# ── In-memory log of last 50 predictions ─────────────────────
prediction_log = []

# ── Feature builder ──────────────────────────────────────────
def build_features(dt: datetime.datetime) -> pd.DataFrame:
    row = {
        "hour"       : dt.hour,
        "minute"     : dt.minute,
        "day_of_week": dt.weekday(),
        "is_weekend" : 1 if dt.weekday() == 6 else 0,
        "time_block" : dt.hour // 6,
        "minute_sin" : np.sin(2 * np.pi * dt.minute / 60),
        "minute_cos" : np.cos(2 * np.pi * dt.minute / 60),
        "hour_sin"   : np.sin(2 * np.pi * dt.hour   / 24),
        "hour_cos"   : np.cos(2 * np.pi * dt.hour   / 24),
        "dow_sin"    : np.sin(2 * np.pi * dt.weekday() / 7),
        "dow_cos"    : np.cos(2 * np.pi * dt.weekday() / 7),
    }
    return pd.DataFrame([row])[FEATURES]

# ── Session detector ─────────────────────────────────────────
def current_session(dt: datetime.datetime) -> str:
    t = dt.time()
    if datetime.time(9, 15) <= t < datetime.time(11, 15):
        return "Session 1  (9:15 – 11:15)"
    if datetime.time(11, 30) <= t < datetime.time(13, 30):
        return "Session 2  (11:30 – 13:30)"
    if datetime.time(14, 15) <= t < datetime.time(16, 15):
        return "Session 3  (14:15 – 16:15)"
    if dt.weekday() == 6:
        return "Sunday — no college"
    return "Outside session hours"

# ── Core: predict and compare (NO automation commands) ───────
def predict_and_compare(source: str = "scheduler"):
    """Run ML prediction and compare with actual states (no commands sent)"""
    now = datetime.datetime.now(IST)
    feats = build_features(now)
    session = current_session(now)

    # Fetch latest actual states
    fetch_success = fetch_actual_states()

    # Run all 4 classifiers
    predictions = {}
    for sw in SWITCHES:
        pred = int(models[sw].predict(feats)[0])
        prob = models[sw].predict_proba(feats)[0][pred]
        predictions[sw] = {
            "state": bool(pred),
            "confidence": round(float(prob), 3)
        }

    # Compare actual vs predicted
    comparison = {}
    for sw in SWITCHES:
        actual = actual_states.get(sw, False)
        predicted = predictions[sw]["state"]
        comparison[sw] = {
            "actual": actual,
            "predicted": predicted,
            "match": actual == predicted,
            "confidence": predictions[sw]["confidence"]
        }

    api_status = "states_fetched" if fetch_success else "fetch_failed"

    # Store entry in memory log
    entry = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "session": session,
        "source": source,
        "predictions": predictions,
        "actual_states": actual_states.copy(),
        "comparison": comparison,
        "api_status": api_status,
    }
    prediction_log.insert(0, entry)
    if len(prediction_log) > 50:
        prediction_log.pop()

    # Log summary
    matches = sum(1 for sw in SWITCHES if comparison[sw]["match"])
    log.info(f"[{source}] Match: {matches}/4 | Fetch: {api_status}")
    return entry

# ── Scheduler thread (only fetches and compares, no commands) ─
schedule.every(30).minutes.do(lambda: predict_and_compare("scheduler"))

def run_scheduler():
    log.info("Scheduler started — fetching states & comparing every 30 minutes")
    predict_and_compare("startup")
    while True:
        schedule.run_pending()
        time.sleep(30)

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# ── Dashboard HTML (Updated with Actual vs Predicted) ────────
DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OLAS · Smart Lab</title>
  <meta http-equiv="refresh" content="60">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }

    :root {
      --bg-page: #f5f7fa;
      --card-bg: #ffffff;
      --text-primary: #1a1f2e;
      --text-secondary: #5e6a7e;
      --text-muted: #8a94a6;
      --border-light: #e9ecf0;
      --accent: #0066ff;
      --accent-soft: #e0ebff;
      --success: #00875a;
      --success-soft: #e3f7ef;
      --warning: #b45309;
      --warning-soft: #fdf0d5;
      --danger: #d92d20;
      --shadow-md: 0 8px 20px rgba(0,0,0,0.06);
    }

    [data-theme="dark"] {
      --bg-page: #0b0e14;
      --card-bg: #141a24;
      --text-primary: #eef2f6;
      --text-secondary: #9aabbf;
      --text-muted: #7d8ea0;
      --border-light: #242c38;
      --accent: #3399ff;
      --accent-soft: #17263a;
      --success: #00cc88;
      --success-soft: #0f2a22;
      --warning: #f0b429;
      --warning-soft: #2e2413;
      --danger: #ff6b6b;
    }

    body {
      background: var(--bg-page);
      color: var(--text-primary);
      padding: 20px 24px 48px;
      line-height: 1.5;
    }

    .dashboard { max-width: 1120px; margin: 0 auto; }

    /* ── Header ── */
    .top-nav {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 16px;
    }

    .brand { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }

    .project-name {
      font-size: 1.9rem;
      font-weight: 700;
      background: linear-gradient(135deg, #0066ff, #00c2ff);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }

    .project-tagline { color: var(--text-muted); font-size: 0.9rem; }

    .nav-controls { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }

    .theme-toggle {
      display: flex;
      background: var(--card-bg);
      border: 1px solid var(--border-light);
      border-radius: 40px;
      padding: 4px;
    }

    .theme-btn {
      background: transparent;
      border: none;
      padding: 6px 16px;
      border-radius: 40px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text-secondary);
    }

    .theme-btn.active { background: var(--accent); color: #fff; }

    .status-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      background: var(--accent-soft);
      padding: 8px 18px;
      border-radius: 40px;
      font-size: 0.85rem;
      color: var(--text-secondary);
    }

    .pulse-dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: var(--success);
      animation: pulse 1.8s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

    /* ── Summary banner ── */
    .summary {
      background: var(--card-bg);
      border: 1px solid var(--border-light);
      border-left: 6px solid var(--success);
      border-radius: 20px;
      padding: 22px 24px;
      box-shadow: var(--shadow-md);
      margin-bottom: 32px;
    }
    .summary.has-action { border-left-color: var(--warning); }

    .summary-headline { font-size: 1.35rem; font-weight: 700; margin-bottom: 6px; }
    .summary-sub { color: var(--text-secondary); font-size: 0.95rem; }

    .summary-facts {
      display: flex;
      flex-wrap: wrap;
      gap: 28px;
      margin-top: 18px;
      padding-top: 18px;
      border-top: 1px solid var(--border-light);
    }
    .fact-label {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      font-weight: 600;
    }
    .fact-value { font-size: 1rem; font-weight: 600; margin-top: 2px; }

    /* ── Sections ── */
    .section { margin-bottom: 32px; }
    .section-title { font-size: 1.05rem; font-weight: 600; margin: 0 0 4px; }
    .section-help { color: var(--text-muted); font-size: 0.88rem; margin-bottom: 16px; }

    /* ── Device cards ── */
    .comparison-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 18px;
    }

    .comparison-card {
      background: var(--card-bg);
      border-radius: 20px;
      padding: 20px 18px;
      box-shadow: var(--shadow-md);
      border: 1px solid var(--border-light);
      border-top: 4px solid var(--success);
    }
    .comparison-card.mismatch { border-top-color: var(--warning); }

    .device-name { font-weight: 600; font-size: 1.05rem; }
    .device-sub { color: var(--text-muted); font-size: 0.78rem; margin-bottom: 16px; }

    .state-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 9px 0;
      border-bottom: 1px solid var(--border-light);
    }
    .state-row .label { color: var(--text-secondary); font-size: 0.85rem; }

    .pill {
      font-weight: 700;
      font-size: 0.78rem;
      letter-spacing: 0.04em;
      padding: 3px 12px;
      border-radius: 40px;
      border: 1px solid var(--border-light);
    }
    .pill.on  { background: var(--success-soft); color: var(--success); border-color: var(--success); }
    .pill.off { background: transparent; color: var(--text-muted); }

    .verdict {
      margin-top: 14px;
      padding: 10px 12px;
      border-radius: 14px;
      font-size: 0.86rem;
      font-weight: 600;
      background: var(--success-soft);
      color: var(--success);
    }
    .mismatch .verdict { background: var(--warning-soft); color: var(--warning); }

    .confidence { margin-top: 10px; font-size: 0.78rem; color: var(--text-muted); }

    /* ── Panels ── */
    .panel {
      background: var(--card-bg);
      border-radius: 20px;
      padding: 22px 24px;
      border: 1px solid var(--border-light);
      box-shadow: var(--shadow-md);
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 16px;
    }

    .btn {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 11px 22px;
      border-radius: 40px;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
    }
    .btn:disabled { opacity: 0.6; cursor: default; }

    /* ── Manual check ── */
    .manual-form {
      display: flex;
      gap: 14px;
      align-items: flex-end;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    .input-group { display: flex; flex-direction: column; gap: 6px; }
    .input-group label { font-size: 0.78rem; color: var(--text-muted); font-weight: 600; }
    .input-group input, .input-group select {
      background: var(--bg-page);
      border: 1px solid var(--border-light);
      border-radius: 14px;
      padding: 11px 14px;
      color: var(--text-primary);
      font-size: 0.95rem;
      width: 130px;
    }

    .prediction-preview { margin-top: 20px; }
    .preview-caption { color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 12px; }
    .preview-grid { display: flex; gap: 14px; flex-wrap: wrap; }
    .preview-item {
      background: var(--bg-page);
      border: 1px solid var(--border-light);
      padding: 14px 20px;
      border-radius: 16px;
      min-width: 120px;
    }
    .preview-item .name { font-size: 0.8rem; color: var(--text-muted); }
    .preview-item .state { font-size: 1.3rem; font-weight: 700; margin: 4px 0; }
    .preview-item .state.on { color: var(--success); }
    .preview-item .state.off { color: var(--text-muted); }
    .preview-item small { color: var(--text-muted); font-size: 0.75rem; }

    /* ── History table ── */
    table.log { width: 100%; border-collapse: collapse; margin-top: 14px; }
    table.log th {
      text-align: left;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      font-weight: 600;
      padding: 0 12px 10px 0;
      border-bottom: 1px solid var(--border-light);
    }
    table.log td {
      padding: 12px 12px 12px 0;
      border-bottom: 1px solid var(--border-light);
      font-size: 0.9rem;
      vertical-align: middle;
    }
    table.log td.time { color: var(--text-secondary); white-space: nowrap; }

    .tag {
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 40px;
      white-space: nowrap;
    }
    .tag.ok   { background: var(--success-soft); color: var(--success); }
    .tag.warn { background: var(--warning-soft); color: var(--warning); }

    .empty { padding: 32px; text-align: center; color: var(--text-muted); }

    @media (max-width: 900px) {
      .comparison-grid { grid-template-columns: repeat(2, 1fr); }
      table.log .hide-sm { display: none; }
    }
    @media (max-width: 520px) {
      .comparison-grid { grid-template-columns: 1fr; }
      body { padding: 16px 14px 40px; }
    }
  </style>
</head>
<body>
<div class="dashboard">

  <!-- Header -->
  <div class="top-nav">
    <div class="brand">
      <span class="project-name">OLAS</span>
      <span class="project-tagline">Optimised Lab Automation System</span>
    </div>
    <div class="nav-controls">
      <div class="theme-toggle">
        <button class="theme-btn active" data-theme="light" onclick="setTheme('light')">☀️ Light</button>
        <button class="theme-btn" data-theme="dark" onclick="setTheme('dark')">🌙 Dark</button>
      </div>
      <div class="status-badge">
        <span class="pulse-dot"></span>
        <span>Live from RainMaker</span>
      </div>
    </div>
  </div>

  <!-- Plain-English summary -->
  <div class="summary" id="summaryCard">
    <div class="summary-headline" id="summaryHeadline">Waiting for the first reading…</div>
    <div class="summary-sub" id="summarySub">The scheduler checks the lab every 30 minutes.</div>
    <div class="summary-facts">
      <div>
        <div class="fact-label">Timetable session</div>
        <div class="fact-value" id="factSession">—</div>
      </div>
      <div>
        <div class="fact-label">Lights on now</div>
        <div class="fact-value" id="factOn">—</div>
      </div>
      <div>
        <div class="fact-label">Last checked</div>
        <div class="fact-value" id="factTime">—</div>
      </div>
      <div>
        <div class="fact-label">Cloud connection</div>
        <div class="fact-value" id="factApi">—</div>
      </div>
    </div>
  </div>

  <!-- Device cards -->
  <div class="section">
    <div class="section-title">Each light: what it is doing vs what it should do</div>
    <div class="section-help">
      <strong>Actual now</strong> is the live state read from the relay. <strong>Should be</strong> is what the
      timetable model expects. When the two differ, the card names the energy-saving action.
      OLAS only advises, it never switches anything itself.
    </div>
    <div class="comparison-grid" id="comparisonGrid"></div>
  </div>

  <!-- Refresh -->
  <div class="section panel panel-head">
    <div>
      <div class="section-title">Run a check now</div>
      <div class="section-help" style="margin:0;">Reads the live switch states and re-runs the model.</div>
    </div>
    <button class="btn" id="refreshBtn">⟳ Fetch &amp; predict</button>
  </div>

  <!-- Manual model check -->
  <div class="section panel">
    <div class="section-title">Try any time of the week</div>
    <div class="section-help">Ask the model what it would expect at a chosen hour, minute and weekday.</div>
    <div class="manual-form">
      <div class="input-group">
        <label for="predHour">Hour (0–23)</label>
        <input type="number" id="predHour" min="0" max="23" value="9">
      </div>
      <div class="input-group">
        <label for="predMinute">Minute</label>
        <input type="number" id="predMinute" min="0" max="59" value="30">
      </div>
      <div class="input-group">
        <label for="predDay">Weekday</label>
        <select id="predDay">
          <option value="0">Monday</option><option value="1">Tuesday</option><option value="2">Wednesday</option>
          <option value="3">Thursday</option><option value="4">Friday</option><option value="5">Saturday</option>
          <option value="6">Sunday</option>
        </select>
      </div>
      <button class="btn" id="predictBtn">Predict</button>
    </div>
    <div id="manualPredictionOutput" class="prediction-preview"></div>
  </div>

  <!-- History -->
  <div class="section panel">
    <div class="section-title">Recent checks</div>
    <div class="section-help">The last few scheduler runs, newest first.</div>
    <div id="logFeedContainer"></div>
  </div>
</div>

<script>
  function setTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.theme-btn[data-theme="${t}"]`).classList.add('active');
    localStorage.setItem('olas-theme', t);
  }
  setTheme(localStorage.getItem('olas-theme') || 'light');

  const switches = ['Switch1','Switch2','Switch3','Switch4'];
  const label = sw => 'Light ' + sw.replace('Switch', '');
  const initialLogs = {{ logs | tojson if logs else [] }};

  function comparisonOf(entry) {
    return switches.map(sw => entry.comparison?.[sw] || {
      actual: false,
      predicted: entry.predictions[sw].state,
      match: false,
      confidence: entry.predictions[sw].confidence
    });
  }

  function renderSummary(logs) {
    if (!logs.length) return;
    const entry = logs[0];
    const comps = comparisonOf(entry);
    const mismatches = comps.filter(c => !c.match).length;
    const onNow = comps.filter(c => c.actual).length;
    const wasted = comps.filter(c => c.actual && !c.predicted).length;

    document.getElementById('summaryCard').classList.toggle('has-action', mismatches > 0);

    let headline, sub;
    if (mismatches === 0) {
      headline = 'All 4 lights match the timetable';
      sub = 'Nothing to change right now.';
    } else {
      headline = `${mismatches} of 4 lights differ from the timetable`;
      sub = wasted > 0
        ? `${wasted} light${wasted > 1 ? 's are' : ' is'} on when the timetable says it need not be. Turning them off saves energy.`
        : 'The model expects some lights to be on that are currently off.';
    }
    document.getElementById('summaryHeadline').textContent = headline;
    document.getElementById('summarySub').textContent = sub;
    document.getElementById('factSession').textContent = entry.session;
    document.getElementById('factOn').textContent = `${onNow} of 4`;
    document.getElementById('factTime').textContent = entry.timestamp;
    document.getElementById('factApi').textContent =
      entry.api_status === 'states_fetched' ? 'Connected' : 'Could not read states';
  }

  function renderComparison(logs) {
    const grid = document.getElementById('comparisonGrid');
    if (!logs.length) {
      grid.innerHTML = '<div class="empty" style="grid-column:1/-1;">No readings yet. Run a check to get started.</div>';
      return;
    }
    const comps = comparisonOf(logs[0]);
    grid.innerHTML = switches.map((sw, i) => {
      const c = comps[i];
      const conf = Math.round(c.confidence * 100);
      const verdict = c.match
        ? 'Correct, no action needed'
        : (c.actual ? 'Suggest: turn this light OFF' : 'Suggest: turn this light ON');
      return `
        <div class="comparison-card ${c.match ? '' : 'mismatch'}">
          <div class="device-name">${label(sw)}</div>
          <div class="device-sub">${sw}</div>
          <div class="state-row">
            <span class="label">Actual now</span>
            <span class="pill ${c.actual ? 'on' : 'off'}">${c.actual ? 'ON' : 'OFF'}</span>
          </div>
          <div class="state-row">
            <span class="label">Should be</span>
            <span class="pill ${c.predicted ? 'on' : 'off'}">${c.predicted ? 'ON' : 'OFF'}</span>
          </div>
          <div class="verdict">${verdict}</div>
          <div class="confidence">Model confidence ${conf}%</div>
        </div>`;
    }).join('');
  }

  function renderLogs(logs) {
    const container = document.getElementById('logFeedContainer');
    if (!logs.length) {
      container.innerHTML = '<div class="empty">No checks recorded yet.</div>';
      return;
    }
    const rows = logs.slice(0, 8).map(entry => {
      const matches = comparisonOf(entry).filter(c => c.match).length;
      const ok = matches === 4;
      const connected = entry.api_status === 'states_fetched';
      return `<tr>
        <td class="time">${entry.timestamp}</td>
        <td class="hide-sm">${entry.session}</td>
        <td><span class="tag ${ok ? 'ok' : 'warn'}">${matches} of 4 match</span></td>
        <td class="hide-sm"><span class="tag ${connected ? 'ok' : 'warn'}">${connected ? 'Connected' : 'Read failed'}</span></td>
      </tr>`;
    }).join('');
    container.innerHTML = `<table class="log">
      <thead><tr>
        <th>Time</th><th class="hide-sm">Session</th><th>Result</th><th class="hide-sm">Cloud</th>
      </tr></thead>
      <tbody>${rows}</tbody></table>`;
  }

  document.getElementById('predictBtn').addEventListener('click', async () => {
    const h = document.getElementById('predHour').value;
    const m = document.getElementById('predMinute').value;
    const daySel = document.getElementById('predDay');
    const out = document.getElementById('manualPredictionOutput');
    out.innerHTML = '<p class="preview-caption">Predicting…</p>';
    try {
      const r = await fetch(`/predict_time/${h}/${m}/${daySel.value}`);
      const data = await r.json();
      const items = switches.map(sw => {
        const p = data.predictions[sw];
        return `<div class="preview-item">
          <div class="name">${label(sw)}</div>
          <div class="state ${p.state ? 'on' : 'off'}">${p.state ? 'ON' : 'OFF'}</div>
          <small>${Math.round(p.confidence * 100)}% confidence</small>
        </div>`;
      }).join('');
      out.innerHTML = `<div class="preview-caption">
          Expected states on <strong>${daySel.options[daySel.selectedIndex].text}</strong>
          at <strong>${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}</strong> · ${data.session}
        </div><div class="preview-grid">${items}</div>`;
    } catch (e) {
      out.innerHTML = '<p class="preview-caption" style="color:var(--danger);">Could not reach the model. Try again.</p>';
    }
  });

  document.getElementById('refreshBtn').addEventListener('click', async () => {
    const btn = document.getElementById('refreshBtn');
    btn.disabled = true;
    btn.textContent = 'Fetching…';
    try {
      await fetch('/trigger');
      setTimeout(() => location.reload(), 500);
    } catch {
      btn.textContent = 'Failed, try again';
      btn.disabled = false;
    }
  });

  renderSummary(initialLogs);
  renderComparison(initialLogs);
  renderLogs(initialLogs);
</script>
</body>
</html>
"""

# ── Flask routes ──────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD, logs=prediction_log)

@app.route("/status")
def status():
    last = prediction_log[0] if prediction_log else None
    return jsonify({
        "project": "OLAS 2026",
        "status": "running",
        "scheduler": "active — every 30 min (fetch & compare only)",
        "node_id": NODE_ID[:8] + "..." if len(NODE_ID) > 8 else NODE_ID,
        "actual_states": actual_states,
        "last_run": last["timestamp"] if last else None,
        "last_session": last["session"] if last else None,
    })

@app.route("/trigger", methods=["GET", "POST"])
def trigger():
    entry = predict_and_compare("manual_trigger")
    return jsonify({
        "message": "States fetched & prediction completed",
        "timestamp": entry["timestamp"],
        "session": entry["session"],
        "actual_states": entry["actual_states"],
        "predictions": entry["predictions"],
        "comparison": entry["comparison"],
    })

@app.route("/predict_time/<int:hour>/<int:minute>/<int:dow>")
def predict_time(hour, minute, dow):
    now = datetime.datetime.now(IST)
    # shift to the requested weekday so day_of_week actually reaches the model
    dt = (now + datetime.timedelta(days=(dow - now.weekday()) % 7)).replace(
        hour=hour, minute=minute
    )
    feats = build_features(dt)
    preds = {}
    for sw in SWITCHES:
        state = int(models[sw].predict(feats)[0])
        prob = models[sw].predict_proba(feats)[0][state]
        preds[sw] = {"state": bool(state), "confidence": round(float(prob), 3)}
    return jsonify({
        "query": f"{hour:02d}:{minute:02d}  day_of_week={dow}",
        "session": current_session(dt),
        "predictions": preds
    })

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)