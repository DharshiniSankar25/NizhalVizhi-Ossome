import streamlit as st
st.set_page_config(
    page_title="Nizhal Vizhi",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="collapsed",
)
import cv2
import numpy as np
import threading
import subprocess
import time
import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from datetime import datetime
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass
YOLO_MODEL_PATH = "yolov8n.pt"   
YOLO_CONFIDENCE = 0.45         
SURVEILLANCE_CLASSES = {
    "cell phone", "remote", "clock", "laptop",
    "mouse", "keyboard", "book",
}
GLINT_THRESHOLD  = 240    
GLINT_MIN_RADIUS = 2
GLINT_MAX_RADIUS = 10    
GLINT_MIN_AREA   = int(np.pi * GLINT_MIN_RADIUS ** 2)
GLINT_MAX_AREA   = int(np.pi * GLINT_MAX_RADIUS ** 2)

#RF / WiFi
RF_SCAN_INTERVAL      = 5   
RF_SIGNAL_MIN         = 30
RF_SUBPROCESS_TIMEOUT = 10   
SUSPICIOUS_KEYWORDS = [
    "v380", "ipc", "cam", "hik", "tuya", "esp32", "espressif",
    "dahua", "reolink", "wyze", "arlo", "ring", "ezviz",
    "tapo", "kasa", "cctv", "nvr", "dvr", "ipcam", "spycam",
    "mini", "hidden", "spy", "nano", "wificam", "ipcamera",
    "vstarcam", "foscam", "tenvis", "sricam", "instar",
]
W_YOLO  = 0.45
W_GLINT = 0.30
W_RF    = 0.25
MAX_LOG_ENTRIES = 80
LOG_COOLDOWN    = 5.0  
DEBUG_RF = True
logging.basicConfig(
    level=logging.DEBUG if DEBUG_RF else logging.WARNING,
    format="[%(name)s] %(levelname)s: %(message)s",
)
rf_logger = logging.getLogger("RF")

@dataclass
class YOLOResult:
    detections:   list = field(default_factory=list)
    threat_count: int  = 0
    confidence:   float = 0.0
@dataclass
class GlintResult:
    glints:       list  = field(default_factory=list)
    threat_count: int   = 0
    confidence:   float = 0.0
@dataclass
class WiFiNetwork:
    ssid:   str
    bssid:  str = ""
    signal: int = 0
    auth:   str = "Unknown"
    source: str = "scan"   

    @property
    def is_hidden(self) -> bool:
        return self.ssid.strip() == ""
    @property
    def signal_bar(self) -> str:
        if self.signal >= 75: return "IIII"
        if self.signal >= 50: return "III."
        if self.signal >= 25: return "II.."
        return "I..."
@dataclass
class RFResult:
    networks:     list  = field(default_factory=list)
    threats:      list  = field(default_factory=list)
    error:        Optional[str] = None
    scan_time:    float = 0.0
    raw_output:   str   = ""      
    last_updated: float = 0.0
    scan_method:  str   = ""     
@dataclass
class ThreatState:
    yolo:  YOLOResult  = field(default_factory=YOLOResult)
    glint: GlintResult = field(default_factory=GlintResult)
    rf:    RFResult    = field(default_factory=RFResult)
    @property
    def privacy_score(self) -> int:
        """
        Privacy Health Score: 100 = safe, 0 = maximum threat.

        Formula:
          threat_level = (yolo_hits/3)*0.5 + (glint_hits/5)*0.3 + (rf_hits/3)*0.2
          score = 100 - threat_level * 100

        Capped so 3+ YOLO hits = full YOLO penalty, etc.
        """
        yolo_hit  = min(self.yolo.threat_count,  3) / 3
        glint_hit = min(self.glint.threat_count, 5) / 5
        rf_hit    = min(len(self.rf.threats),    3) / 3
        raw = yolo_hit * W_YOLO + glint_hit * W_GLINT + rf_hit * W_RF
        return max(0, int(100 - raw * 100))

    @property
    def threat_confidence(self) -> float:
        scores = []
        if self.yolo.threat_count:
            scores.append(self.yolo.confidence)
        if self.glint.threat_count:
            scores.append(self.glint.confidence)
        if self.rf.threats:
            scores.append(0.85)
        return round(max(scores) if scores else 0.0, 2)

    @property
    def status(self):
        score = self.privacy_score
        if score >= 85:
            return "GREEN  SAFE",       "safe"
        elif score >= 45:
            return "YELLOW  SUSPICIOUS", "suspicious"
        else:
            return "RED  HIGH RISK",    "danger"

def load_yolo():
    if not YOLO_AVAILABLE:
        return None
    try:
        model = YOLO(YOLO_MODEL_PATH)
        return model
    except Exception as exc:
        rf_logger.warning("YOLO load failed: %s", exc)
        return None


def run_yolo(frame: np.ndarray, model):
    
    result    = YOLOResult()
    out_frame = frame.copy()

    if model is None:
        return out_frame, result

    try:
        preds    = model(frame, verbose=False, conf=YOLO_CONFIDENCE)[0]
        names    = preds.names
        max_conf = 0.0

        for box in preds.boxes:
            cls_id = int(box.cls[0])
            label  = names[cls_id].lower()
            conf   = float(box.conf[0])

            if label not in SURVEILLANCE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            result.detections.append({"label": label, "conf": conf,
                                      "bbox": (x1, y1, x2, y2)})
            result.threat_count += 1
            max_conf = max(max_conf, conf)

            # Blue box
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (30, 80, 230), 2)
            tag = f"SURVEILLANCE FORM-FACTOR  {label}  {conf:.0%}"
            tw  = len(tag) * 8
            cv2.rectangle(out_frame, (x1, y1 - 22), (x1 + tw, y1), (30, 80, 230), -1)
            cv2.putText(out_frame, tag, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        result.confidence = round(max_conf, 2)

    except Exception as exc:
        rf_logger.warning("YOLO error: %s", exc)

    return out_frame, result

def run_glint_detection(frame: np.ndarray):

    result    = GlintResult()
    out_frame = frame.copy()

    try:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, GLINT_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Remove single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (GLINT_MIN_AREA <= area <= GLINT_MAX_AREA):
                continue

            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circularity = 4 * np.pi * area / (perim ** 2)
            if circularity < 0.55:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy, radius   = int(cx), int(cy), int(radius)
            if not (GLINT_MIN_RADIUS < radius < GLINT_MAX_RADIUS):
                continue

            result.glints.append((cx, cy, radius))
            result.threat_count += 1

            # Yellow ring + dot
            cv2.circle(out_frame, (cx, cy), radius + 5, (0, 210, 255), 2)
            cv2.circle(out_frame, (cx, cy), 2,           (0, 210, 255), -1)
            cv2.putText(out_frame, "LENS GLINT", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 210, 255), 1, cv2.LINE_AA)

        result.confidence = round(min(1.0, result.threat_count * 0.25), 2)

    except Exception as exc:
        rf_logger.warning("Glint error: %s", exc)

    return out_frame, result

_RE_SSID_ANCHOR = re.compile(r"^\s*SSID\s+\d+\s*[:\uff1a]\s*(.*)$",            re.I | re.M)
_RE_BSSID_LINE  = re.compile(r"^\s*BSSID\s+\d+\s*[:\uff1a]\s*([0-9a-fA-F:]+)", re.I | re.M)
_RE_SIGNAL_LINE = re.compile(r"^\s*Signal\s*[:\uff1a]\s*(\d+)\s*%",             re.I | re.M)
_RE_AUTH_LINE   = re.compile(r"^\s*Authentication\s*[:\uff1a]\s*(.+)$",          re.I | re.M)

# For netsh wlan show interfaces (different format -- no "SSID 1", just "SSID")
_RE_IF_SSID   = re.compile(r"^\s*SSID\s*[:\uff1a]\s*(.+)$",             re.I | re.M)
_RE_IF_BSSID  = re.compile(r"^\s*BSSID\s*[:\uff1a]\s*([0-9a-fA-F:]+)", re.I | re.M)
_RE_IF_SIGNAL = re.compile(r"^\s*Signal\s*[:\uff1a]\s*(\d+)\s*%",       re.I | re.M)
_RE_IF_AUTH   = re.compile(r"^\s*Authentication\s*[:\uff1a]\s*(.+)$",   re.I | re.M)

# For netsh wlan show profiles
_RE_PROFILE = re.compile(
    r"^\s*(?:All User Profile|User Profile)\s*[:\uff1a]\s*(.+)$", re.I | re.M
)


def _decode_bytes(raw_bytes: bytes) -> str:

    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def _run_cmd(args: list) -> tuple:

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            timeout=RF_SUBPROCESS_TIMEOUT,
        )
        text = _decode_bytes(proc.stdout)
        if proc.returncode != 0:
            err = _decode_bytes(proc.stderr).strip()
            return text, f"exit {proc.returncode}: {err}"
        return text, None
    except subprocess.TimeoutExpired:
        return "", f"timeout after {RF_SUBPROCESS_TIMEOUT}s"
    except FileNotFoundError:
        return "", "netsh not found -- Windows + WLAN adapter required"
    except Exception as exc:
        return "", str(exc)


# -- COMMAND A: parse `netsh wlan show networks mode=bssid` -------------------

def _parse_networks_scan(raw: str) -> list:
    networks = []
    anchors  = list(_RE_SSID_ANCHOR.finditer(raw))

    if not anchors:
        rf_logger.debug("[CMD-A] No SSID anchors found in scan output")
        return networks

    for i, match in enumerate(anchors):
        ssid_name = match.group(1).strip()
        start     = match.start()
        end       = anchors[i + 1].start() if i + 1 < len(anchors) else len(raw)
        block     = raw[start:end]

        # Take first BSSID in block
        bssid_m = _RE_BSSID_LINE.search(block)
        auth_m  = _RE_AUTH_LINE.search(block)

        # Take the MAXIMUM signal across all BSSIDs in this block
        signals = [int(m.group(1)) for m in _RE_SIGNAL_LINE.finditer(block)]
        signal  = max(signals) if signals else 0

        bssid = bssid_m.group(1).strip() if bssid_m else ""
        auth  = auth_m.group(1).strip()  if auth_m  else "Unknown"

        net = WiFiNetwork(ssid=ssid_name, bssid=bssid, signal=signal,
                          auth=auth, source="scan")
        rf_logger.debug("[CMD-A] SSID=%-28s signal=%3d%%  bssid=%s",
                        repr(ssid_name), signal, bssid)
        networks.append(net)

    rf_logger.debug("[CMD-A] Parsed %d networks from scan command", len(networks))
    return networks

def _parse_connected_interface(raw: str):

    ssid_m   = _RE_IF_SSID.search(raw)
    bssid_m  = _RE_IF_BSSID.search(raw)
    signal_m = _RE_IF_SIGNAL.search(raw)
    auth_m   = _RE_IF_AUTH.search(raw)

    if not ssid_m:
        rf_logger.debug("[CMD-B] No connected interface SSID found")
        return None

    ssid   = ssid_m.group(1).strip()
    bssid  = bssid_m.group(1).strip()  if bssid_m  else ""
    signal = int(signal_m.group(1))    if signal_m else 0
    auth   = auth_m.group(1).strip()   if auth_m   else "Unknown"

    rf_logger.debug("[CMD-B] Connected -> SSID=%-28s signal=%3d%%  bssid=%s",
                    repr(ssid), signal, bssid)
    return WiFiNetwork(ssid=ssid, bssid=bssid, signal=signal,
                       auth=auth, source="connected")
def _parse_profiles(raw: str) -> list:

    names    = [m.group(1).strip() for m in _RE_PROFILE.finditer(raw)]
    networks = [WiFiNetwork(ssid=name, signal=0, source="profile")
                for name in names if name]
    rf_logger.debug("[CMD-C] Profiles found: %d", len(networks))
    return networks


def _merge_networks(scan_nets: list, connected_net, profile_nets: list) -> list:

    merged = {}   

    for net in scan_nets:
        key = net.ssid.lower().strip()
        if key not in merged or net.signal > merged[key].signal:
            merged[key] = net

    if connected_net:
        key = connected_net.ssid.lower().strip()
        if key not in merged or connected_net.signal > merged[key].signal:
            merged[key] = connected_net

    for net in profile_nets:
        key = net.ssid.lower().strip()
        if key not in merged:
            merged[key] = net

    result = sorted(merged.values(), key=lambda n: -n.signal)
    rf_logger.debug("[MERGE] Total unique networks after merge: %d", len(result))
    return result


def _classify_threats(networks: list) -> list:
   
    threats = []

    for net in networks:
        if net.source != "profile" and net.signal < RF_SIGNAL_MIN:
            continue

        if net.is_hidden:
            rf_logger.debug("[THREAT] Hidden SSID  bssid=%s signal=%d%%",
                            net.bssid, net.signal)
            threats.append(net)
            continue

        ssid_lower = net.ssid.lower()
        for kw in SUSPICIOUS_KEYWORDS:
            if kw in ssid_lower:
                rf_logger.debug("[THREAT] Keyword=%r  SSID=%s  signal=%d%%",
                                kw, net.ssid, net.signal)
                threats.append(net)
                break

    rf_logger.debug("[CLASSIFY] %d threats from %d networks",
                    len(threats), len(networks))
    return threats

def scan_wifi() -> RFResult:

    t0     = time.monotonic()
    result = RFResult(last_updated=time.time())
    errors = []
    methods_ok = []
    raw_a, err_a = _run_cmd(["netsh", "wlan", "show", "networks", "mode=bssid"])
    if DEBUG_RF and raw_a:
        rf_logger.debug("[CMD-A RAW]\n%s", raw_a[:1200])
    if err_a:
        errors.append(f"CMD-A: {err_a}")
    scan_nets = _parse_networks_scan(raw_a) if raw_a else []
    if scan_nets:
        methods_ok.append("scan")
    raw_b, err_b = _run_cmd(["netsh", "wlan", "show", "interfaces"])
    if DEBUG_RF and raw_b:
        rf_logger.debug("[CMD-B RAW]\n%s", raw_b[:600])
    if err_b:
        errors.append(f"CMD-B: {err_b}")
    connected_net = _parse_connected_interface(raw_b) if raw_b else None
    if connected_net:
        methods_ok.append("interfaces")
    raw_c, err_c = _run_cmd(["netsh", "wlan", "show", "profiles"])
    if DEBUG_RF and raw_c:
        rf_logger.debug("[CMD-C RAW]\n%s", raw_c[:600])
    if err_c:
        errors.append(f"CMD-C: {err_c}")
    profile_nets = _parse_profiles(raw_c) if raw_c else []
    if profile_nets:
        methods_ok.append("profiles")
    all_nets        = _merge_networks(scan_nets, connected_net, profile_nets)
    result.networks = all_nets
    result.threats  = _classify_threats(all_nets)
    result.scan_time= round(time.monotonic() - t0, 2)
    result.scan_method = " + ".join(methods_ok) if methods_ok else "none"

    if errors:
        result.error = " | ".join(errors)

    if DEBUG_RF:
        result.raw_output = (f"=== CMD-A ===\n{raw_a}\n"
                             f"=== CMD-B ===\n{raw_b}\n"
                             f"=== CMD-C ===\n{raw_c}")

    rf_logger.info("[SCAN DONE] %.2fs  networks=%d  threats=%d  method=%s",
                   result.scan_time, len(all_nets),
                   len(result.threats), result.scan_method)
    return result

class BackgroundRFScanner:

    def __init__(self):
        self._lock   = threading.Lock()
        self._result = None
        self._thread = None
        self._stop   = threading.Event()

    def start(self):
        """Start the daemon thread. Safe to call multiple times (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="rf-scanner"
        )
        self._thread.start()
        rf_logger.info("RF background scanner started (interval=%ds)", RF_SCAN_INTERVAL)

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            result = scan_wifi()
            with self._lock:
                self._result = result
            self._stop.wait(timeout=RF_SCAN_INTERVAL)

    @property
    def latest_result(self) -> RFResult:
        with self._lock:
            return self._result if self._result else RFResult()

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())
class LogManager:

    def __init__(self, maxlen: int = MAX_LOG_ENTRIES, cooldown: float = LOG_COOLDOWN):
        self._entries   = deque(maxlen=maxlen)
        self._cooldown  = cooldown
        self._last_seen = {}        
        self._lock      = threading.Lock()

    def add(self, message: str, level: str = "info"):

        now = time.time()
        with self._lock:
            if now - self._last_seen.get(message, 0) < self._cooldown:
                return
            self._last_seen[message] = now
            self._entries.appendleft({
                "time":    datetime.now().strftime("%H:%M:%S"),
                "message": message,
                "level":   level,
            })

    def recent(self, n: int = 14) -> list:
        with self._lock:
            return list(self._entries)[:n]
def _init_session():
    if "app_ready" in st.session_state:
        return

    st.session_state["yolo_model"]   = load_yolo()

    scanner = BackgroundRFScanner()
    scanner.start()
    st.session_state["rf_scanner"]   = scanner
    st.session_state["log"]          = LogManager()
    st.session_state["threat_state"] = ThreatState()
    st.session_state["app_ready"]    = True
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap');

:root {
  --bg:        #070b0f;
  --surface:   #0c1016;
  --border:    #18232e;
  --border-hi: #1d3a58;
  --text:      #c0d2e4;
  --dim:       #3d5570;
  --accent:    #00ccff;
  --safe:      #00e07a;
  --warn:      #f0c000;
  --danger:    #ff3030;
  --mono:      'Share Tech Mono', monospace;
  --sans:      'Syne', sans-serif;
}
html, body, [data-testid="stApp"] {
  background: var(--bg) !important; color: var(--text) !important;
  font-family: var(--sans) !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"] { display:none !important; }
section[data-testid="stSidebar"] { display:none !important; }
.block-container { padding: .8rem 1.4rem !important; max-width:100% !important; }

.pg-header {
  display:flex; align-items:center; justify-content:space-between;
  border-bottom:1px solid var(--border-hi); padding:.5rem 0 .7rem;
  margin-bottom:.9rem;
}
.pg-title { font-family:var(--sans); font-weight:800; font-size:1.3rem;
  letter-spacing:.08em; color:var(--accent); text-transform:uppercase; }
.pg-sub   { font-family:var(--mono); font-size:.65rem; color:var(--dim); }
.pg-badge { font-family:var(--mono); font-size:.6rem; padding:.18rem .5rem;
  border:1px solid var(--accent); color:var(--accent); border-radius:3px;
  letter-spacing:.1em; }

.pg-card { background:var(--surface); border:1px solid var(--border);
  border-radius:7px; padding:.9rem 1rem; margin-bottom:.65rem; }
.pg-card-title { font-family:var(--mono); font-size:.6rem; letter-spacing:.14em;
  color:var(--dim); text-transform:uppercase; margin-bottom:.6rem; }

.score-wrap { display:flex; align-items:center; gap:1.1rem; }
.score-ring { position:relative; width:78px; height:78px; flex-shrink:0; }
.score-ring svg { transform:rotate(-90deg); }
.ring-bg  { fill:none; stroke:var(--border); stroke-width:7; }
.ring-val { fill:none; stroke-width:7; stroke-linecap:round;
  transition:stroke-dashoffset .5s ease, stroke .4s; }
.score-num { position:absolute; inset:0; display:flex; align-items:center;
  justify-content:center; font-family:var(--sans); font-weight:800;
  font-size:1.35rem; color:var(--text); }
.score-meta { flex:1; }
.score-label { font-family:var(--sans); font-weight:700; font-size:1rem; }
.score-conf  { font-family:var(--mono); font-size:.65rem; color:var(--dim); margin-top:.2rem; }
.score-conf span { color:var(--text); }
.status-safe       .score-label { color:var(--safe);   }
.status-suspicious .score-label { color:var(--warn);   }
.status-danger     .score-label { color:var(--danger); }
.status-safe       .ring-val { stroke:var(--safe);   }
.status-suspicious .ring-val { stroke:var(--warn);   }
.status-danger     .ring-val { stroke:var(--danger); }

.mod-row { display:flex; align-items:center; gap:.55rem; padding:.32rem 0;
  border-bottom:1px solid var(--border); font-family:var(--mono); font-size:.68rem; }
.mod-row:last-child { border-bottom:none; }
.mod-icon { font-size:.85rem; }
.mod-name { color:var(--dim); min-width:95px; }
.mod-val  { color:var(--text); flex:1; }
.badge { font-size:.57rem; padding:.08rem .3rem; border-radius:2px;
  font-family:var(--mono); letter-spacing:.07em; }
.b-ok    { background:#001f12; color:var(--safe);   border:1px solid #004028; }
.b-warn  { background:#221900; color:var(--warn);   border:1px solid #4a3500; }
.b-alert { background:#1f0808; color:var(--danger); border:1px solid #4a1010; }
.b-off   { background:#0e1520; color:var(--dim);    border:1px solid var(--border); }

.net-item { display:flex; align-items:center; gap:.45rem; padding:.25rem 0;
  border-bottom:1px solid var(--border); font-family:var(--mono); font-size:.65rem; }
.net-item:last-child { border-bottom:none; }
.net-dot  { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
.d-safe   { background:var(--safe); }
.d-warn   { background:var(--warn); }
.d-alert  { background:var(--danger); }
.d-dim    { background:var(--dim); }
.net-src  { color:var(--dim); font-size:.55rem; min-width:55px; }
.net-ssid { flex:1; color:var(--text); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.net-sig  { color:var(--dim); min-width:36px; text-align:right; }
.net-bar  { color:var(--accent); letter-spacing:0; min-width:34px; font-size:.6rem; }

.threat-box { background:#1a0505; border:1px solid #5a1010; border-radius:6px;
  padding:.55rem .8rem; margin-bottom:.5rem; }
.threat-title { font-family:var(--mono); font-size:.6rem; color:var(--danger);
  letter-spacing:.12em; margin-bottom:.4rem; text-transform:uppercase; }
.threat-net { font-family:var(--mono); font-size:.7rem; color:#ff7070; padding:.15rem 0; }

.log-panel { font-family:var(--mono); font-size:.64rem;
  max-height:190px; overflow-y:auto; }
.log-row { display:flex; gap:.55rem; padding:.2rem 0;
  border-bottom:1px solid #0d1620; }
.log-time { color:var(--dim); flex-shrink:0; }
.l-info   .log-msg { color:var(--text); }
.l-warn   .log-msg { color:var(--warn); }
.l-danger .log-msg { color:var(--danger); }

[data-testid="stImage"] img {
  border:1px solid var(--border-hi) !important; border-radius:6px !important;
  width:100% !important; }

[data-testid="stToggle"] label p { font-family:var(--mono) !important;
  font-size:.65rem !important; color:var(--dim) !important; }

::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-track { background:var(--surface); }
::-webkit-scrollbar-thumb { background:var(--border-hi); border-radius:2px; }
</style>
"""
def _ring(score: int) -> str:
    r    = 33
    circ = 2 * np.pi * r
    off  = circ * (1 - score / 100)
    return (f'<svg width="78" height="78" viewBox="0 0 78 78">'
            f'<circle class="ring-bg" cx="39" cy="39" r="{r}"/>'
            f'<circle class="ring-val" cx="39" cy="39" r="{r}" '
            f'stroke-dasharray="{circ:.1f}" stroke-dashoffset="{off:.1f}" '
            f'style="transform-origin:center;transform:rotate(-90deg) scaleX(-1)"/>'
            f'</svg>'
            f'<div class="score-num">{score}</div>')


def _badge(text: str, kind: str) -> str:
    return f'<span class="badge b-{kind}">{text}</span>'


def _net_dot_cls(is_threat: bool, signal: int, source: str) -> str:
    if is_threat:            return "d-alert"
    if source == "connected": return "d-warn"
    if signal >= 50:          return "d-safe"
    return "d-dim"


def _log_rows(entries: list) -> str:
    html = ""
    for e in entries:
        html += (f'<div class="log-row l-{e["level"]}">'
                 f'<span class="log-time">{e["time"]}</span>'
                 f'<span class="log-msg">{e["message"]}</span>'
                 f'</div>')
    return html

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    _init_session()
    model   = st.session_state["yolo_model"]
    scanner = st.session_state["rf_scanner"]
    log     = st.session_state["log"]
    state   = st.session_state["threat_state"]

    if not scanner.is_running:
        scanner.start()
    st.markdown("""
    <div class="pg-header">
      <div>
        <div class="pg-title">&#x1F6E1; NIZHAL VIZHI</div>
        <div class="pg-sub">Real-time surveillance threat detection</div>
      </div>
      <div class="pg-badge">LIVE</div>
    </div>
    """, unsafe_allow_html=True)
    vid_col, dash_col = st.columns([3, 2], gap="medium")

    with vid_col:
        c1, c2, c3, c4 = st.columns(4)
        run_app    = c1.toggle("Live",  value=True,  key="run")
        show_yolo  = c2.toggle("CV",    value=True,  key="yolo")
        show_glint = c3.toggle("Glint", value=True,  key="glint")
        show_rf    = c4.toggle("RF",    value=True,  key="rf")
        video_slot = st.empty()

    with dash_col:
        score_slot  = st.empty()
        module_slot = st.empty()
        rf_slot     = st.empty()
        log_slot    = st.empty()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error(
            "Webcam not accessible. "
            "Try changing cv2.VideoCapture(0) to VideoCapture(1) "
            "if you have multiple cameras."
        )
        return


    try:
        while run_app:
            ret, frame = cap.read()
            if not ret or frame is None:
                log.add("Webcam: frame dropped", "warn")
                time.sleep(0.04)
                continue

            frame = cv2.flip(frame, 1)   # mirror for natural selfie feel

            # -- 1. YOLO Computer Vision --------------------------------------
            try:
                if show_yolo and model is not None:
                    frame, yolo_res = run_yolo(frame, model)
                    state.yolo      = yolo_res
                    if yolo_res.threat_count:
                        labels = ", ".join(d["label"] for d in yolo_res.detections)
                        log.add(f"[CV] Surveillance object -> {labels}", "danger")
                elif not show_yolo:
                    state.yolo = YOLOResult()
            except Exception as exc:
                log.add(f"[CV] Error: {exc}", "warn")

            # -- 2. Optical Glint Detection ------------------------------------
            try:
                if show_glint:
                    frame, glint_res = run_glint_detection(frame)
                    state.glint      = glint_res
                    if glint_res.threat_count:
                        log.add(
                            f"[GLINT] {glint_res.threat_count} micro-lens "
                            f"reflection(s) detected", "danger"
                        )
                else:
                    state.glint = GlintResult()
            except Exception as exc:
                log.add(f"[GLINT] Error: {exc}", "warn")

            try:
                if show_rf:
                    rf_res   = scanner.latest_result
                    state.rf = rf_res
                    if rf_res.threats:
                        for t in rf_res.threats:
                            log.add(
                                f"[RF] Suspicious: {t.ssid or '[hidden]'} "
                                f"({t.signal}%) [{t.source}]", "danger"
                            )
                    elif rf_res.networks:
                        log.add(
                            f"[RF] {len(rf_res.networks)} networks via "
                            f"{rf_res.scan_method or '...'} -- clear", "info"
                        )
                else:
                    state.rf = RFResult()
            except Exception as exc:
                log.add(f"[RF] Error: {exc}", "warn")

            score      = state.privacy_score
            conf       = state.threat_confidence
            s_label, s_cls = state.status
            video_slot.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            score_slot.markdown(f"""
            <div class="pg-card status-{s_cls}">
              <div class="pg-card-title">Privacy Health Score</div>
              <div class="score-wrap">
                <div class="score-ring">{_ring(score)}</div>
                <div class="score-meta">
                  <div class="score-label">{s_label}</div>
                  <div class="score-conf">
                    Threat confidence &nbsp;<span>{conf:.0%}</span>
                  </div>
                  <div class="score-conf" style="margin-top:.35rem;">
                    CV <span>{state.yolo.threat_count}</span> &nbsp;
                    Glint <span>{state.glint.threat_count}</span> &nbsp;
                    RF <span>{len(state.rf.threats)}</span>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)            
            yb = _badge("ACTIVE", "ok") if YOLO_AVAILABLE else _badge("OFFLINE", "off")
            yv = f"{state.yolo.threat_count} object(s)" if state.yolo.threat_count else "Clear"
            gv = f"{state.glint.threat_count} glint(s)" if state.glint.threat_count else "Clear"
            rv = (f"{len(state.rf.threats)} threat(s)" if state.rf.threats
                  else f"{len(state.rf.networks)} networks ({state.rf.scan_method or '...'})")

            module_slot.markdown(f"""
            <div class="pg-card">
              <div class="pg-card-title">Detection Modules</div>
              <div class="mod-row">
                <span class="mod-icon">&#x1F441;</span>
                <span class="mod-name">Computer Vision</span>
                <span class="mod-val">{yv}</span>
                {_badge(yv.upper(), "alert" if state.yolo.threat_count else "ok")}
                {yb}
              </div>
              <div class="mod-row">
                <span class="mod-icon">&#x2726;</span>
                <span class="mod-name">Optical Glint</span>
                <span class="mod-val">{gv}</span>
                {_badge(gv.upper(), "alert" if state.glint.threat_count else "ok")}
              </div>
              <div class="mod-row">
                <span class="mod-icon">&#x1F4E1;</span>
                <span class="mod-name">RF / WiFi</span>
                <span class="mod-val">{rv}</span>
                {_badge("THREAT" if state.rf.threats else "CLEAR",
                        "alert" if state.rf.threats else "ok")}
              </div>
            </div>
            """, unsafe_allow_html=True)
            if show_rf:
                rf_html = ""

                # Red threat highlight (only shown when threats exist)
                if state.rf.threats:
                    rows = ""
                    for t in state.rf.threats:
                        src = f"[{t.source}]" if t.source != "scan" else ""
                        rows += (f'<div class="threat-net">'
                                 f'&#x26A0; {t.ssid or "[hidden SSID]"} '
                                 f'-- {t.signal}%  {src}'
                                 f'</div>')
                    rf_html += (f'<div class="threat-box">'
                                f'<div class="threat-title">&#x1F6A8; Suspicious Networks</div>'
                                f'{rows}'
                                f'</div>')
                if state.rf.networks:
                    threat_bssids = {t.bssid for t in state.rf.threats}
                    threat_ssids  = {t.ssid.lower() for t in state.rf.threats}
                    net_rows = ""

                    for net in state.rf.networks[:14]:   # cap at 14 rows
                        is_threat = (net.bssid in threat_bssids or
                                     net.ssid.lower() in threat_ssids)
                        dot       = _net_dot_cls(is_threat, net.signal, net.source)
                        ssid_show = net.ssid if net.ssid else "<em>[hidden]</em>"
                        sig_show  = f"{net.signal}%" if net.signal else "--"
                        src_label = {
                            "scan":      "scan",
                            "connected": "conn",
                            "profile":   "saved",
                        }.get(net.source, net.source)

                        net_rows += (
                            f'<div class="net-item">'
                            f'<div class="net-dot {dot}"></div>'
                            f'<span class="net-src">{src_label}</span>'
                            f'<span class="net-ssid">{ssid_show}</span>'
                            f'<span class="net-bar">{net.signal_bar}</span>'
                            f'<span class="net-sig">{sig_show}</span>'
                            f'</div>'
                        )

                    age = int(time.time() - state.rf.last_updated)
                    rf_html += (
                        f'<div class="pg-card">'
                        f'<div class="pg-card-title">'
                        f'WiFi Networks'
                        f'<span style="float:right;color:var(--dim)">'
                        f'{age}s ago &middot; {len(state.rf.networks)} found'
                        f'</span></div>'
                        f'{net_rows}'
                        f'</div>'
                    )

                elif state.rf.error:
                    rf_html += (
                        f'<div class="pg-card">'
                        f'<div class="pg-card-title">WiFi Networks</div>'
                        f'<div style="font-family:var(--mono);font-size:.65rem;'
                        f'color:var(--warn)">&#x26A0; {state.rf.error}</div>'
                        f'</div>'
                    )
                else:
                    rf_html += (
                        f'<div class="pg-card">'
                        f'<div class="pg-card-title">WiFi Networks</div>'
                        f'<div style="font-family:var(--mono);font-size:.65rem;'
                        f'color:var(--dim)">Scanning... first result in {RF_SCAN_INTERVAL}s</div>'
                        f'</div>'
                    )

                rf_slot.markdown(rf_html, unsafe_allow_html=True)
            entries = log.recent(14)
            if entries:
                log_slot.markdown(
                    f'<div class="pg-card">'
                    f'<div class="pg-card-title">Event Log</div>'
                    f'<div class="log-panel">{_log_rows(entries)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    finally:
        cap.release()
        log.add("Webcam released", "info")
if __name__ == "__main__":
    main()

