import sys, time, json, math, queue, pathlib, threading
import numpy as np
import sounddevice as sd
import pyttsx3
import pandas as pd
import matplotlib.pyplot as plt
import random
from alertlistworker import generate_alert_messages

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar,
    QPushButton, QHBoxLayout
)

# ---------- Config ----------
SAMPLE_RATE = 16000
BLOCK_MS    = 100
BLOCKSIZE   = int(SAMPLE_RATE * (BLOCK_MS / 1000.0))
EMA_ALPHA   = 0.8
LOG_PATH    = pathlib.Path("noise_history.jsonl")

DBFS_MIN, DBFS_MAX = -90.0, 0.0

# Alerting / TTS
MODES = {
    "Quite":      (-57.0, -70.0),
    "Moderate":   (-45.0, -50.0), # default
    "Loud":       (-10.0, -15.0),
}
THRESHOLD_ON_DBFS  = -45.0    # speak when EMA >= this, fallback if error on load
THRESHOLD_OFF_DBFS = -50.0    # re-arm when EMA <= this (hysteresis), fallback if error on load
REARM_HOLD_S       = 1.0      # must stay below OFF this long to re-arm
ALERT_COOLDOWN_S   = 10.0      # minimum seconds between spoken alerts
REPEAT_WHILE_LOUD_EVERY_S = 5.0  # None to disable repeats while loud

TTS_VOLUME   = 1.0            # 0.0..1.0 
TTS_RATE_WPM = 150          # try 160–220; or None for default

# ---------- Alerts fallback messages (if API fails) ----------
FALLBACK_ALERTS = [
    "Please lower your voice to maintain a quiet environment.",
    "Let's keep the noise down for everyone's comfort.",
    "A quiet space helps everyone focus better.",
    "Remember to speak softly to respect others.",
    "Your cooperation in keeping the area quiet is appreciated.",
    "Please be mindful of your volume.",
]

def dbfs_from_pcm(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12)
    dbfs = 20.0 * math.log10(rms + 1e-12)
    return max(DBFS_MIN, min(dbfs, DBFS_MAX))

# ---------- Alert messages worker (from alertlistworker.py) ----------
class AlertMessageGenerator:
    def __init__(self, scenario: str):
        self.scenario = scenario
        self._latest: list[str] | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._req = queue.Queue()  # commands: "load" or "refresh" or "quit"
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        # kick off initial load
        self.load_async()

    def _loop(self):
        while True:
            cmd = self._req.get()
            if cmd == "quit":
                return
            try:
                msgs = generate_alert_messages(self.scenario, THRESHOLD_OFF_DBFS)
                if msgs and len(msgs) >= 3:
                    with self._lock:
                        self._latest = msgs
                    self._ready.set()
            except Exception:
                # swallow errors or log; don't crash thread
                pass

    def load_async(self):
        try: self._req.put_nowait("load")
        except queue.Full: pass

    def refresh_async(self):
        try: self._req.put_nowait("refresh")
        except queue.Full: pass

    def get_latest(self, timeout: float | None = 0.0) -> list[str] | None:
        # non-blocking by default; can wait a little on first start if you want
        self._ready.wait(timeout=timeout)
        with self._lock:
            return list(self._latest) if self._latest else None

    def shutdown(self):
        try: self._req.put_nowait("quit")
        except queue.Full: pass

# ---------- Single TTS worker with queue (so speech never overlaps) ----------
class TTSWorker:
    """pyttsx3 re-init per utterance to avoid 'second runAndWait does nothing' bug."""
    def __init__(self, volume=1.0, rate=None, max_queue=20):
        self.q = queue.Queue(maxsize=max_queue)
        # store desired props; apply on each fresh engine
        self.volume = float(max(0.0, min(1.0, volume)))
        self.rate   = int(rate) if rate is not None else None
        self._stop = False
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while True:
            text = self.q.get()
            if text is None:  # shutdown signal
                break
            try:
                eng = pyttsx3.init()
                try:
                    eng.setProperty("volume", self.volume)  # 0.0..1.0
                    if self.rate is not None:
                        eng.setProperty("rate", self.rate)   # words/min
                except Exception:
                    pass
                eng.say(text)
                eng.runAndWait()
                try:
                    eng.stop()  # be nice to the backend
                except Exception:
                    pass
                del eng
            except Exception:
                # swallow TTS errors so app keeps running
                pass

    def say(self, text: str):
        try:
            self.q.put_nowait(text)
        except queue.Full:
            # drop if queue full; or use put() to block
            pass

    def shutdown(self):
        """Call on app exit to stop the worker thread cleanly."""
        try:
            self.q.put_nowait(None)
        except queue.Full:
            # force stop if needed
            pass

class NoiseMeter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Class Noise Meter (PyQt6)")
        self.resize(560, 240)

        self.queue = queue.Queue(maxsize=50)
        self.ema_value = None
        self.logging = False
        self.log_file = None
        self.stream = None

        # Setup detecting config
        respn = input(f"Select noise mode {list(MODES.keys())}, or enter custom ON,OFF dBFS: ")
        if respn in MODES:
            THRESHOLD_ON_DBFS, THRESHOLD_OFF_DBFS = MODES[respn]
            print(f"Using preset '{respn}': ON={THRESHOLD_ON_DBFS} dBFS, OFF={THRESHOLD_OFF_DBFS} dBFS")
        else:
            try:
                on_dbfs, off_dbfs = map(float, respn.split(","))
                if on_dbfs <= off_dbfs:
                    raise ValueError("ON must be greater than OFF")
                THRESHOLD_ON_DBFS, THRESHOLD_OFF_DBFS = on_dbfs, off_dbfs
                print(f"Using custom thresholds: ON={THRESHOLD_ON_DBFS} dBFS, OFF={THRESHOLD_OFF_DBFS} dBFS")
            except Exception as e:
                print(f"Invalid input, using default ON={THRESHOLD_ON_DBFS} dBFS, OFF={THRESHOLD_OFF_DBFS} dBFS")
                
        # Alert state
        self.last_alert_ts = 0.0
        self.alert_count_spoken = 0
        self.over_armed = True
        self.below_off_since = None    # NEW: track how long we've been below OFF

        # TTS worker (single queue)
        self.tts = TTSWorker(volume=TTS_VOLUME, rate=TTS_RATE_WPM)

        # Setup alerts input
        self.scenario = input("Enter the scenario to generate alert messages for: ")
        self.alertsworker = AlertMessageGenerator(self.scenario)
        self.alert_messages = None
        self._first_check_deadline = time.time() + 2.0

        # Clear old log
        if LOG_PATH.exists():
            try:
                LOG_PATH.unlink()
            except Exception:
                pass

        # --- UI ---
        layout = QVBoxLayout(self)
        self.label = QLabel("Current level: --.- dBFS", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.label.setStyleSheet("font-size: 24px;")

        self.bar = QProgressBar(self)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(24)

        self.status = QLabel("Status: stopped", self)
        self.status.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        btns = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop  = QPushButton("Stop")
        self.btn_plot  = QPushButton("Chart / Table")
        self.btn_stop.setEnabled(False)

        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_plot)

        layout.addWidget(self.label)
        layout.addWidget(self.bar)
        layout.addWidget(self.status)
        layout.addLayout(btns)

        self.btn_start.clicked.connect(self.start_measure)
        self.btn_stop.clicked.connect(self.stop_measure)
        self.btn_plot.clicked.connect(self.show_chart_table)

        self.timer = QTimer(self)
        self.timer.setInterval(BLOCK_MS)
        self.timer.timeout.connect(self.on_tick)



    # ---------------- Audio ----------------
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            pass
        x = indata[:, 0].astype(np.float32)
        level = dbfs_from_pcm(x)
        try:
            self.queue.put_nowait((time.time(), level))
        except queue.Full:
            _ = self.queue.get_nowait()
            self.queue.put_nowait((time.time(), level))

    def start_measure(self):
        if self.stream is not None:
            return
        self.log_file = LOG_PATH.open("a", encoding="utf-8")
        self.logging = True
        self.last_alert_ts = 0.0
        self.alert_count_spoken = 0
        self.over_armed = True

        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
                dtype="float32",
                callback=self.audio_callback,
            )
            self.stream.start()
        except Exception as e:
            self.status.setText(f"Status: ERROR opening mic → {e}")
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            self.logging = False
            self.stream = None
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText("Status: running")
        self.timer.start()

    def stop_measure(self):
        self.timer.stop()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("Status: stopped")

        if self.log_file:
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None

        self.logging = False
        self.ema_value = None

    # ------------- GUI tick ---------------
    def on_tick(self):
        if self.alert_messages is None:
            msgs = self.alertsworker.get_latest(timeout=0.0)  # never block the GUI
            if msgs:
                self.alert_messages = msgs
                self.status.setText("Status: alert messages ready")
            else:
                # still loading; don't kill the app
                if time.time() < self._first_check_deadline:
                    self.status.setText("Status: loading alert messages…")
                else:
                    # still not ready; keep running but skip alert logic this tick
                    self.status.setText("Status: still fetching messages…")
                return  # <- EARLY RETURN, no alert logic yet
        # Drain queue, keep latest
        latest = None
        try:
            while True:
                latest = self.queue.get_nowait()
        except queue.Empty:
            pass
        if latest is None:
            return

        ts, raw_dbfs = latest

        # EMA
        if self.ema_value is None:
            self.ema_value = raw_dbfs
        else:
            self.ema_value = EMA_ALPHA * self.ema_value + (1.0 - EMA_ALPHA) * raw_dbfs

        # UI
        self.label.setText(f"Current level: {self.ema_value:6.1f} dBFS")
        norm = (self.ema_value - DBFS_MIN) / (DBFS_MAX - DBFS_MIN)
        pct  = int(100 * max(0.0, min(1.0, norm)))
        self.bar.setValue(pct)

        # Log
        if self.logging and self.log_file:
            rec = {"ts": ts, "dbfs": round(float(self.ema_value), 2)}
            self.log_file.write(json.dumps(rec) + "\n")

        # ---------- Alert logic: rising-edge + hysteresis + cooldown + optional repeat ----------
        now = time.time()
        over_now = (self.ema_value >= THRESHOLD_ON_DBFS)
        below_off = (self.ema_value <= THRESHOLD_OFF_DBFS)

        # Track how long we've been below OFF (for re-arm stability)
        if below_off:
            if self.below_off_since is None:
                self.below_off_since = now
            # If we've been below OFF long enough, re-arm
            if (now - self.below_off_since) >= REARM_HOLD_S:
                self.over_armed = True
        else:
            self.below_off_since = None

        # 1) Rising-edge alert (only when armed) + cooldown
        if over_now and self.over_armed and (now - self.last_alert_ts) >= ALERT_COOLDOWN_S:
            self.over_armed = False          # disarm until we spend time below OFF
            self.last_alert_ts = now
            self.alert_count_spoken += 1
            # if self.alert_count_spoken <= len(self.alert_messages):
            if self.alert_messages:
                next_msg = self.alert_messages.pop(0)
            else:
                next_msg = random.choice(FALLBACK_ALERTS)
            # else:
            #     next_msg = "Out of preset messages, required human control."
            print(next_msg)
            self.tts.say(next_msg)
            print(f"[ALERT] rising-edge #{self.alert_count_spoken} at {time.strftime('%H:%M:%S')}")

        # 2) Optional repeat while still loud (even if not re-armed)
        if (REPEAT_WHILE_LOUD_EVERY_S is not None and
            over_now and
            (now - self.last_alert_ts) >= max(ALERT_COOLDOWN_S, REPEAT_WHILE_LOUD_EVERY_S)):
            self.last_alert_ts = now
            self.alert_count_spoken += 1
            # if self.alert_count_spoken <= len(self.alert_messages):
            #     next_msg = self.alert_messages[(self.alert_count_spoken - 1) % len(self.alert_messages)]
            # else:
            #     next_msg = "Out of preset messages, required human control."
            if self.alert_messages:
                next_msg = self.alert_messages.pop(0)
            else:
                next_msg = random.choice(FALLBACK_ALERTS)
            print(next_msg)         
            self.tts.say(next_msg)
            print(f"[ALERT] repeat-while-loud #{self.alert_count_spoken} at {time.strftime('%H:%M:%S')}")

        # -------- Alert logic check ------------
        self.alert_messages_helper()
    
    def alert_messages_helper(self):
        if self.alert_messages and len(self.alert_messages) < 3:
            self.alertsworker.refresh_async()
            msgs = self.alertsworker.get_latest(timeout=0.0)  # non-blocking peek
            if msgs and len(msgs) >= 3:
                self.alert_messages = msgs
                self.status.setText("Status: alert messages refreshed")
            else:
                # Keep using the current (possibly short) list; do not blank it out.
                self.status.setText("Status: refreshing messages…")

    # ---------- Plot: line chart + table ----------
    def show_chart_table(self):
        if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
            self.status.setText("Status: no data to plot yet")
            return

        rows = []
        with LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not rows:
            self.status.setText("Status: no valid rows in log")
            return

        df = pd.DataFrame(rows)
        if "ts" not in df.columns or "dbfs" not in df.columns:
            self.status.setText("Status: invalid log format")
            return

        t0 = df["ts"].iloc[0]
        df["t_rel_s"] = df["ts"] - t0 

        total = len(df)
        above = int((df["dbfs"] >= THRESHOLD_ON_DBFS).sum())
        quiet_ratio = 1.0 - (above / total)
        mean_dbfs = float(df["dbfs"].mean())
        max_dbfs = float(df["dbfs"].max())

        # Use spoken alert count
        alerts = self.alert_count_spoken

        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_axes([0.08, 0.42, 0.9, 0.55])
        ax2 = fig.add_axes([0.08, 0.05, 0.9, 0.27])

        ax1.plot(df["t_rel_s"], df["dbfs"])
        ax1.axhline(THRESHOLD_ON_DBFS, linestyle="--", linewidth=1)
        ax1.axhline(THRESHOLD_OFF_DBFS, linestyle=":", linewidth=1)
        ax1.set_title("Noise level over time (dBFS)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("dBFS (0=max)")
        ax1.grid(True, alpha=0.3)

        ax2.axis("off")
        summary_data = [
            ["Samples recorded", total],
            ["Quiet ratio", f"{quiet_ratio*100:.1f}%"],
            ["Mean dBFS", f"{mean_dbfs:.1f}"],
            ["Max dBFS",  f"{max_dbfs:.1f}"],
            ["Threshold ON",  f"{THRESHOLD_ON_DBFS:.1f} dBFS"],
            ["Threshold OFF", f"{THRESHOLD_OFF_DBFS:.1f} dBFS"],
            ["# Spoken alerts", alerts],
            ["App run time", f"{df['t_rel_s'].iloc[-1]:.1f} s"],
        ]
        table = ax2.table(cellText=summary_data,
                          colLabels=["Metric", "Value"],
                          loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.3)

        plt.show()

    def closeEvent(self, event):
        self.stop_measure()
        self.tts.shutdown()
        try:
            self.alertsworker.shutdown()
        except Exception:
            pass
        return super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    w = NoiseMeter()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()