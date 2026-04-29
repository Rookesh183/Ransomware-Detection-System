# MERGED RANSOMWARE DETECTION SYSTEM

from collections import Counter
from collections import deque
from cryptography.fernet import Fernet
from datetime import datetime
from flask import Flask, render_template, jsonify, send_file
from fpdf import FPDF
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import glob
import json
import logging
import math
import numpy as np
import os
import pickle
import psutil
import random
import shutil
import socket
import sys
import threading
import time
import warnings



try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object


# --- EARLY GLOBALS ---

app = Flask(__name__)


# --- CLASSES & FUNCTIONS ---

def calculate_shannon_entropy(data):
    """
    Calculate Shannon entropy H(X) = -Σ P(xi) * log2(P(xi))
    
    Shannon entropy measures the randomness/unpredictability of data.
    Higher entropy (close to 8.0 for byte data) suggests encrypted content,
    which is a strong indicator of ransomware activity.
    
    Args:
        data (bytes or str): Input data to calculate entropy for
        
    Returns:
        float: Shannon entropy value (0.0 to 8.0 for byte data)
    """
    if not data:
        return 0.0
    if isinstance(data, str):
        data = data.encode('utf-8')
    byte_counts = Counter(data)
    total_bytes = len(data)
    entropy = 0.0
    for count in byte_counts.values():
        probability = count / total_bytes
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return round(entropy, 4)

def get_process_features(proc):
    """
    Extract 10 behavioral features from a single process.
    
    Uses psutil to gather real-time process metrics and simulates
    file I/O operations based on the number of open file handles.
    
    Args:
        proc (psutil.Process): Process object to analyze
        
    Returns:
        dict: Dictionary containing all 10 behavioral features,
              or None if the process cannot be accessed
    """
    try:
        pinfo = proc.as_dict(attrs=['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'num_threads', 'create_time', 'cmdline'])
        try:
            open_files = len(proc.open_files())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            open_files = 0
        try:
            io = proc.io_counters()
            file_read = io.read_count
            file_write = io.write_count
        except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
            file_read = random.randint(0, max(1, open_files * 5))
            file_write = random.randint(0, max(1, open_files * 5))
        file_delete = random.randint(0, max(1, open_files // 3))
        file_rename = random.randint(0, max(1, open_files // 3))
        cpu = pinfo.get('cpu_percent', 0.0) or 0.0
        memory = round(pinfo.get('memory_percent', 0.0) or 0.0, 2)
        threads = pinfo.get('num_threads', 1) or 1
        process_name = pinfo.get('name', '') or ''
        entropy = calculate_shannon_entropy(process_name)
        cmdline = pinfo.get('cmdline', []) or []
        cmd_str = ' '.join(cmdline).lower()
        suspicious_args = ['vssadmin', 'delete', 'shadows', 'wbadmin', 'bcdedit', 'recoveryenabled', 'ignoreallfailures']
        if any((arg in cmd_str for arg in suspicious_args)):
            entropy = max(entropy, min(7.0, entropy + 0.5))
        create_time = pinfo.get('create_time', time.time())
        uptime = round(time.time() - create_time, 2) if create_time else 0
        features = {'pid': pinfo['pid'], 'name': process_name, 'username': pinfo.get('username', 'unknown'), 'open_files': open_files, 'file_read': file_read, 'file_write': file_write, 'file_delete': file_delete, 'file_rename': file_rename, 'cpu': round(cpu, 2), 'memory': memory, 'threads': threads, 'entropy': entropy, 'uptime': uptime}
        return features
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None

def get_process_activity():
    """
    Enumerate all running processes and extract behavioral features.
    
    Iterates through all system processes, extracts 10 behavioral
    features from each, and returns a list of feature dictionaries
    suitable for ML classification.
    
    Returns:
        list: List of dictionaries, each containing process features
    """
    print('\n' + '=' * 60)
    print('[*] 🔍 Starting Process Behavioral Analysis...')
    print('=' * 60)
    activities = []
    skipped = 0
    for proc in psutil.process_iter():
        features = get_process_features(proc)
        if features:
            activities.append(features)
        else:
            skipped += 1
    print(f'[✓] Analyzed {len(activities)} processes successfully')
    if skipped > 0:
        print(f'[!] Skipped {skipped} processes (access denied or terminated)')
    print(f'\n{'=' * 60}')
    print(f'{'Process':<25} {'PID':<8} {'CPU%':<8} {'MEM%':<8} {'Threads':<8} {'Entropy':<8}')
    print(f'{'-' * 60}')
    sorted_procs = sorted(activities, key=lambda x: x['cpu'], reverse=True)[:10]
    for p in sorted_procs:
        print(f'{p['name'][:24]:<25} {p['pid']:<8} {p['cpu']:<8} {p['memory']:<8} {p['threads']:<8} {p['entropy']:<8}')
    print(f'{'=' * 60}')
    print(f'[✓] Total processes monitored: {len(activities)}')
    return activities

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic training data for ransomware detection.
    
    Creates two classes of synthetic behavioral data:
    - Benign processes: Low resource usage, low entropy, long uptime
    - Malicious processes: High file I/O, high CPU/memory, high entropy, short uptime
    
    Feature distributions are designed to model real-world behavioral
    differences between normal applications and ransomware.
    
    Args:
        n_samples (int): Number of samples per class (default: 1000)
        
    Returns:
        tuple: (X, y) where X is feature matrix and y is label vector
    """
    print('[*] 📊 Generating synthetic training data...')
    np.random.seed(42)
    benign = np.column_stack([np.random.uniform(0, 10, n_samples), np.random.uniform(0, 50, n_samples), np.random.uniform(0, 20, n_samples), np.random.uniform(0, 5, n_samples), np.random.uniform(0, 5, n_samples), np.random.uniform(0, 15, n_samples), np.random.uniform(0.1, 5, n_samples), np.random.randint(1, 6, n_samples), np.random.uniform(0, 4.5, n_samples), np.random.uniform(10, 100000, n_samples)])
    malicious = np.column_stack([np.random.uniform(10, 100, n_samples), np.random.uniform(50, 500, n_samples), np.random.uniform(50, 500, n_samples), np.random.uniform(5, 50, n_samples), np.random.uniform(5, 50, n_samples), np.random.uniform(40, 100, n_samples), np.random.uniform(5, 30, n_samples), np.random.randint(8, 33, n_samples), np.random.uniform(6.5, 8.0, n_samples), np.random.uniform(1, 500, n_samples)])
    X = np.vstack([benign, malicious])
    y = np.array([0] * n_samples + [1] * n_samples)
    print(f'[✓] Generated {n_samples * 2} samples ({n_samples} benign + {n_samples} malicious)')
    print(f'[*] Feature matrix shape: {X.shape}')
    return (X, y)

def train_models():
    """
    Train both Decision Tree and MLP classifiers on synthetic data.
    
    Training pipeline:
    1. Generate synthetic data (1000 benign + 1000 malicious)
    2. Split into training (80%) and test (20%) sets
    3. Train Decision Tree with Gini impurity criterion
    4. Train MLP with single hidden layer (8 neurons, ReLU)
    5. Evaluate both models and display metrics
    6. Save trained models to disk as pickle files
    
    Returns:
        tuple: (dt_model, mlp_model) trained classifier objects
    """
    print('\n' + '=' * 60)
    print('  🧠 RANSOMWARE DETECTION - Model Training')
    print('  Machine Learning Classification Engine')
    print('=' * 60)
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = generate_synthetic_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f'\n[*] Training set: {X_train.shape[0]} samples')
    print(f'[*] Test set:     {X_test.shape[0]} samples')
    print(f'\n{'=' * 60}')
    print('[*] 🌳 Training Decision Tree Classifier...')
    print(f'    Algorithm:        CART')
    print(f'    Criterion:        Gini impurity')
    print(f'    Max depth:        15')
    print(f'    Min samples split: 5')
    dt_model = DecisionTreeClassifier(max_depth=15, min_samples_split=5, criterion='gini', random_state=42)
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    dt_f1 = f1_score(y_test, dt_predictions, average='weighted')
    print(f'\n[✓] Decision Tree Results:')
    print(f'    Accuracy:  {dt_accuracy * 100:.2f}%')
    print(f'    F1-Score:  {dt_f1:.4f}')
    print(f'\n    Classification Report:')
    print(classification_report(y_test, dt_predictions, target_names=['Benign', 'Malicious']))
    with open(DT_MODEL_PATH, 'wb') as f:
        pickle.dump(dt_model, f)
    print(f'[✓] Decision Tree saved to: {DT_MODEL_PATH}')
    print(f'\n{'=' * 60}')
    print('[*] 🧬 Training MLP Neural Network...')
    print(f'    Architecture:     Input(10) -> Hidden(8, ReLU) -> Output(2)')
    print(f'    Optimizer:        Adam (lr=0.0001)')
    print(f'    Loss:             Cross-entropy')
    print(f'    Max iterations:   10000')
    mlp_model = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', solver='adam', learning_rate_init=0.0001, max_iter=10000, random_state=42, early_stopping=True, validation_fraction=0.1)
    mlp_model.fit(X_train, y_train)
    mlp_predictions = mlp_model.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, mlp_predictions)
    mlp_f1 = f1_score(y_test, mlp_predictions, average='weighted')
    print(f'\n[✓] MLP Neural Network Results:')
    print(f'    Accuracy:  {mlp_accuracy * 100:.2f}%')
    print(f'    F1-Score:  {mlp_f1:.4f}')
    print(f'\n    Classification Report:')
    print(classification_report(y_test, mlp_predictions, target_names=['Benign', 'Malicious']))
    with open(MLP_MODEL_PATH, 'wb') as f:
        pickle.dump(mlp_model, f)
    print(f'[✓] MLP model saved to: {MLP_MODEL_PATH}')
    print(f'\n{'=' * 60}')
    print(f'  📊 TRAINING SUMMARY')
    print(f'{'=' * 60}')
    print(f'  Decision Tree:  Accuracy={dt_accuracy * 100:.2f}%  F1={dt_f1:.4f}')
    print(f'  MLP Network:    Accuracy={mlp_accuracy * 100:.2f}%  F1={mlp_f1:.4f}')
    print(f'{'=' * 60}')
    print(f'\n[✓] All models trained and saved successfully ✓')
    return (dt_model, mlp_model)

def predict(process_features, model_type='decision_tree'):
    """
    Classify a process as benign or malicious using a trained model.
    
    Loads the specified pre-trained model and classifies the process
    based on its behavioral feature vector.
    
    Args:
        process_features (dict): Dictionary with keys matching FEATURE_NAMES
        model_type (str): 'decision_tree' or 'mlp' (default: 'decision_tree')
        
    Returns:
        tuple: (status, confidence) where:
            - status (str): 'Malicious' or 'Benign'
            - confidence (float): Prediction confidence (0.0 to 1.0)
    """
    if model_type == 'mlp':
        model_path = MLP_MODEL_PATH
    else:
        model_path = DT_MODEL_PATH
    if not os.path.exists(model_path):
        print(f'[!] Model not found at {model_path}')
        print('[*] Training models automatically...')
        train_models()
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    feature_vector = np.array([[process_features.get('open_files', 0), process_features.get('file_read', 0), process_features.get('file_write', 0), process_features.get('file_delete', 0), process_features.get('file_rename', 0), process_features.get('cpu', 0), process_features.get('memory', 0), process_features.get('threads', 1), process_features.get('entropy', 0), process_features.get('uptime', 0)]])
    prediction = model.predict(feature_vector)[0]
    try:
        probabilities = model.predict_proba(feature_vector)[0]
        confidence = round(max(probabilities) * 100, 2)
    except AttributeError:
        confidence = 95.0
    status = 'Malicious' if prediction == 1 else 'Benign'
    return (status, confidence)

def setup_logging():
    """Configures the plain text logger for the application."""
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and (not os.path.exists(log_dir)):
        os.makedirs(log_dir)
    logger = logging.getLogger('pure_detector')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(LOG_FILE)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

def calculate_entropy(filepath):
    """
    Calculates the Shannon entropy of a file's contents.
    High entropy (close to 8.0) indicates the file is likely compressed or encrypted.
    """
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        if not data:
            return 0.0
        file_size = len(data)
        byte_counts = Counter(data)
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / file_size
            entropy -= probability * math.log2(probability)
        return entropy
    except Exception as e:
        logger.debug(f'Could not calculate entropy for {filepath}: {e}')
        return 0.0

def quarantine_file(filepath):
    """
    Safely moves a suspicious file to the quarantine directory to isolate it.
    """
    try:
        if not os.path.exists(QUARANTINE_DIRECTORY):
            os.makedirs(QUARANTINE_DIRECTORY)
        filename = os.path.basename(filepath)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f'{filename}_{timestamp}.quarantined'
        destination = os.path.join(QUARANTINE_DIRECTORY, safe_filename)
        shutil.move(filepath, destination)
        logger.warning(f'ACTION TAKEN: Quarantined suspicious file -> {destination}')
        return {'original_path': filepath, 'quarantine_path': destination}
    except Exception as e:
        logger.error(f'FAILED to quarantine file {filepath}: {e}')
        return None

class DirectoryMonitor:
    """
    Monitors directories for changes using standard library polling.
    Tracks file states to detect modifications, creations, and deletions.
    """

    def __init__(self, watch_dirs=None):
        if watch_dirs is None:
            self.watch_dirs = [WATCH_DIRECTORY]
        elif isinstance(watch_dirs, str):
            self.watch_dirs = [watch_dirs]
        else:
            self.watch_dirs = watch_dirs
        self.file_states = {}
        self.recent_modifications = []
        self.quarantined_files = []
        self.is_running = False
        self._thread = None

    def _get_current_state(self):
        """Scans the directory structures and returns a dictionary of file paths to their modification times."""
        current_state = {}
        for watch_dir in self.watch_dirs:
            if not os.path.exists(watch_dir):
                continue
            try:
                for root, _, files in os.walk(watch_dir):
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        try:
                            mtime = os.path.getmtime(filepath)
                            current_state[filepath] = mtime
                        except OSError:
                            pass
            except Exception as e:
                logger.error(f'Error scanning directory {watch_dir}: {e}')
        return current_state

    def _check_mass_modifications(self):
        """
        Checks if the number of recent modifications exceeds the threshold,
        indicating a potential ransomware attack encrypting many files quickly.
        """
        current_time = time.time()
        self.recent_modifications = [t for t in self.recent_modifications if current_time - t <= MASS_MODIFICATION_WINDOW]
        if len(self.recent_modifications) >= MASS_MODIFICATION_THRESHOLD:
            logger.critical(f'MASS MODIFICATION DETECTED: {len(self.recent_modifications)} files changed within {MASS_MODIFICATION_WINDOW} seconds!')
            return True
        return False

    def _analyze_file(self, filepath, event_type):
        """
        Analyzes a newly created or modified file for ransomware characteristics (High Entropy).
        """
        logger.info(f'File {event_type}: {filepath}')
        self.recent_modifications.append(time.time())
        self._check_mass_modifications()
        time.sleep(0.1)
        if os.path.exists(filepath):
            entropy = calculate_entropy(filepath)
            logger.info(f'Entropy for {filepath}: {entropy:.2f}')
            if entropy >= ENTROPY_THRESHOLD:
                logger.warning(f'HIGH ENTROPY FILE DETECTED: {filepath} (Entropy: {entropy:.2f})')
                quarantine_result = quarantine_file(filepath)
                if quarantine_result:
                    self.quarantined_files.append(quarantine_result)

    def start(self):
        """Starts the monitoring loop in the foreground."""
        for watch_dir in self.watch_dirs:
            if not os.path.exists(watch_dir):
                if watch_dir.startswith('/run/') or watch_dir.startswith('/sys/'):
                    logger.debug(f'Skipping system directory:  {watch_dir}')
                else:
                    try:
                        os.makedirs(watch_dir, exist_ok=True)
                        logger.info(f'Created watch directory: {watch_dir}')
                    except Exception as e:
                        logger.warning(f'Could not create {watch_dir}: {e}')
        logger.info(f'Starting standard-library real-time monitor on: {self.watch_dirs}')
        logger.info(f'Poll interval: {POLL_INTERVAL}s | Entropy Threshold: {ENTROPY_THRESHOLD}')
        self.is_running = True
        self.file_states = self._get_current_state()
        logger.info(f'Initial scan complete. Tracking {len(self.file_states)} existing files.')
        try:
            while self.is_running:
                new_state = self._get_current_state()
                for filepath, new_mtime in new_state.items():
                    if filepath not in self.file_states:
                        self._analyze_file(filepath, 'CREATED')
                    elif new_mtime > self.file_states[filepath]:
                        self._analyze_file(filepath, 'MODIFIED')
                for filepath in self.file_states:
                    if filepath not in new_state:
                        logger.info(f'File DELETED: {filepath}')
                        self.recent_modifications.append(time.time())
                        self._check_mass_modifications()
                self.file_states = new_state
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            self.stop()

    def start_background(self):
        """Starts the monitoring loop in a background thread."""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self.start, daemon=True)
            self._thread.start()

    def stop(self):
        """Stops the monitoring loop gracefully."""
        logger.info('Stopping standard-library real-time monitor. Goodbye.')
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_quarantined_files(self, clear=False):
        """Returns the list of files quarantined during this session."""
        q_files = self.quarantined_files.copy()
        if clear:
            self.quarantined_files.clear()
        return q_files

class ThreatAnalyzer:
    """
    Correlates filesystem events and process data to detect
    ransomware patterns using a time-based sliding window.
    """

    def __init__(self, allowed_paths=None):
        self.allowed_paths = allowed_paths or []
        self._write_events = deque()
        self._rename_events = deque()
        self._suspicious_ext_events = deque()
        self._entropy_spikes = deque()
        self._cpu_spikes = deque()
        self.alerts = deque(maxlen=200)
        self.threats_blocked = 0
        self.files_quarantined = 0
        self.start_time = None
        self._lock = threading.Lock()
        self._responded_pids = set()
        os.makedirs(QUARANTINE_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

    def record_write(self, filepath):
        now = time.time()
        ext = os.path.splitext(filepath)[1].lower()
        with self._lock:
            self._write_events.append((now, filepath))
            if ext in SUSPICIOUS_EXTENSIONS:
                self._suspicious_ext_events.append((now, filepath))
            self._prune_old_events(now)
            self._evaluate_threat(now)

    def record_rename(self, src_path, dest_path):
        now = time.time()
        ext = os.path.splitext(dest_path)[1].lower()
        with self._lock:
            self._rename_events.append((now, src_path, dest_path))
            if ext in SUSPICIOUS_EXTENSIONS:
                self._suspicious_ext_events.append((now, dest_path))
            self._prune_old_events(now)
            self._evaluate_threat(now)

    def record_entropy_spike(self, pid, entropy_value):
        now = time.time()
        with self._lock:
            self._entropy_spikes.append((now, pid, entropy_value))
            self._prune_old_events(now)

    def record_cpu_spike(self, pid, cpu_value):
        now = time.time()
        with self._lock:
            self._cpu_spikes.append((now, pid, cpu_value))
            self._prune_old_events(now)

    def _prune_old_events(self, now):
        cutoff = now - WINDOW_SIZE
        while self._write_events and self._write_events[0][0] < cutoff:
            self._write_events.popleft()
        while self._rename_events and self._rename_events[0][0] < cutoff:
            self._rename_events.popleft()
        while self._suspicious_ext_events and self._suspicious_ext_events[0][0] < cutoff:
            self._suspicious_ext_events.popleft()
        while self._entropy_spikes and self._entropy_spikes[0][0] < cutoff:
            self._entropy_spikes.popleft()
        while self._cpu_spikes and self._cpu_spikes[0][0] < cutoff:
            self._cpu_spikes.popleft()

    def _evaluate_threat(self, now):
        indicators = 0
        reasons = []
        burst_cutoff = now - WRITE_BURST_WINDOW
        recent_writes = sum((1 for t, _ in self._write_events if t >= burst_cutoff))
        if recent_writes >= WRITE_BURST_THRESHOLD:
            indicators += 1
            reasons.append(f'Write burst: {recent_writes} files in {WRITE_BURST_WINDOW}s')
        rename_count = len(self._rename_events)
        if rename_count >= RENAME_BURST_THRESHOLD:
            indicators += 1
            reasons.append(f'Rename burst: {rename_count} renames in {WINDOW_SIZE}s')
        sus_ext_count = len(self._suspicious_ext_events)
        if sus_ext_count >= SUSPICIOUS_EXT_THRESHOLD:
            indicators += 1
            reasons.append(f'Suspicious extensions: {sus_ext_count} files in {WINDOW_SIZE}s')
        if len(self._entropy_spikes) > 0:
            indicators += 1
            reasons.append(f'Entropy spike detected ({len(self._entropy_spikes)} processes)')
        if len(self._cpu_spikes) > 0:
            indicators += 1
            reasons.append(f'CPU spike detected ({len(self._cpu_spikes)} processes)')
        if recent_writes >= WRITE_BURST_THRESHOLD or indicators >= MULTI_INDICATOR_THRESHOLD:
            self._trigger_response(reasons, now)

    def _trigger_response(self, reasons, now):
        affected_files = set()
        for _, fpath in self._write_events:
            affected_files.add(fpath)
        for _, src, dest in self._rename_events:
            affected_files.add(dest)
        for _, fpath in self._suspicious_ext_events:
            affected_files.add(fpath)
        terminated_pids = []
        offending_processes = self._find_offending_processes(affected_files)
        for pid, pname in offending_processes:
            if pid in self._responded_pids:
                continue
            self._responded_pids.add(pid)
            try:
                proc = psutil.Process(pid)
                try:
                    exe = proc.exe()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    exe = None
                if exe and self._path_allowed(exe):
                    proc.suspend()
                    time.sleep(0.1)
                    proc.terminate()
                    terminated_pids.append({'pid': pid, 'name': pname})
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        quarantined = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        q_dir = os.path.join(QUARANTINE_DIR, f'rt_{timestamp}')
        if affected_files:
            os.makedirs(q_dir, exist_ok=True)
            for fpath in affected_files:
                if not self._path_allowed(fpath):
                    continue
                try:
                    if os.path.isfile(fpath):
                        dest = os.path.join(q_dir, os.path.basename(fpath))
                        if os.path.exists(dest):
                            dest = f'{dest}_{len(quarantined)}'
                        shutil.move(fpath, dest)
                        quarantined.append(fpath)
                except Exception:
                    pass
        self.threats_blocked += 1
        self.files_quarantined += len(quarantined)
        alert = {'id': f'RT-{int(now * 1000)}', 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'type': 'RANSOMWARE_ATTACK', 'severity': 'CRITICAL', 'reasons': reasons, 'terminated_processes': terminated_pids, 'quarantined_files': quarantined, 'quarantine_dir': q_dir if quarantined else None, 'affected_file_count': len(affected_files)}
        self.alerts.append(alert)
        self._log_alert(alert)
        self._write_events.clear()
        self._rename_events.clear()
        self._suspicious_ext_events.clear()
        print(f'\n[ALERT] Ransomware attack detected!')
        print(f'    Reasons: {', '.join(reasons)}')
        print(f'    Terminated: {len(terminated_pids)} processes')
        print(f'    Quarantined: {len(quarantined)} files')

    def _find_offending_processes(self, affected_files):
        offenders = []
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    open_files = proc.open_files()
                    for of in open_files:
                        if of.path in affected_files:
                            offenders.append((proc.pid, proc.name()))
                            break
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
        except Exception:
            pass
        return offenders

    def _path_allowed(self, filepath):
        try:
            real = os.path.realpath(os.path.abspath(filepath))
            for base in self.allowed_paths:
                if not base:
                    continue
                base_real = os.path.realpath(os.path.abspath(base))
                if real == base_real or real.startswith(base_real + os.sep):
                    return True
            return False
        except (OSError, ValueError):
            return False

    def _log_alert(self, alert):
        import json
        log_path = os.path.join(LOG_DIR, 'realtime_alerts.json')
        logs = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    logs = json.load(f)
                if not isinstance(logs, list):
                    logs = [logs]
            except Exception:
                logs = []
        logs.append(alert)
        logs = logs[-500:]
        try:
            with open(log_path, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception:
            pass

    def get_window_stats(self):
        now = time.time()
        with self._lock:
            self._prune_old_events(now)
            burst_cutoff = now - WRITE_BURST_WINDOW
            return {'write_count': len(self._write_events), 'write_burst': sum((1 for t, _ in self._write_events if t >= burst_cutoff)), 'rename_count': len(self._rename_events), 'suspicious_ext_count': len(self._suspicious_ext_events), 'entropy_spikes': len(self._entropy_spikes), 'cpu_spikes': len(self._cpu_spikes)}

    def get_recent_alerts(self, limit=50):
        return list(self.alerts)[-limit:]

    def get_stats(self):
        return {'threats_blocked': self.threats_blocked, 'files_quarantined': self.files_quarantined, 'uptime_seconds': round(time.time() - self.start_time, 1) if self.start_time else 0, 'total_alerts': len(self.alerts)}

class RansomwareFileHandler(FileSystemEventHandler):
    """Forwards filesystem events to the ThreatAnalyzer."""

    def __init__(self, threat_analyzer):
        super().__init__()
        self.analyzer = threat_analyzer

    def on_modified(self, event):
        if event.is_directory:
            return
        try:
            self.analyzer.record_write(event.src_path)
        except Exception:
            pass

    def on_created(self, event):
        if event.is_directory:
            return
        try:
            self.analyzer.record_write(event.src_path)
        except Exception:
            pass

    def on_moved(self, event):
        if event.is_directory:
            return
        try:
            self.analyzer.record_rename(event.src_path, event.dest_path)
        except Exception:
            pass

class FileWatcher:
    """Manages watchdog observers for monitored directories."""

    def __init__(self, threat_analyzer, watch_paths=None):
        self.analyzer = threat_analyzer
        self.watch_paths = watch_paths or self._default_paths()
        self._observer = None
        self._running = False
        self._lock = threading.Lock()

    @staticmethod
    def _default_paths():
        paths = [os.path.expanduser('~/Desktop'), os.path.expanduser('~/Documents'), os.path.expanduser('~/Pictures'), '/tmp']
        return [p for p in paths if os.path.isdir(p)]

    def start(self):
        with self._lock:
            if self._running:
                return
            if not WATCHDOG_AVAILABLE:
                print('[!] watchdog not installed, file watching disabled')
                print('    Install with: pip install watchdog')
                return
            self._observer = Observer()
            handler = RansomwareFileHandler(self.analyzer)
            watched = []
            for path in self.watch_paths:
                if os.path.isdir(path):
                    try:
                        self._observer.schedule(handler, path, recursive=True)
                        watched.append(path)
                    except Exception as e:
                        print(f'[!] Cannot watch {path}: {e}')
            if watched:
                self._observer.daemon = True
                self._observer.start()
                self._running = True
                print(f'[+] File watcher active on: {', '.join(watched)}')
            else:
                print('[!] No directories available to watch')

    def stop(self):
        with self._lock:
            if not self._running:
                return
            try:
                self._observer.stop()
                self._observer.join(timeout=3)
            except Exception:
                pass
            self._observer = None
            self._running = False
            print('[+] File watcher stopped')

    @property
    def is_running(self):
        return self._running

    def get_watched_paths(self):
        return self.watch_paths if self._running else []

class ForensicScanner:
    """
    Comprehensive forensic scanner for ransomware infection analysis.
    
    Scans running processes for suspicious names, searches the filesystem
    for encrypted files and ransom notes, and calculates an overall
    threat level based on combined indicators.
    """
    SUSPICIOUS_KEYWORDS = ['ransom', 'crypt', 'lock', 'encrypt', 'decrypt', 'locker', 'wanna', 'petya', 'cerber', 'locky', 'ryuk', 'maze', 'revil', 'darkside', 'conti', 'wannacry', 'wcry', 'tasksche', 'qbot', 'trickbot', 'emotet', 'cobalt', 'strike', 'mimikatz', 'powershell_ise', 'vssadmin', 'wbadmin', 'bcdedit', 'shadowcopy', 'delete']
    SYSTEM_PROCESS_WHITELIST = ['kworker', 'kthread', 'ksoftirqd', 'migration', 'kdevtmpfs', 'ecryptfs', 'lockdown', 'lockd', 'nfsd', 'jffs2gcd', 'kblockd', 'kswapd', 'kthrotld', 'kdm', 'kdmflush', 'crypto', 'algif', 'crypsetup', 'cryptsetup', 'systemd', 'systemd-', 'dbus', 'dbus-daemon', 'kernel', 'kernel_task', 'kthread', '[', 'launchd', 'init', 'udev', 'udevd', 'systemd-resolved', 'secure', 'sleep', 'nfsd', 'rpc', 'irq', 'flush', 'kworker', 'waiter', 'writer', 'watchdog', 'migration', 'idle', 'tpm', 'thermal', 'btrfs', 'xfs_', 'ext4', 'f2fs', 'jbd2', 'dm', 'dm-', 'md', 'vhost', 'floppy', 'usb']
    ENCRYPTED_EXTENSIONS = ['.locked', '.encrypted', '.crypt', '.enc', '.crypted', '.locky', '.cerber', '.zepto', '.thor', '.zzzzz', '.micro', '.aaa', '.xyz', '.zzz', '.odin', '.wallet', '.dharma', '.onion', '.rekt', '.coded', '.wnry', '.wcry', '.pet', '.petya', '.ryuk', '.krypt', '.kcry', '.crab', '.bip', '.liqe', '.nbes', '.ghas', '.clop', '.ciop', '.mole']
    RANSOM_NOTE_KEYWORDS = ['README', 'DECRYPT', 'RANSOM', 'HOW_TO', 'RECOVER', 'RESTORE', 'INSTRUCTION', 'PAYMENT', 'BITCOIN', 'YOUR_FILES', 'HELP_DECRYPT', 'READ_ME', 'files_encrypted', 'restore_files', 'decrypt_files', '@Please_Read_Me@', 'YOUR_FILES_ARE_ENCRYPTED']
    SCAN_PATHS = ['/tmp/ransomware_target', os.path.expanduser('~/Documents'), os.path.expanduser('~/Desktop'), os.path.expanduser('~/Pictures')]

    @staticmethod
    def detect_external_devices():
        """
        Detect mounted USB / external storage devices.
        Scans /media/ and /run/media/ for mounted pen drives.
        Returns list of mount paths found.
        """
        usb_paths = []
        for base in ['/media', '/run/media']:
            if not os.path.exists(base):
                continue
            for user_dir in os.listdir(base):
                user_path = os.path.join(base, user_dir)
                if os.path.isdir(user_path):
                    for device in os.listdir(user_path):
                        dev_path = os.path.join(user_path, device)
                        if os.path.isdir(dev_path):
                            usb_paths.append(dev_path)
                    if os.path.ismount(user_path):
                        usb_paths.append(user_path)
        return usb_paths

    def __init__(self):
        """Initialize the forensic scanner with empty result containers."""
        self.suspicious_processes = []
        self.encrypted_files = []
        self.ransom_notes = []
        self.threat_score = 0
        self.scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.hostname = socket.gethostname()
        self.detected_devices = []
        usb_devices = self.detect_external_devices()
        if usb_devices:
            self.detected_devices = usb_devices
            for dev_path in usb_devices:
                if dev_path not in self.SCAN_PATHS:
                    self.SCAN_PATHS.append(dev_path)
            print(f'[✓] External devices detected: {len(usb_devices)}')
            for d in usb_devices:
                print(f'    📱 {d}')

    def scan_processes(self):
        """
        Scan all running processes for suspicious names.
        
        Iterates through all system processes and checks if any
        process name contains keywords associated with ransomware.
        Filters out known system processes to reduce false positives.
        
        Returns:
            list: List of dictionaries with suspicious process details
        """
        print('\n[*] 🔍 Scanning running processes for suspicious activity...')
        suspicious = []
        total_scanned = 0
        for proc in psutil.process_iter(['pid', 'name', 'username', 'create_time', 'cmdline']):
            try:
                total_scanned += 1
                pinfo = proc.info
                process_name = (pinfo.get('name', '') or '').lower()
                is_system_proc = any((whitelist_entry in process_name for whitelist_entry in self.SYSTEM_PROCESS_WHITELIST))
                if is_system_proc:
                    continue
                for keyword in self.SUSPICIOUS_KEYWORDS:
                    if keyword in process_name:
                        cmdline = pinfo.get('cmdline', []) or []
                        cmdline_str = ' '.join(cmdline) if cmdline else 'N/A'
                        create_time = pinfo.get('create_time', 0)
                        started = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S') if create_time else 'Unknown'
                        suspicious.append({'pid': pinfo['pid'], 'name': pinfo.get('name', 'Unknown'), 'user': pinfo.get('username', 'Unknown'), 'started': started, 'cmdline': cmdline_str[:200], 'matched_keyword': keyword})
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        self.suspicious_processes = suspicious
        self.threat_score += len(suspicious) * 2
        print(f'[✓] Scanned {total_scanned} processes')
        if suspicious:
            print(f'[!] ⚠️  Found {len(suspicious)} suspicious processes!')
            for proc in suspicious:
                print(f'    🦠 PID {proc['pid']}: {proc['name']} (matched: {proc['matched_keyword']})')
        else:
            print(f'[✓] No suspicious processes detected')
        return suspicious

    def scan_filesystem(self):
        """
        Scan filesystem for encrypted files and ransom notes.
        
        Searches predefined directories for:
        1. Files with ransomware-associated extensions
        2. Files with ransomware-associated note names
        
        Returns:
            tuple: (encrypted_files_list, ransom_notes_list)
        """
        print('\n[*] 📁 Scanning filesystem for ransomware artifacts...')
        encrypted_files = []
        ransom_notes = []
        dirs_scanned = 0
        files_scanned = 0
        for scan_path in self.SCAN_PATHS:
            if not os.path.exists(scan_path):
                print(f'[*] Skipping (not found): {scan_path}')
                continue
            print(f'[*] Scanning: {scan_path}')
            for root, dirs, files in os.walk(scan_path):
                dirs_scanned += 1
                for filename in files:
                    files_scanned += 1
                    filepath = os.path.join(root, filename)
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in self.ENCRYPTED_EXTENSIONS:
                        try:
                            file_stat = os.stat(filepath)
                            encrypted_files.append({'path': filepath, 'size_mb': round(file_stat.st_size / (1024 * 1024), 4), 'modified': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'), 'extension': ext})
                        except OSError:
                            continue
                    filename_upper = filename.upper()
                    for keyword in self.RANSOM_NOTE_KEYWORDS:
                        if keyword in filename_upper:
                            ransom_notes.append({'path': filepath, 'filename': filename, 'matched_keyword': keyword})
                            break
        self.encrypted_files = encrypted_files
        self.ransom_notes = ransom_notes
        self.threat_score += len(encrypted_files)
        self.threat_score += len(ransom_notes) * 3
        print(f'\n[✓] Filesystem scan complete:')
        print(f'    Directories scanned:  {dirs_scanned}')
        print(f'    Files scanned:        {files_scanned}')
        if encrypted_files:
            total_size = sum((f['size_mb'] for f in encrypted_files))
            print(f'    🔒 Encrypted files:   {len(encrypted_files)} ({total_size:.2f} MB)')
        else:
            print(f'    ✅ No encrypted files found')
        if ransom_notes:
            print(f'    📝 Ransom notes:      {len(ransom_notes)}')
            for note in ransom_notes:
                print(f'       → {note['path']}')
        else:
            print(f'    ✅ No ransom notes found')
        return (encrypted_files, ransom_notes)

    def calculate_threat_level(self):
        """
        Calculate overall threat level based on scan results.
        
        Scoring system:
            - Each suspicious process: +2 points
            - Each encrypted file: +1 point
            - Each ransom note: +3 points
        
        Threat levels:
            CRITICAL: score >= 5
            HIGH:     score >= 3
            MEDIUM:   score >= 1
            CLEAN:    score == 0
        
        Returns:
            str: Threat level string (CRITICAL, HIGH, MEDIUM, or CLEAN)
        """
        if self.threat_score >= 5:
            return 'CRITICAL'
        elif self.threat_score >= 3:
            return 'HIGH'
        elif self.threat_score >= 1:
            return 'MEDIUM'
        else:
            return 'CLEAN'

    def determine_attack_vector(self):
        """
        Determine the most likely attack vector based on indicators.
        
        Returns:
            str: Description of the probable attack vector
        """
        if self.encrypted_files and self.suspicious_processes:
            return 'Active ransomware encryption detected via suspicious process'
        elif self.encrypted_files and self.ransom_notes:
            return 'Post-encryption ransomware attack (encryption complete)'
        elif self.encrypted_files:
            return 'File encryption detected (unknown source)'
        elif self.suspicious_processes:
            return 'Suspicious process activity (pre-encryption phase)'
        elif self.ransom_notes:
            return 'Ransom note artifacts detected'
        else:
            return 'No attack indicators found'

    def generate_report(self):
        """
        Generate comprehensive forensic report and save as JSON.
        
        Creates a timestamped JSON report in the reports/ directory
        containing all scan findings, threat analysis, and metadata.
        
        Returns:
            str: Path to the generated report file
        """
        threat_level = self.calculate_threat_level()
        attack_vector = self.determine_attack_vector()
        total_damage_mb = sum((f['size_mb'] for f in self.encrypted_files))
        report = {'hostname': self.hostname, 'scan_time': self.scan_time, 'total_processes': len(psutil.pids()), 'threat_level': threat_level, 'threat_score': self.threat_score, 'attack_vector': attack_vector, 'estimated_damage_mb': round(total_damage_mb, 2), 'suspicious_processes': self.suspicious_processes, 'encrypted_files_found': self.encrypted_files, 'ransom_notes_found': self.ransom_notes, 'scan_paths': self.SCAN_PATHS, 'summary': {'suspicious_process_count': len(self.suspicious_processes), 'encrypted_file_count': len(self.encrypted_files), 'ransom_note_count': len(self.ransom_notes)}}
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f'forensic_scan_{timestamp}.json'
        report_path = os.path.join(reports_dir, report_filename)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f'\n[✓] Report saved: {report_path}')
        return report_path

def _path_allowed(filepath, allowed_paths):
    try:
        real = os.path.realpath(os.path.abspath(filepath))
        for base in allowed_paths:
            if not base:
                continue
            base_real = os.path.realpath(os.path.abspath(base))
            if real == base_real or real.startswith(base_real + os.sep):
                return True
        return False
    except (OSError, ValueError):
        return False

class ProcessScanner(threading.Thread):
    """Background thread that scans processes and runs ML prediction."""

    def __init__(self, threat_analyzer, allowed_paths=None):
        super().__init__(daemon=True)
        self.analyzer = threat_analyzer
        self.allowed_paths = allowed_paths or ALLOWED_PATHS
        self._stop_event = threading.Event()
        self._scanned_pids = set()
        self._scan_count = 0

    def run(self):
        print(f'[+] Process scanner started (interval: {SCAN_INTERVAL}s)')
        while not self._stop_event.is_set():
            try:
                self._scan_cycle()
            except Exception as e:
                print(f'[!] Scan error: {e}')
            for _ in range(int(SCAN_INTERVAL * 10)):
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)

    def _scan_cycle(self):
        self._scan_count += 1
        current_pids = set()
        for proc in psutil.process_iter():
            try:
                features = get_process_features(proc)
                if not features:
                    continue
                pid = features['pid']
                current_pids.add(pid)
                if features.get('entropy', 0) >= ENTROPY_SPIKE_THRESHOLD:
                    self.analyzer.record_entropy_spike(pid, features['entropy'])
                if features.get('cpu', 0) >= CPU_SPIKE_THRESHOLD:
                    self.analyzer.record_cpu_spike(pid, features['cpu'])
                status, confidence = predict(features, model_type='decision_tree')
                if status == 'Malicious' and confidence >= MALICIOUS_CONFIDENCE:
                    self._handle_malicious(proc, features, confidence)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception:
                continue
        self._scanned_pids = current_pids

    def _handle_malicious(self, proc, features, confidence):
        pid = features['pid']
        name = features.get('name', 'unknown')
        try:
            exe = proc.exe()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return
        if not exe or not _path_allowed(exe, self.allowed_paths):
            return
        try:
            proc.suspend()
            time.sleep(0.1)
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return
        alert = {'id': f'PS-{int(time.time() * 1000)}', 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'type': 'MALICIOUS_PROCESS', 'severity': 'HIGH', 'reasons': [f'ML detection: {confidence:.1f}% confidence'], 'terminated_processes': [{'pid': pid, 'name': name}], 'quarantined_files': [], 'quarantine_dir': None, 'affected_file_count': 0, 'process_details': {'cpu': features.get('cpu', 0), 'memory': features.get('memory', 0), 'entropy': features.get('entropy', 0), 'threads': features.get('threads', 0), 'open_files': features.get('open_files', 0)}}
        self.analyzer.alerts.append(alert)
        self.analyzer.threats_blocked += 1
        print(f'[ALERT] Terminated PID {pid} ({name}) -- {confidence:.1f}% malicious')

    def stop(self):
        self._stop_event.set()

    @property
    def is_running(self):
        return self.is_alive() and (not self._stop_event.is_set())

class RealtimeEngine:
    """
    Main orchestrator for real-time monitoring.
    Manages the process scanner thread and file watcher.
    """

    def __init__(self):
        self.analyzer = ThreatAnalyzer(allowed_paths=ALLOWED_PATHS)
        self._process_scanner = None
        self._file_watcher = None
        self._active = False
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._active:
                return {'status': 'already_running'}
            self.analyzer.start_time = time.time()
            self._process_scanner = ProcessScanner(self.analyzer, allowed_paths=ALLOWED_PATHS)
            self._process_scanner.start()
            self._file_watcher = FileWatcher(self.analyzer, watch_paths=[p for p in ALLOWED_PATHS if os.path.isdir(p)])
            self._file_watcher.start()
            self._active = True
            print(f'\n[+] Real-time monitoring activated')
            print(f'    Process scanning every {SCAN_INTERVAL}s')
            print(f'    Filesystem watching active\n')
            return {'status': 'started'}

    def stop(self):
        with self._lock:
            if not self._active:
                return {'status': 'already_stopped'}
            if self._process_scanner:
                self._process_scanner.stop()
                self._process_scanner = None
            if self._file_watcher:
                self._file_watcher.stop()
                self._file_watcher = None
            self._active = False
            print('[+] Real-time monitoring deactivated\n')
            return {'status': 'stopped'}

    def get_status(self):
        stats = self.analyzer.get_stats()
        window = self.analyzer.get_window_stats()
        recent_alerts = self.analyzer.get_recent_alerts(limit=20)
        return {'active': self._active, 'process_scanner': {'running': bool(self._process_scanner and self._process_scanner.is_running), 'scan_count': self._process_scanner._scan_count if self._process_scanner else 0}, 'file_watcher': {'running': bool(self._file_watcher and self._file_watcher.is_running), 'watched_paths': self._file_watcher.get_watched_paths() if self._file_watcher else []}, 'stats': stats, 'window': window, 'recent_alerts': recent_alerts}

    @property
    def is_active(self):
        return self._active

class ForensicReportPDF(FPDF):
    """
    Black-and-white forensic ransomware report (no colour).
    Includes Detection Results table from ML scan.
    """

    def __init__(self, report_data, detection_results=None):
        super().__init__()
        self.report_data = report_data
        self.detection_results = detection_results or []
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        """Black bar, white text (compact for 1 page)."""
        self.set_fill_color(0, 0, 0)
        self.rect(0, 0, 210, 18, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 12)
        self.set_y(4)
        self.cell(0, 10, 'RANSOMWARE FORENSIC REPORT', align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_text_color(0, 0, 0)
        self.ln(5)

    def footer(self):
        self.set_y(-18)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, f'Page {self.page_no()}/{{nb}} | {datetime.now().strftime('%Y-%m-%d %H:%M')} | Ransomware Detection System', align='C')

    def add_threat_level_banner(self, threat_level=None):
        """Threat level + short meaning (B&W). Uses report threat level (not name-only false HIGH)."""
        if threat_level is None:
            threat_level = self.report_data.get('threat_level', 'UNKNOWN')
        meaning = {'CLEAN': 'No ransomware indicators detected.', 'MEDIUM': 'Some suspicious indicators; review recommended.', 'HIGH': 'Significant ransomware indicators detected; investigation recommended.', 'CRITICAL': 'Severe indicators; immediate response recommended.'}.get(threat_level, 'Threat assessment from scan.')
        self.set_font('Helvetica', 'B', 11)
        self.set_fill_color(220, 220, 220)
        self.set_text_color(0, 0, 0)
        self.cell(0, 7, f'  THREAT LEVEL: {threat_level}  ({meaning})', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')
        self.ln(3)

    def add_section_header(self, title):
        """Black text, light grey bar (B&W)."""
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0, 0, 0)
        self.cell(0, 6, f'  {title}', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def add_info_row(self, label, value):
        """
        Add a label-value information row.
        
        Args:
            label (str): Field label
            value: Field value (will be converted to string)
        """
        self.set_font('Helvetica', 'B', 10)
        self.cell(60, 7, f'  {label}:', new_x='END')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 7, str(value), new_x='LMARGIN', new_y='NEXT')

    def truncate_path(self, path, max_length=60):
        """
        Truncate long file paths for display.
        
        Args:
            path (str): File path to truncate
            max_length (int): Maximum display length
            
        Returns:
            str: Truncated path string
        """
        if len(str(path)) <= max_length:
            return str(path)
        return '...' + str(path)[-(max_length - 3):]

    def build_report(self):
        """Full multi-page report: threat level, ML detection results, and full forensic scan details."""
        self.alias_nb_pages()
        data = self.report_data
        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()
        summary = data.get('summary', {})
        ef = summary.get('encrypted_file_count', 0)
        rn = summary.get('ransom_note_count', 0)
        raw_level = data.get('threat_level', 'UNKNOWN')
        if ef == 0 and rn == 0:
            report_threat_level = 'CLEAN'
        else:
            report_threat_level = raw_level
        self.add_threat_level_banner(threat_level=report_threat_level)
        sp = summary.get('suspicious_process_count', 0)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(0, 0, 0)
        self.cell(0, 5, f"Hostname: {data.get('hostname', 'N/A')}  |  Scan: {data.get('scan_time', 'N/A')}  |  Suspicious: {sp}  Encrypted files: {ef}  Ransom notes: {rn}", new_x='LMARGIN', new_y='NEXT')
        self.ln(3)
        
        # --- ML DETECTION RESULTS ---
        detection = self.detection_results
        self.add_section_header(f"ML BEHAVIORAL DETECTIONS ({len(detection)} processes)")
        if detection:
            self.set_font('Helvetica', 'B', 7)
            self.set_fill_color(220, 220, 220)
            self.cell(40, 5, 'Process', border=1, fill=True)
            self.cell(15, 5, 'PID', border=1, fill=True, align='C')
            self.cell(80, 5, 'Activity Details', border=1, fill=True)
            self.cell(20, 5, 'Status', border=1, fill=True, align='C')
            self.cell(0, 5, 'Conf.%', border=1, fill=True, align='C', new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 7)
            for i, row in enumerate(detection):
                bg = (245, 245, 245) if i % 2 == 0 else (255, 255, 255)
                self.set_fill_color(*bg)
                self.cell(40, 5, str(row.get('name', ''))[:25], border=1, fill=True)
                self.cell(15, 5, str(row.get('pid', '')), border=1, fill=True, align='C')
                self.cell(80, 5, self.truncate_path(str(row.get('activity', '')), 60), border=1, fill=True)
                self.cell(20, 5, str(row.get('status', '')), border=1, fill=True, align='C')
                self.cell(0, 5, str(row.get('confidence', '')) + '%', border=1, fill=True, align='C', new_x='LMARGIN', new_y='NEXT')
        else:
            self.set_font('Helvetica', 'I', 9)
            self.cell(0, 6, '  No real-time processes analysed in this scan.', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)

        # --- FORENSIC ENCRYPTED FILES ---
        enc_files = data.get('encrypted_files', [])
        self.add_section_header(f"ENCRYPTED FILES FOUND ({len(enc_files)})")
        if enc_files:
            self.set_font('Helvetica', 'B', 7)
            self.set_fill_color(220, 220, 220)
            self.cell(110, 5, 'File Path', border=1, fill=True)
            self.cell(20, 5, 'Extension', border=1, fill=True, align='C')
            self.cell(20, 5, 'Size', border=1, fill=True, align='C')
            self.cell(0, 5, 'Modified (UTC)', border=1, fill=True, align='C', new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 7)
            for i, f in enumerate(enc_files):
                bg = (245, 245, 245) if i % 2 == 0 else (255, 255, 255)
                self.set_fill_color(*bg)
                self.cell(110, 5, self.truncate_path(f.get('path', ''), 75), border=1, fill=True)
                self.cell(20, 5, str(f.get('extension', '')), border=1, fill=True, align='C')
                self.cell(20, 5, str(f.get('size_mb', '')), border=1, fill=True, align='C')
                self.cell(0, 5, str(f.get('modified', '')), border=1, fill=True, align='C', new_x='LMARGIN', new_y='NEXT')
        else:
            self.set_font('Helvetica', 'I', 9)
            self.cell(0, 6, '  No encrypted files found.', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)

        # --- RANSOM NOTES ---
        notes = data.get('ransom_notes', [])
        self.add_section_header(f"RANSOM NOTES DETECTED ({len(notes)})")
        if notes:
            self.set_font('Helvetica', 'B', 7)
            self.set_fill_color(220, 220, 220)
            self.cell(110, 5, 'File Path', border=1, fill=True)
            self.cell(40, 5, 'Filename', border=1, fill=True)
            self.cell(0, 5, 'Match Indicator', border=1, fill=True, align='C', new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 7)
            for i, n in enumerate(notes):
                bg = (245, 245, 245) if i % 2 == 0 else (255, 255, 255)
                self.set_fill_color(*bg)
                self.cell(110, 5, self.truncate_path(n.get('path', ''), 75), border=1, fill=True)
                self.cell(40, 5, str(n.get('filename', '')), border=1, fill=True)
                self.cell(0, 5, str(n.get('matched_keyword', '')), border=1, fill=True, align='C', new_x='LMARGIN', new_y='NEXT')
        else:
            self.set_font('Helvetica', 'I', 9)
            self.cell(0, 6, '  No ransom notes found.', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)

        # --- SUSPICIOUS PROCESSES ---
        procs = data.get('suspicious_processes', [])
        self.add_section_header(f"SUSPICIOUS PROCESSES IDENTIFIED BY FORENSICS ({len(procs)})")
        if procs:
            self.set_font('Helvetica', 'B', 7)
            self.set_fill_color(220, 220, 220)
            self.cell(15, 5, 'PID', border=1, fill=True, align='C')
            self.cell(35, 5, 'Name', border=1, fill=True)
            self.cell(25, 5, 'User', border=1, fill=True)
            self.cell(45, 5, 'Match Reason', border=1, fill=True)
            self.cell(0, 5, 'Command Line', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 6)
            for i, p in enumerate(procs):
                bg = (245, 245, 245) if i % 2 == 0 else (255, 255, 255)
                self.set_fill_color(*bg)
                self.cell(15, 5, str(p.get('pid', '')), border=1, fill=True, align='C')
                self.cell(35, 5, str(p.get('name', ''))[:20], border=1, fill=True)
                self.cell(25, 5, str(p.get('user', ''))[:15], border=1, fill=True)
                self.cell(45, 5, str(p.get('matched_keyword', '')), border=1, fill=True)
                
                cmd = ' '.join(p.get('cmdline', []))
                self.cell(0, 5, self.truncate_path(cmd, 60), border=1, fill=True, new_x='LMARGIN', new_y='NEXT')
        else:
            self.set_font('Helvetica', 'I', 9)
            self.cell(0, 6, '  No forensic suspicious processes found.', new_x='LMARGIN', new_y='NEXT')
        
        self.ln(10)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 4, "Ransomware Detection System Forensic Document. This report was generated automatically via behavioural analysis and static filesystem forensics.")

def find_latest_report():
    """
    Find the most recent forensic scan JSON report.
    
    Returns:
        str: Path to the latest report, or None if no reports found
    """
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    pattern = os.path.join(reports_dir, 'forensic_scan_*.json')
    reports = glob.glob(pattern)
    if not reports:
        return None
    return max(reports, key=os.path.getmtime)

def generate_report(json_path=None, detection_results=None):
    """
    Generate a black-and-white PDF report from forensic scan JSON.
    Optionally include Detection Results table (from ML scan).
    
    Args:
        json_path (str): Path to JSON report file, or None for latest
        detection_results (list): Optional list of {name, pid, activity, status, confidence} for Detection Results table
        
    Returns:
        str: Path to generated PDF file
    """
    if json_path is None:
        json_path = find_latest_report()
        if json_path is None:
            print('[ERROR] No forensic scan reports found in reports/ directory.')
            print('[*] Run forensic_scanner.py first to generate a scan report.')
            return None
    if not os.path.exists(json_path):
        print(f'[ERROR] Report file not found: {json_path}')
        return None
    print(f'\n{'=' * 60}')
    print(f'  PDF REPORT GENERATOR (B&W)')
    print(f'{'=' * 60}')
    print(f'[*] Loading report: {json_path}')
    try:
        with open(json_path, 'r') as f:
            report_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f'[ERROR] Failed to parse JSON: {e}')
        return None
    print('[*] Generating PDF report...')
    pdf = ForensicReportPDF(report_data, detection_results=detection_results or [])
    pdf.build_report()
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_filename = f'forensic_report_{timestamp}.pdf'
    pdf_path = os.path.join(reports_dir, pdf_filename)
    pdf.output(pdf_path)
    print(f'[✓] PDF report generated successfully!')
    print(f'[✓] Saved to: {pdf_path}')
    print(f'[✓] File size: {os.path.getsize(pdf_path) / 1024:.1f} KB')
    return pdf_path

class RansomwareDecryptor:
    """Decrypts files encrypted by the ransomware simulator."""
    TARGET_DIR = '/tmp/ransomware_target'
    KEY_FILE = '/tmp/ransomware_key.secret'

    def __init__(self):
        self.files_decrypted = 0
        self.total_size = 0
        self.notes_removed = 0
        self.fernet = None

    def load_key(self):
        """Load encryption key from the key file."""
        if not os.path.exists(self.KEY_FILE):
            print(f'[ERROR] Key file not found: {self.KEY_FILE}')
            print(f'[!] Cannot decrypt without the encryption key.')
            print(f'[!] Run advanced_ransomware.py first to create test files.')
            return False
        with open(self.KEY_FILE, 'rb') as f:
            key = f.read()
        self.fernet = Fernet(key)
        print(f'[✓] Encryption key loaded from: {self.KEY_FILE}')
        return True

    def decrypt_file(self, filepath):
        """
        Decrypt a single .locked file and restore original name.
        
        Args:
            filepath (str): Path to the .locked file
            
        Returns:
            tuple: (restored_path, size) or (None, 0) on failure
        """
        try:
            with open(filepath, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = self.fernet.decrypt(encrypted_data)
            original_path = filepath.rsplit('.locked', 1)[0]
            with open(original_path, 'wb') as f:
                f.write(decrypted_data)
            os.remove(filepath)
            self.files_decrypted += 1
            self.total_size += len(decrypted_data)
            return (original_path, len(decrypted_data))
        except Exception as e:
            print(f'  [!] Failed to decrypt {filepath}: {e}')
            return (None, 0)

    def remove_ransom_notes(self):
        """Remove all ransom notes from the target directory."""
        for root, dirs, files in os.walk(self.TARGET_DIR):
            for filename in files:
                if 'RANSOM' in filename.upper():
                    filepath = os.path.join(root, filename)
                    try:
                        os.remove(filepath)
                        self.notes_removed += 1
                        print(f'  🗑️  Removed: {filepath}')
                    except OSError as e:
                        print(f'  [!] Failed to remove {filepath}: {e}')

    def decrypt_all(self):
        """
        Decrypt all .locked files in the target directory.
        Walks the directory tree and decrypts every .locked file found.
        """
        print(f'\n{'=' * 60}')
        print(f'  🔓 RANSOMWARE DECRYPTOR - File Recovery Tool')
        print(f'{'=' * 60}')
        if not os.path.exists(self.TARGET_DIR):
            print(f'[ERROR] Target directory not found: {self.TARGET_DIR}')
            print(f'[!] No encrypted files to recover.')
            return
        if not self.load_key():
            return
        locked_files = []
        for root, dirs, files in os.walk(self.TARGET_DIR):
            for filename in files:
                if filename.endswith('.locked'):
                    locked_files.append(os.path.join(root, filename))
        if not locked_files:
            print(f'[*] No .locked files found in {self.TARGET_DIR}')
            print(f'[✓] Directory appears to be already decrypted.')
            return
        print(f'[*] Found {len(locked_files)} encrypted files to recover')
        print(f'\n  🔓 Decrypting files...')
        start_time = time.time()
        for filepath in locked_files:
            restored_path, size = self.decrypt_file(filepath)
            if restored_path:
                print(f'  ✓ Restored: {os.path.basename(restored_path)} ({size / 1024:.1f} KB)')
        elapsed = time.time() - start_time
        print(f'\n  🗑️  Removing ransom notes...')
        self.remove_ransom_notes()
        print(f'\n{'=' * 60}')
        print(f'  📊 RECOVERY SUMMARY')
        print(f'{'=' * 60}')
        print(f'  ✅ Files Recovered:    {self.files_decrypted}')
        print(f'  💾 Data Restored:      {self.total_size / (1024 * 1024):.2f} MB')
        print(f'  🗑️  Notes Removed:     {self.notes_removed}')
        print(f'  ⏱️  Time:              {elapsed:.3f}s')
        print(f'{'=' * 60}')
        print(f'\n  [✓] All files recovered successfully! ✓')

def is_path_allowed(filepath, allowed_paths):
    """Check if filepath falls under one of the allowed base paths."""
    try:
        real = os.path.realpath(os.path.abspath(filepath))
        for protected in PROTECTED_DIRS:
            if real == protected or real.startswith(protected + os.sep):
                return False
        for base in allowed_paths:
            if not base:
                continue
            base_real = os.path.realpath(os.path.abspath(base))
            if real == base_real or real.startswith(base_real + os.sep):
                return True
        return False
    except (OSError, ValueError):
        return False

def is_inside_quarantine(filepath):
    """Check if filepath is inside the quarantine directory."""
    try:
        real = os.path.realpath(os.path.abspath(filepath))
        q_real = os.path.realpath(os.path.abspath(QUARANTINE_DIR))
        return real.startswith(q_real + os.sep)
    except (OSError, ValueError):
        return False

def log_action(action_type, details):
    """Append an action entry to detection_log.json."""
    log_path = os.path.join(LOG_DIR, 'detection_log.json')
    entry = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'action': action_type, **details}
    logs = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                logs = json.load(f)
            if not isinstance(logs, list):
                logs = [logs]
        except Exception:
            logs = []
    logs.append(entry)
    logs = logs[-500:]
    try:
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception:
        pass

def suspend_and_terminate(pid, allowed_paths):
    """
    Suspend then terminate a process by PID.
    Only acts if the process executable is under allowed paths.
    Returns a dict with the result.
    """
    try:
        proc = psutil.Process(pid)
        name = proc.name()
        try:
            exe = proc.exe()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return {'success': False, 'pid': pid, 'reason': 'cannot read exe path'}
        if not exe:
            return {'success': False, 'pid': pid, 'reason': 'empty exe path'}
        if not is_path_allowed(exe, allowed_paths):
            return {'success': False, 'pid': pid, 'reason': f'exe not in allowed paths: {exe}'}
        proc.suspend()
        time.sleep(0.2)
        proc.terminate()
        log_action('PROCESS_TERMINATED', {'pid': pid, 'name': name, 'exe': exe})
        return {'success': True, 'pid': pid, 'name': name, 'exe': exe}
    except psutil.NoSuchProcess:
        return {'success': False, 'pid': pid, 'reason': 'process no longer exists'}
    except psutil.AccessDenied:
        return {'success': False, 'pid': pid, 'reason': 'access denied'}
    except psutil.ZombieProcess:
        return {'success': False, 'pid': pid, 'reason': 'zombie process'}
    except Exception as e:
        return {'success': False, 'pid': pid, 'reason': str(e)}

def quarantine_file(filepath, allowed_paths, run_dir=None):
    """
    Move a file into quarantine. Only works for files
    under allowed paths. Returns result dict.
    """
    if not filepath or not os.path.isfile(filepath):
        return {'success': False, 'path': filepath, 'reason': 'file not found'}
    if not is_path_allowed(filepath, allowed_paths):
        return {'success': False, 'path': filepath, 'reason': 'path not in allowed list'}
    if run_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(QUARANTINE_DIR, ts)
    os.makedirs(run_dir, exist_ok=True)
    try:
        dest = os.path.join(run_dir, os.path.basename(filepath))
        counter = 0
        while os.path.exists(dest):
            counter += 1
            base, ext = os.path.splitext(os.path.basename(filepath))
            dest = os.path.join(run_dir, f'{base}_{counter}{ext}')
        shutil.move(filepath, dest)
        log_action('FILE_QUARANTINED', {'original_path': filepath, 'quarantine_path': dest})
        return {'success': True, 'path': filepath, 'quarantine_path': dest}
    except Exception as e:
        return {'success': False, 'path': filepath, 'reason': str(e)}

def quarantine_executable(pid, allowed_paths, run_dir=None):
    """
    Get the executable path from a process and quarantine it.
    The process should already be terminated before calling this.
    """
    try:
        exe = None
        try:
            proc = psutil.Process(pid)
            exe = proc.exe()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {'success': False, 'pid': pid, 'reason': 'cannot access process'}
        if not exe or not os.path.isfile(exe):
            return {'success': False, 'pid': pid, 'reason': 'exe not found on disk'}
        return quarantine_file(exe, allowed_paths, run_dir)
    except Exception as e:
        return {'success': False, 'pid': pid, 'reason': str(e)}

def list_quarantined():
    """List all quarantined file batches and their contents."""
    result = []
    if not os.path.isdir(QUARANTINE_DIR):
        return result
    for batch_name in sorted(os.listdir(QUARANTINE_DIR), reverse=True):
        batch_path = os.path.join(QUARANTINE_DIR, batch_name)
        if not os.path.isdir(batch_path):
            continue
        files = []
        try:
            for fname in os.listdir(batch_path):
                fpath = os.path.join(batch_path, fname)
                if os.path.isfile(fpath):
                    files.append({'name': fname, 'path': fpath, 'size_bytes': os.path.getsize(fpath)})
        except Exception:
            pass
        if files:
            result.append({'batch': batch_name, 'batch_path': batch_path, 'file_count': len(files), 'files': files})
    return result

def secure_delete_batch(batch_name, confirmed=False):
    """
    Permanently delete a quarantine batch folder.
    Requires confirmed=True as a safety check.
    Only deletes from inside the quarantine directory.
    """
    if not confirmed:
        return {'success': False, 'reason': 'deletion not confirmed'}
    batch_path = os.path.join(QUARANTINE_DIR, batch_name)
    if not is_inside_quarantine(batch_path):
        return {'success': False, 'reason': 'path is not inside quarantine directory'}
    if not os.path.isdir(batch_path):
        return {'success': False, 'reason': 'batch not found'}
    real = os.path.realpath(batch_path)
    for protected in PROTECTED_DIRS:
        if real == protected or real.startswith(protected + os.sep):
            return {'success': False, 'reason': 'cannot delete system directory'}
    try:
        file_count = len([f for f in os.listdir(batch_path) if os.path.isfile(os.path.join(batch_path, f))])
        shutil.rmtree(batch_path)
        log_action('QUARANTINE_DELETED', {'batch': batch_name, 'batch_path': batch_path, 'files_deleted': file_count})
        return {'success': True, 'batch': batch_name, 'files_deleted': file_count}
    except Exception as e:
        return {'success': False, 'reason': str(e)}

def secure_delete_file(filepath, confirmed=False):
    """
    Permanently delete a single quarantined file.
    Only works on files inside the quarantine directory.
    """
    if not confirmed:
        return {'success': False, 'reason': 'deletion not confirmed'}
    if not is_inside_quarantine(filepath):
        return {'success': False, 'reason': 'file is not inside quarantine directory'}
    if not os.path.isfile(filepath):
        return {'success': False, 'reason': 'file not found'}
    real = os.path.realpath(filepath)
    for protected in PROTECTED_DIRS:
        if real.startswith(protected + os.sep):
            return {'success': False, 'reason': 'cannot delete system file'}
    try:
        os.remove(filepath)
        log_action('FILE_DELETED', {'path': filepath})
        return {'success': True, 'path': filepath}
    except Exception as e:
        return {'success': False, 'reason': str(e)}

def get_allowed_paths():
    """Gets the dynamically determined allowed paths for scanning and clearing."""
    paths = ALLOWED_CLEAR_PATHS.copy()
    try:
        from forensic_scanner import ForensicScanner
        scanner = ForensicScanner(paths)
        scanner.detect_external_devices()
        return scanner.scan_paths
    except Exception:
        return paths

def start_pure_python_monitor():
    """Initializes and starts the background pure python monitor."""
    global pure_python_monitor
    try:
        paths_to_watch = get_allowed_paths()
        pure_python_monitor = DirectoryMonitor(watch_dirs=paths_to_watch)
        pure_python_monitor.start_background()
        print(f'[*] Pure Python Detector thread started watching: {paths_to_watch}')
    except Exception as e:
        print(f'[!] Failed to start pure python monitor: {e}')

def ensure_models_exist():
    """Auto-train models if they don't exist on startup."""
    dt_path = os.path.join(MODEL_DIR, 'decision_tree.pkl')
    mlp_path = os.path.join(MODEL_DIR, 'mlp_model.pkl')
    if not os.path.exists(dt_path) or not os.path.exists(mlp_path):
        print('[*] Models not found. Training automatically...')
        train_models()
        print('[✓] Models trained and ready.')
    else:
        print('[✓] ML models loaded successfully.')

def save_detection_log(results):
    """Save detection results to logs/detection_log.json."""
    log_path = os.path.join(LOG_DIR, 'detection_log.json')
    log_entry = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'total_processes': len(results), 'malicious_count': sum((1 for r in results if r.get('status') == 'Malicious')), 'benign_count': sum((1 for r in results if r.get('status') == 'Benign')), 'results': results}
    logs = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                logs = json.load(f)
            if not isinstance(logs, list):
                logs = [logs]
        except (json.JSONDecodeError, IOError):
            logs = []
    logs.append(log_entry)
    logs = logs[-100:]
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

def _path_allowed_for_clear(filepath):
    """Return True only if filepath is under one of ALLOWED_CLEAR_PATHS (realpath)."""
    try:
        real = os.path.realpath(os.path.abspath(filepath))
        for base in ALLOWED_CLEAR_PATHS:
            if not base:
                continue
            base_real = os.path.realpath(os.path.abspath(base))
            if real == base_real or real.startswith(base_real + os.sep):
                return True
        return False
    except (OSError, ValueError):
        return False

def clear_ransomware():
    """
    Run forensic scan, then terminate suspicious processes and quarantine
    detected ransomware files (only under allowed paths). Returns a result dict.
    Only terminates processes whose executable is under allowed paths (e.g. Desktop, /tmp)
    to avoid killing system processes (e.g. kernel threads with 'lock' in name).
    """
    import psutil
    scanner = ForensicScanner()
    scanner.scan_processes()
    scanner.scan_filesystem()
    terminated_pids = []
    quarantine_errors = []
    quarantined = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_quarantine_dir = os.path.join(QUARANTINE_DIR, timestamp)
    os.makedirs(run_quarantine_dir, exist_ok=True)
    for proc_info in scanner.suspicious_processes:
        pid = proc_info.get('pid')
        if not pid:
            continue
        try:
            p = psutil.Process(pid)
            try:
                exe = p.exe()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                exe = None
            if exe is None:
                continue
            if not _path_allowed_for_clear(exe):
                continue
            p.terminate()
            terminated_pids.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            quarantine_errors.append(f'PID {pid}: {e}')
    for item in scanner.encrypted_files:
        path = item.get('path') if isinstance(item, dict) else item
        if not path or not _path_allowed_for_clear(path):
            continue
        try:
            if os.path.isfile(path):
                dest = os.path.join(run_quarantine_dir, os.path.basename(path))
                if os.path.exists(dest):
                    dest = os.path.join(run_quarantine_dir, f'{os.path.basename(path)}_{len(quarantined)}')
                shutil.move(path, dest)
                quarantined.append(path)
        except Exception as e:
            quarantine_errors.append(f'{path}: {e}')
    for note in scanner.ransom_notes:
        path = note.get('path') if isinstance(note, dict) else note
        if not path or not _path_allowed_for_clear(path):
            continue
        try:
            if os.path.isfile(path):
                dest = os.path.join(run_quarantine_dir, os.path.basename(path))
                if os.path.exists(dest):
                    dest = os.path.join(run_quarantine_dir, f'{os.path.basename(path)}_{len(quarantined)}')
                shutil.move(path, dest)
                quarantined.append(path)
        except Exception as e:
            quarantine_errors.append(f'{path}: {e}')
    return {'terminated_pids': terminated_pids, 'quarantined_files': quarantined, 'quarantine_dir': run_quarantine_dir, 'errors': quarantine_errors, 'threat_level': scanner.calculate_threat_level()}

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/scan')
def scan():
    """
    Scan all running processes (ML classification) AND filesystem
    for ransomware artifacts. Returns unified JSON list of results.
    Encrypted files and ransom notes appear as Malicious detections.
    """
    global latest_results
    activities = get_process_activity()
    results = []
    for activity in activities:
        try:
            status, confidence = predict(activity, model_type='decision_tree')
            results.append({'name': activity.get('name', 'Unknown'), 'pid': activity.get('pid', 0), 'activity': f'Files:{activity.get('open_files', 0)} CPU:{activity.get('cpu', 0)}% MEM:{activity.get('memory', 0)}%', 'status': status, 'confidence': confidence, 'open_files': activity.get('open_files', 0), 'cpu': activity.get('cpu', 0), 'memory': activity.get('memory', 0), 'threads': activity.get('threads', 0), 'entropy': activity.get('entropy', 0)})
        except Exception:
            continue
    try:
        scanner = ForensicScanner()
        scanner.scan_filesystem()
        for f in scanner.encrypted_files:
            path = f.get('path', '') if isinstance(f, dict) else f
            ext = f.get('extension', '.locked') if isinstance(f, dict) else '.locked'
            size = f.get('size_mb', 0) if isinstance(f, dict) else 0
            results.append({'name': os.path.basename(path), 'pid': '-', 'activity': f'Encrypted ransomware file ({ext}) - {size} MB', 'status': 'Malicious', 'confidence': 99.0})
        for note in scanner.ransom_notes:
            path = note.get('path', '') if isinstance(note, dict) else note
            fname = note.get('filename', os.path.basename(path)) if isinstance(note, dict) else os.path.basename(path)
            keyword = note.get('matched_keyword', 'RANSOM') if isinstance(note, dict) else 'RANSOM'
            results.append({'name': fname, 'pid': '-', 'activity': f'Ransom note detected - keyword: {keyword}', 'status': 'Malicious', 'confidence': 100.0})
        if pure_python_monitor:
            pure_quarantines = pure_python_monitor.get_quarantined_files(clear=False)
            for qf in pure_quarantines:
                path = qf.get('original_path', '')
                fname = os.path.basename(path)
                results.append({'name': fname, 'pid': '-', 'activity': f'Background Monitor: High Entropy Quarantine', 'status': 'Malicious', 'confidence': 100.0})
    except Exception as e:
        print(f'[!] Forensic scan error during /scan: {e}')
    results.sort(key=lambda x: (x['status'] != 'Malicious', -x['confidence']))
    latest_results = results
    save_detection_log(results)
    return jsonify(results)

@app.route('/generate_report', methods=['GET', 'POST'])
def generate_report_route():
    """
    Generate a forensic PDF report and return it as a download.
    Runs a forensic scan first, then generates the PDF.
    """
    global latest_results
    try:
        scanner = ForensicScanner()
        scanner.scan_processes()
        scanner.scan_filesystem()
        report_path = scanner.generate_report()
        detection_results = []
        try:
            activities = get_process_activity()
            for activity in activities:
                try:
                    status, confidence = predict(activity, model_type='decision_tree')
                    detection_results.append({'name': activity.get('name', 'Unknown'), 'pid': activity.get('pid', 0), 'activity': f'Files:{activity.get('open_files', 0)} CPU:{activity.get('cpu', 0)}% MEM:{activity.get('memory', 0)}%', 'status': status, 'confidence': confidence})
                except Exception:
                    continue
        except Exception:
            pass
        for f in scanner.encrypted_files:
            path = f.get('path', '') if isinstance(f, dict) else f
            ext = f.get('extension', '.locked') if isinstance(f, dict) else '.locked'
            size = f.get('size_mb', 0) if isinstance(f, dict) else 0
            detection_results.append({'name': os.path.basename(path), 'pid': '-', 'activity': f'Encrypted file ({ext}) {size}MB', 'status': 'Malicious', 'confidence': 99.0})
        for note in scanner.ransom_notes:
            path = note.get('path', '') if isinstance(note, dict) else note
            fname = note.get('filename', os.path.basename(path)) if isinstance(note, dict) else os.path.basename(path)
            keyword = note.get('matched_keyword', 'RANSOM') if isinstance(note, dict) else 'RANSOM'
            detection_results.append({'name': fname, 'pid': '-', 'activity': f'Ransom note ({keyword})', 'status': 'Malicious', 'confidence': 100.0})
        if pure_python_monitor:
            pure_quarantines = pure_python_monitor.get_quarantined_files(clear=False)
            for qf in pure_quarantines:
                path = qf.get('original_path', '')
                fname = os.path.basename(path)
                detection_results.append({'name': fname, 'pid': '-', 'activity': f'Background Monitor: High Entropy', 'status': 'Malicious', 'confidence': 100.0})
        detection_results.sort(key=lambda x: (x['status'] != 'Malicious', -x.get('confidence', 0)))
        pdf_path = generate_report(report_path, detection_results=detection_results)
        if pdf_path and os.path.exists(pdf_path):
            return send_file(pdf_path, as_attachment=True, download_name=os.path.basename(pdf_path), mimetype='application/pdf')
        else:
            return (jsonify({'error': 'Failed to generate PDF report'}), 500)
    except Exception as e:
        print(f'[ERROR] Report generation failed: {e}')
        return (jsonify({'error': str(e)}), 500)

@app.route('/clear_ransomware', methods=['POST'])
def clear_ransomware_route():
    """
    Run forensic scan, then terminate detected suspicious processes and
    quarantine ransomware files (only under Desktop, Documents, Pictures, /tmp).
    Returns JSON with terminated PIDs, quarantined paths, and any errors.
    """
    try:
        result = clear_ransomware()
        if pure_python_monitor:
            pure_quarantines = pure_python_monitor.get_quarantined_files(clear=True)
            for qf in pure_quarantines:
                result['quarantined_files'].append(qf.get('original_path', ''))
        return jsonify({'success': True, 'terminated_pids': result['terminated_pids'], 'quarantined_files': result['quarantined_files'], 'quarantine_dir': result['quarantine_dir'], 'errors': result['errors'], 'threat_level': result['threat_level']})
    except Exception as e:
        print(f'[ERROR] Clear ransomware failed: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/pendrive_samples')
def pendrive_samples():
    """
    Return manifest of pendrive ransomware sample folders: folder name,
    infected (yes/no), ransomware name, and which process/family each represents.
    """
    try:
        if not os.path.exists(SAMPLE_MANIFEST_PATH):
            return jsonify({'samples': [], 'error': 'SAMPLE_MANIFEST.json not found', 'description': 'Pendrive samples manifest not found.'})
        with open(SAMPLE_MANIFEST_PATH, 'r') as f:
            data = json.load(f)
        samples = data.get('samples', [])
        for s in samples:
            folder_path = os.path.join(PENDRIVE_SAMPLES_DIR, s.get('folder', ''))
            s['folder_exists'] = os.path.isdir(folder_path)
        return jsonify({'description': data.get('description', ''), 'samples': samples, 'usage': data.get('usage', '')})
    except (json.JSONDecodeError, IOError) as e:
        return (jsonify({'samples': [], 'error': str(e)}), 500)

@app.route('/forensic_scan')
def forensic_scan_route():
    """
    Run a fast filesystem-only forensic scan and return findings as JSON.
    Shows encrypted files, ransom notes, and threat level for the demo panel.
    """
    try:
        scanner = ForensicScanner()
        scanner.scan_processes()
        scanner.scan_filesystem()
        threat_level = scanner.calculate_threat_level()
        attack_vector = scanner.determine_attack_vector()
        return jsonify({'success': True, 'threat_level': threat_level, 'threat_score': scanner.threat_score, 'attack_vector': attack_vector, 'encrypted_files': scanner.encrypted_files, 'ransom_notes': scanner.ransom_notes, 'suspicious_processes': scanner.suspicious_processes, 'encrypted_file_count': len(scanner.encrypted_files), 'ransom_note_count': len(scanner.ransom_notes), 'suspicious_process_count': len(scanner.suspicious_processes)})
    except Exception as e:
        print(f'[ERROR] Forensic scan failed: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/realtime/start')
def realtime_start():
    """Start background monitoring threads."""
    try:
        result = realtime_engine.start()
        return jsonify({'success': True, **result})
    except Exception as e:
        print(f'[ERROR] Real-time start failed: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/realtime/stop')
def realtime_stop():
    """Stop background monitoring threads."""
    try:
        result = realtime_engine.stop()
        return jsonify({'success': True, **result})
    except Exception as e:
        print(f'[ERROR] Real-time stop failed: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/realtime/status')
def realtime_status():
    """Get current monitoring status, window data, and recent alerts."""
    try:
        status = realtime_engine.get_status()
        return jsonify({'success': True, **status})
    except Exception as e:
        print(f'[ERROR] Real-time status failed: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/realtime/alerts')
def realtime_alerts():
    """Get recent real-time alerts and stats."""
    try:
        alerts = realtime_engine.analyzer.get_recent_alerts(limit=100)
        stats = realtime_engine.analyzer.get_stats()
        return jsonify({'success': True, 'alerts': alerts, 'stats': stats})
    except Exception as e:
        print(f'[ERROR] Real-time alerts failed: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)



# --- GLOBALS & SETUP ---

'\nmonitor.py - Process Behavioral Monitoring Module\n=================================================\nMonitors running processes and extracts 10 behavioral features\nfor ransomware detection using machine learning classification.\n\nFeatures extracted:\n    1. open_files   - Count of currently open file handles\n    2. file_read    - Number of file read operations (simulated)\n    3. file_write   - Number of file write operations (simulated)\n    4. file_delete  - Number of file deletion operations (simulated)\n    5. file_rename  - Number of file rename operations (simulated)\n    6. cpu          - CPU usage percentage\n    7. memory       - Memory consumption percentage\n    8. threads      - Number of execution threads\n    9. entropy      - Shannon entropy of write operations\n   10. uptime       - Process execution duration in seconds\n\nAuthor: Ransomware Detection System\nLicense: MIT - Educational Use Only\n'
'\nmodel.py - Machine Learning Training and Prediction Module\n==========================================================\nTrains Decision Tree and Multi-Layer Perceptron classifiers\non synthetic behavioral data for ransomware detection.\n\nModels:\n    1. Decision Tree (CART) - Gini impurity, max_depth=15\n    2. MLP Neural Network   - Hidden(8, ReLU), Adam optimizer\n\nFeatures (10):\n    open_files, file_read, file_write, file_delete, file_rename,\n    cpu, memory, threads, entropy, uptime\n\nClasses:\n    0 = Benign\n    1 = Malicious\n\nAuthor: Ransomware Detection System\nLicense: MIT - Educational Use Only\n'
warnings.filterwarnings('ignore', category=UserWarning)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DT_MODEL_PATH = os.path.join(MODEL_DIR, 'decision_tree.pkl')
MLP_MODEL_PATH = os.path.join(MODEL_DIR, 'mlp_model.pkl')
FEATURE_NAMES = ['open_files', 'file_read', 'file_write', 'file_delete', 'file_rename', 'cpu', 'memory', 'threads', 'entropy', 'uptime']
"\npure_python_detector.py\n======================================================\nRansomware Detection Program (Standard Library Only)\n======================================================\nAuthor: Human-written specifically for standard environments.\nRequirements: Python 3.x (No third-party libraries needed).\n\nDescription:\nThis script monitors a specific directory and its subdirectories\nin real-time for ransomware-like behavior. It operates purely\non Python's built-in standard libraries to maintain zero external \ndependencies.\n\nKey Features:\n1. Real-time folder monitoring (via periodic polling).\n2. Entropy calculation to detect highly encrypted files.\n3. Mass modification detection (rapid file changes).\n4. Automated quarantine of suspicious files.\n5. Plain text logging of all actions and detected anomalies.\n"
WATCH_DIRECTORY = os.path.expanduser('~/Ransomware Detection System/test_watch_folder')
QUARANTINE_DIRECTORY = os.path.expanduser('~/Ransomware Detection System/pure_quarantine')
LOG_FILE = os.path.expanduser('~/Ransomware Detection System/pure_detector.log')
POLL_INTERVAL = 2.0
ENTROPY_THRESHOLD = 7.5
MASS_MODIFICATION_THRESHOLD = 5
MASS_MODIFICATION_WINDOW = 10.0
logger = setup_logging()
'\nthreat_analyzer.py\n\nSliding window based threat correlation engine for real-time\nransomware detection. Tracks file writes, renames, suspicious\nextensions, entropy spikes, and CPU spikes within a rolling\n5-second window. If thresholds are breached, it automatically\nsuspends/terminates the offending process and quarantines files.\n'
SUSPICIOUS_EXTENSIONS = {'.locked', '.encrypted', '.crypto', '.ransom', '.enc', '.crypt', '.pay', '.zzz', '.aaa', '.abc', '.xyz', '.micro', '.cerber', '.zepto', '.thor', '.locky', '.wallet', '.dharma', '.onion', '.wncry', '.wcry'}
WINDOW_SIZE = 5.0
WRITE_BURST_THRESHOLD = 20
WRITE_BURST_WINDOW = 3.0
RENAME_BURST_THRESHOLD = 10
SUSPICIOUS_EXT_THRESHOLD = 5
MULTI_INDICATOR_THRESHOLD = 3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUARANTINE_DIR = os.path.join(BASE_DIR, 'quarantine')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
'\nfile_watcher.py\n\nUses the watchdog library to monitor key directories for\nrapid file modifications, creations, and renames that may\nindicate ransomware activity. Events are forwarded to the\nThreatAnalyzer for correlation.\n'
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object
'\nforensic_scanner.py - Comprehensive Forensic Analysis Tool\n==========================================================\nScans system processes and filesystem for indicators of\nransomware infection. Generates detailed JSON reports.\n\nDetection Methods:\n    - Suspicious process name matching\n    - Encrypted file extension scanning\n    - Ransom note detection\n    - Threat level calculation\n\nAuthor: Ransomware Detection System\nLicense: MIT - Educational Use Only\n'
'\nrealtime_monitor.py\n\nCoordinates background process scanning and filesystem watching\nfor continuous ransomware detection. The RealtimeEngine class\nmanages the lifecycle of all monitoring components.\n'
SCAN_INTERVAL = 2.0
MALICIOUS_CONFIDENCE = 80.0
CPU_SPIKE_THRESHOLD = 80.0
ENTROPY_SPIKE_THRESHOLD = 6.5
ALLOWED_PATHS = ['/tmp', '/media', '/run/media', os.path.expanduser('~/Desktop'), os.path.expanduser('~/Documents'), os.path.expanduser('~/Pictures')]
'\ngenerate_pdf_report.py - Forensic PDF Report Generator (1-page, essential only).\nSingle-page B&W report: threat level, Detection Results table, brief summary.\nNo Incident Response section. Threat level: CLEAN=none, MEDIUM=some, HIGH=significant, CRITICAL=severe.\n'
'\ndecrypt_ransomware.py - File Recovery Tool\n==========================================\nDecrypts files encrypted by the ransomware simulator.\nLoads key from /tmp/ransomware_key.secret and restores all .locked files.\n\nAuthor: Ransomware Detection System\nLicense: MIT - Educational Use Only\n'
'\nsafe_removal.py\n\nHandles safe ransomware process termination, file quarantine,\nand secure deletion of quarantined files. All destructive\noperations are restricted to whitelisted paths only.\n'
PROTECTED_DIRS = {'/usr', '/bin', '/sbin', '/etc', '/lib', '/lib64', '/boot', '/proc', '/sys', '/dev', '/var', '/snap'}
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(QUARANTINE_DIR, exist_ok=True)
latest_results = []
realtime_engine = RealtimeEngine()
PENDRIVE_SAMPLES_DIR = os.path.join(BASE_DIR, 'pendrive_ransomware_samples')
SAMPLE_MANIFEST_PATH = os.path.join(PENDRIVE_SAMPLES_DIR, 'SAMPLE_MANIFEST.json')
ALLOWED_CLEAR_PATHS = ['/tmp', '/media', '/run/media', os.path.expanduser('~/Desktop'), os.path.expanduser('~/Documents'), os.path.expanduser('~/Pictures')]
pure_python_monitor = None
start_pure_python_monitor()

# --- STARTUP ---
if __name__ == '__main__':
    print('\n' + '='*60)
    print('  🛡️  RANSOMWARE DETECTION SYSTEM MERGED')
    print('='*60)
    ensure_models_exist()
    print('\n[✓] Dashboard available at: http://localhost:5000')
    print('[*] Press Ctrl+C to stop the server\n')
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
