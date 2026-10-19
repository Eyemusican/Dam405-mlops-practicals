from flask import Flask, request, jsonify
import joblib, numpy as np
import logging
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from sklearn.datasets import load_iris

# ---------------- Task 1: Configure logging ----------------
# One named logger for the whole service
logger = logging.getLogger('ml-service')
logger.setLevel(logging.INFO)  # use DEBUG in development

# Every line gets a timestamp, a level and the message
fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

# Handler 1: print log lines to the console
console = logging.StreamHandler()
console.setFormatter(fmt)

# Handler 2: write log lines to service.log
# Rotates at about 1 MB and keeps 3 old files
fileh = RotatingFileHandler('service.log', maxBytes=1_000_000, backupCount=3)
fileh.setFormatter(fmt)

logger.addHandler(console)
logger.addHandler(fileh)

# ---------------- Task 4: Monitoring signals ----------------
# Simple counters; latencies keeps only the last 100 values
stats = {'requests': 0, 'errors': 0, 'latencies': deque(maxlen=100)}

def record(latency_ms, ok=True):
    # update the counters after every request
    stats['requests'] += 1
    if not ok:
        stats['errors'] += 1
    stats['latencies'].append(latency_ms)

def summary():
    # log one line summarising the service so far
    lat = stats['latencies']
    avg = sum(lat) / len(lat) if lat else 0
    logger.info(f"requests={stats['requests']} "
                f"errors={stats['errors']} avg_latency_ms={avg:.1f}")
    return {'requests': stats['requests'], 'errors': stats['errors'],
            'avg_latency_ms': round(avg, 1)}

SUMMARY_EVERY = 5  # log a summary automatically every 5 requests

# ---------------- Task 5: Basic drift check ----------------
FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Training mean of each feature, from the iris data the model was trained on
train_means = load_iris().data.mean(axis=0)

# Live feature values from recent successful requests (last 50 per feature)
live = [deque(maxlen=50) for _ in FEATURES]

def drift_check(train_mean, live_values, tol=0.25, name='feature'):
    # relative change between the live mean and the training mean
    live_mean = float(np.mean(live_values))
    rel = abs(live_mean - train_mean) / (abs(train_mean) + 1e-9)
    if rel > tol:
        logger.warning(f'possible drift in {name}: live={live_mean:.3f} '
                       f'train={train_mean:.3f} change={rel:.0%}')
    return rel

# ---------------- App and model ----------------
app = Flask(__name__)
model = joblib.load('model.joblib')
logger.info('model loaded from model.joblib')

@app.get('/')
def health():
    logger.info('health check called')
    return {'status': 'ok'}

# ---------------- Task 2: Log around predictions ----------------
@app.post('/predict')
def predict():
    start = time.perf_counter()  # start timing the request
    try:
        data = request.get_json()
        logger.debug(f'request features: {data}')  # input details, only shown in DEBUG mode
        x = np.array(data['features']).reshape(1, -1)
        pred = int(model.predict(x)[0])
        ms = (time.perf_counter() - start) * 1000  # latency in milliseconds
        logger.info(f'prediction={pred} latency_ms={ms:.1f}')
        record(ms, ok=True)  # Task 4: count a successful request
        # Task 5: keep the live feature values for the drift check
        for i, value in enumerate(data['features']):
            live[i].append(float(value))
        return jsonify({'prediction': pred})
    except Exception as e:
        # any failure is logged at ERROR level and returned as a 400
        ms = (time.perf_counter() - start) * 1000
        logger.error(f'prediction failed: {e}')
        record(ms, ok=False)  # Task 4: count a failed request
        return jsonify({'error': str(e)}), 400
    finally:
        # periodic summary line, as the handout suggests
        if stats['requests'] % SUMMARY_EVERY == 0:
            summary()

# Task 4: endpoint to see the summary on demand
@app.get('/stats')
def get_stats():
    return jsonify(summary())

# Task 5: endpoint to run the drift check on the live data
@app.get('/drift')
def get_drift():
    if len(live[0]) < 5:
        return jsonify({'message': 'not enough live data yet, need at least 5 requests'})
    logger.info(f'drift check on last {len(live[0])} requests')
    result = {}
    for i, name in enumerate(FEATURES):
        rel = drift_check(float(train_means[i]), list(live[i]), name=name)
        result[name] = f'{rel:.0%}'  # percentage change from training mean
    return jsonify(result)

if __name__ == '__main__':
    logger.info('starting ml-service on port 5000')
    app.run(host='0.0.0.0', port=5000)