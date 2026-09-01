# Practical 3 : Containerise a Machine Learning Model for Deployment

A trained ML model (RandomForest on Iris) served as a REST API using Flask,
packaged into a Docker container.

## What this project does

- Trains and saves a RandomForest model on the Iris dataset (train.py)
- Serves predictions over HTTP via a Flask app (app.py) with a health
  route (/) and a prediction route (/predict)
- Packages everything into a Docker image (iris-api) that runs the same
  way on any machine

## Project structure
```

practical-3-containerise-ml/
├── train.py
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── model.joblib

```

## Setup

1. Train the model (only needed if model.joblib is missing):
   python train.py

2. Build the Docker image:
   docker build -t iris-api .

3. Run the container:
   docker run -p 5000:5000 iris-api

## Testing the API

Health check:
```
   curl http://localhost:5000/

```

Prediction request:

```
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

```

Expected response:

```
   {"prediction": 0}

```