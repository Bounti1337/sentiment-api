# Link 
https://sentiment-api-s7yc.onrender.com

# Sentiment Analysis API

REST API for sentiment analysis of English text using a trained machine learning model.
The service predicts sentiment (`negative`, `neutral`, `positive`) and confidence score.

---

##  Features

- Sentiment prediction for text input
- Confidence score for each prediction
- REST API built with FastAPI
- Dockerized application
- API versioning (`/v1`)
- Health check endpoint

---

##  Model

- TF-IDF vectorizer
- Logistic Regression classifier
- Trained on Twitter Airline Sentiment dataset

---

##  Tech Stack

- Python 3.11
- FastAPI
- scikit-learn
- spaCy
- Docker

---

##  Run with Docker

### Through terminal
```bash
curl -X POST https://sentiment-api-s7yc.onrender.com/v1/predict \
-H "Content-Type: application/json" \
-d '{"text":"This airline is terrible"}'

