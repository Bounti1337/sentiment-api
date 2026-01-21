FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl


# ❗ КОПИРУЕМ ВСЮ ПАПКУ model
COPY model/ ./model/

CMD ["uvicorn", "model.prepare_model_for_docker:app", "--host", "0.0.0.0", "--port", "8000"]
