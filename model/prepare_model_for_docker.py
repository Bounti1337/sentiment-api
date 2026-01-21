import os
import logging
import joblib
from fastapi import FastAPI, APIRouter , Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .preproc_class import prepocessing_data
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------- config ----------
BASE_DIR = os.path.dirname(__file__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment-api")

# ---------- load artifacts ----------
model = joblib.load(os.path.join(BASE_DIR, "logreg_text.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))
preproc = prepocessing_data()

label_map = {0: "negative", 1: "neutral", 2: "positive"}

# ---------- app ----------
app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0"
)

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- schemas ----------
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)

# ---------- routes ----------

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/health")
def health():
    return {"status": "ok"}

router_v1 = APIRouter(prefix="/v1")

@router_v1.post("/predict")
def predict_sentiment(req: TextRequest):
    logger.info(f"Text length={len(req.text)}")

    clean_text = preproc.clean(req.text)
    lem_text = preproc.lemma(clean_text)
    tfidf_text = tfidf.transform([lem_text])

    pred = model.predict(tfidf_text)[0]
    proba = model.predict_proba(tfidf_text).max()

    result = {
        "label": label_map[pred],
        "confidence": float(proba)
    }

    logger.info(f"Result={result}")
    return result

app.include_router(router_v1)
