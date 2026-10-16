import os
import time
import uuid
import json
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from Book_recommender.constant import PROJECT_ROOT
from Book_recommender.logging.log import logger
from Book_recommender.pipeline.prediction import RecommenderPredictor

INDEX_HTML = PROJECT_ROOT / "templates" / "index.html"

MODEL_CARD_PATH = PROJECT_ROOT / "artifacts" / "model_evaluation" / "model_card.json"


def _load_model_card():
    try:
        return json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("model_card.json missing or unreadable")
        return None


# ---------- response schemas ----------
class Book(BaseModel):
    title: str
    authors: str
    average_rating: float
    genre: str
    cover_url: str
    isbn13: str


class RecommendResponse(BaseModel):
    recommendations: List[Book]
    count: int


class TopResponse(BaseModel):
    top_books: List[Book]
    count: int


class GenresResponse(BaseModel):
    genres: List[str]
    count: int


# ---------- startup ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_card = _load_model_card()
    app.state.predictor = None
    app.state.genres = []
    try:
        predictor = RecommenderPredictor()
        app.state.genres = sorted(
            g for g in predictor.data["genre_clean"].dropna().unique() if g
        )
        app.state.predictor = predictor
    except Exception:
        # Stay up so /health works and /ready reports the problem
        logger.exception("Could not load model artifacts; service is NOT ready")
    yield


app = FastAPI(
    title="Book Recommender API",
    description="Get book recommendations by title, author, or genre",
    version="1.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")

# The UI is served by this same app, so CORS is off unless you opt in:
#   CORS_ORIGINS="https://example.com,https://other.com"
_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID", "")
    rid = incoming if incoming and len(incoming) <= 64 and incoming.replace("-", "").isalnum() \
        else uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"rid={rid} unhandled error on {request.method} {request.url.path}")
        return JSONResponse(
            {"detail": "Internal server error", "request_id": rid},
            status_code=500,
            headers={"X-Request-ID": rid},
        )
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    logger.info(f"rid={rid} {request.method} {request.url.path} -> {response.status_code} {ms:.0f}ms")
    return response


def get_predictor(request: Request) -> RecommenderPredictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return predictor


# ---------- ops endpoints ----------
@app.get("/health", tags=["ops"])
def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready(request: Request):
    """Readiness: model and data are loaded."""
    p = request.app.state.predictor
    if p is None:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return {"status": "ready", "model": type(p.model).__name__, "books": len(p.data)}


@app.get("/", include_in_schema=False)
def root():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(INDEX_HTML)

# ---------- API ----------
api = APIRouter()


@api.get("/recommend", response_model=RecommendResponse)
def recommend(
    title: Optional[str] = Query(None, min_length=1, max_length=200),
    author: Optional[str] = Query(None, min_length=1, max_length=200),
    genre: Optional[str] = Query(None, min_length=1, max_length=100),
    min_rating: float = Query(0.0, ge=0.0, le=5.0),
    language: str = Query("eng", pattern=r"^[A-Za-z]{2,3}(-[A-Za-z]{2,4})?$"),
    n: int = Query(5, ge=1, le=20),
    predictor: RecommenderPredictor = Depends(get_predictor),
):
    title, author, genre = (v.strip() if v else None for v in (title, author, genre))
    if not (title or author or genre):
        raise HTTPException(status_code=422, detail="Provide at least one of: title, author, genre")
    try:
        results = predictor.recommend(
            book_title=title, author=author, genre=genre,
            min_rating=min_rating, language=language, n=n,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"recommendations": results, "count": len(results)}


@api.get("/top", response_model=TopResponse)
def top_books(
    genre: str = Query(..., min_length=1, max_length=100),
    min_rating: float = Query(0.0, ge=0.0, le=5.0),
    n: int = Query(5, ge=1, le=20),
    predictor: RecommenderPredictor = Depends(get_predictor),
):
    try:
        results = predictor.top_by_genre(genre=genre.strip(), min_rating=min_rating, n=n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"top_books": results, "count": len(results)}


@api.get("/genres", response_model=GenresResponse)
def list_genres(request: Request, _: RecommenderPredictor = Depends(get_predictor)):
    genres = request.app.state.genres
    return {"genres": genres, "count": len(genres)}

@api.get("/popular", response_model=TopResponse)
def popular(
    response: Response,
    n: int = Query(15, ge=1, le=30),
    predictor: RecommenderPredictor = Depends(get_predictor),
):
    response.headers["Cache-Control"] = "no-store"   # new picks on every load
    books = predictor.popular_random(n=n)
    return {"top_books": books, "count": len(books)}


@api.get("/model")
def model_info(request: Request, predictor: RecommenderPredictor = Depends(get_predictor)):
    card = request.app.state.model_card
    if card is None:
        raise HTTPException(status_code=404, detail="Model details are not available in this build.")
    commit = (os.environ.get("RAILWAY_GIT_COMMIT_SHA") or "")[:7] or None
    return {**card, "runtime": {"class": type(predictor.model).__name__,
                                "books_loaded": len(predictor.data), "commit": commit}}


app.include_router(api, prefix="/v1")                     # the supported API
app.include_router(api, include_in_schema=False)          # legacy paths, so index.html keeps working

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8080)),
    )

