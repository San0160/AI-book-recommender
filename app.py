# app.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from contextlib import asynccontextmanager
from Book_recommender.pipeline.prediction import RecommenderPredictor

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = RecommenderPredictor()
    yield

app = FastAPI(
    title="Book Recommender API",
    description="Get book recommendations by title, author, or genre",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

templates = Jinja2Templates(directory="templates")

@app.get("/")
def root():
    return {"status": "API is running"}

@app.get("/recommend")
def recommend(
    title:      str   = Query(None, description="Book title to find similar books"),
    author:     str   = Query(None, description="Author name"),
    genre:      str   = Query(None, description="Genre filter"),
    min_rating: float = Query(0.0,  description="Minimum average rating"),
    language:   str   = Query("eng", description="Language code"),
    n:          int   = Query(5,    description="Number of recommendations")
):
    try:
        results = predictor.recommend(
            book_title=title,
            author=author,
            genre=genre,
            min_rating=min_rating,
            language=language,
            n=n
        )
        return {"recommendations": results, "count": len(results)}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/top")
def top_books(
    genre:      str   = Query(...,  description="Genre to get top books for"),
    min_rating: float = Query(0.0,  description="Minimum average rating"),
    n:          int   = Query(5,    description="Number of results")
):
    try:
        results = predictor.top_by_genre(genre=genre, min_rating=min_rating, n=n)
        return {"top_books": results, "count": len(results)}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/genres")
def list_genres():
    genres = predictor.data["genre_clean"].dropna()
    genres = genres[genres != ""].unique().tolist()
    genres.sort()
    return {"genres": genres, "count": len(genres)}


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(app, host="0.0.0.0", port=port)