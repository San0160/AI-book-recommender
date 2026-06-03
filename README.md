# End to End Book recomender system

# How to Run?

### Step1: Create a conda env
```bash
conda create -n books python=3.10 -y
```

```bash
conda activate books
```

### Step2: Install the requirements

```bash
pip install -r requirements.txt
```

### Step3: Get model artifacts
```bash
python main.py
```

# Step4: Run the app locally
```bash
app.py
```

## Description
# 📚 AI Book Recommender

An end-to-end content-based book recommendation system built with a modular ML pipeline, served via a REST API.

## Overview

Given a book title, author, or genre, the system returns personalised recommendations using TF-IDF vectorisation and K-Nearest Neighbours similarity. The full pipeline covers data ingestion through model evaluation, with a FastAPI serving layer on top.

## Tech Stack

| Layer | Tools |
|---|---|
| Data pipeline | Python, Pandas, scikit-learn |
| Enrichment | Open Library API |
| Model | TF-IDF + KNN (scikit-learn) |
| API | FastAPI, Uvicorn |
| CI/CD | GitHub Actions |
| Containerisation | Docker |

## Project Structure

```
├── src/Book_recommender/
│   ├── components/        # Data ingestion, validation, transformation, training, evaluation
│   ├── config/            # Configuration manager
│   ├── entity/            # Dataclass configs
│   ├── pipeline/          # Stage pipelines + inference
│   └── utils/             # Shared utilities
├── config/config.yaml     # Pipeline configuration
├── artifacts/             # Generated model artifacts (gitignored)
├── templates/index.html   # Frontend UI
├── app.py                 # FastAPI application
├── main.py                # Full training pipeline entry point
└── dockerfile
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Search UI |
| `GET /recommend` | Content-based recommendations by title, author, or genre |
| `GET /top` | Top-rated books filtered by genre |
| `GET /genres` | List all available genres |
| `GET /docs` | Interactive Swagger UI |


Visit `http://localhost:8080` for the search UI or `http://localhost:8080/docs` for the API explorer.

## Model Performance

Evaluated on a sample of 200 books from the Goodreads dataset (11,127 books total):

| Metric | Score |
|---|---|
| Precision@5 | 0.43 |
| Diversity score | 0.61 |
| Catalog coverage | 0.09 |

## License

MIT

### Imporvements
### exception handler not present in source code
### Data/model not from hugging face
### Data processing -> read_csv in pipeline code 
