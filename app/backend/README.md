# FastAPI Backend

Backend server for running scene generation sampling.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or if using the project's virtual environment:
```bash
source ../../.venv/bin/activate
pip install fastapi uvicorn[standard] pydantic python-multipart
```

## Running

Start the server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API documentation available at `http://localhost:8000/docs`




