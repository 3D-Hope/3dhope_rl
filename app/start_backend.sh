#!/bin/bash
# Start the FastAPI backend server

cd "$(dirname "$0")/backend"

# Activate virtual environment if it exists
if [ -d "../../.venv" ]; then
    source ../../.venv/bin/activate
fi

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing backend dependencies..."
    pip install -r requirements.txt
fi

# Start the server
echo "Starting FastAPI backend server on http://localhost:8000"
uvicorn main:app --reload --host 0.0.0.0 --port 8000




