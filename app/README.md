# Scene Generation Web Application

A web application with FastAPI backend and React frontend for running scene generation sampling.

## Project Structure

```
app/
├── backend/          # FastAPI backend server
│   ├── main.py      # Main API server
│   └── requirements.txt
├── frontend/        # React frontend application
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   └── package.json
└── sampling.py      # Wrapper script for sampling
```

## Quick Start

### Backend Setup

1. Navigate to the backend directory:
```bash
cd app/backend
```

2. Install dependencies (using the project's virtual environment):
```bash
source ../../.venv/bin/activate
pip install fastapi uvicorn[standard] pydantic python-multipart
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

3. Start the backend server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd app/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The app will open at `http://localhost:3000`

## Usage

1. Start both the backend and frontend servers
2. Open the frontend in your browser
3. Enter the number of scenes you want to generate
4. Optionally configure advanced parameters
5. Click "Generate Scenes" to start the process
6. Monitor the status in real-time

## API Endpoints

- `POST /api/sampling/run` - Start a new sampling task
- `GET /api/sampling/status/{task_id}` - Get status of a task
- `GET /api/sampling/tasks` - List all tasks

## Notes

- The backend runs the sampling command asynchronously
- Task status is polled every 2 seconds
- Output directories are automatically detected from the `outputs/` folder
- The application uses the project's virtual environment if available




