"""The web server: serves the page and one endpoint that builds a plan.

Run it with:  uv run uvicorn app:app --reload
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from agent import Plan, PlanError, StudentProfile, generate_plan

app = FastAPI(title="Skills-to-Income Planner")

INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)


@app.post("/api/plan", response_model=Plan)
async def create_plan(profile: StudentProfile):
    # FastAPI has already validated `profile` (types and length limits) and
    # answers 422 by itself if it is invalid. Nothing is stored or logged.
    try:
        return await generate_plan(profile)
    except PlanError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
