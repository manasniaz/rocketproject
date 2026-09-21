"""Tests. The AI is mocked, so these are free, fast and need no API key."""

import asyncio

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

import agent
import app as app_module
from agent import Plan, PlanError, StudentProfile

client = TestClient(app_module.app)

PROFILE = {
    "skills": "Python, Excel",
    "experience": "2nd-year student",
    "hours_per_week": 8,
    "goal": "earn $300 a month",
    "constraints": "remote only",
}

VALID_PLAN = {
    "summary": "A short summary.",
    "opportunities": [
        {"title": "T", "who_might_pay": "W", "what_to_offer": "O", "why_it_fits": "F"}
    ],
    "hypotheses_to_test": [{"statement": "S", "cheap_test": "C"}],
    "skill_gaps": [{"skill": "K", "why_it_matters": "M", "how_to_close_it": "H"}],
    "top_opportunity": "T",
    "next_actions": [{"step": "Do it", "time_needed": "1 h"}],
    "assumptions": ["A"],
    "uncertainties": ["U"],
}


# ---- The web endpoint --------------------------------------------------------


def test_home_page_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Skills-to-Income Planner" in response.text


def test_plan_endpoint_returns_structured_plan(monkeypatch):
    async def fake_generate(profile):
        return Plan.model_validate(VALID_PLAN)

    monkeypatch.setattr(app_module, "generate_plan", fake_generate)
    response = client.post("/api/plan", json=PROFILE)
    assert response.status_code == 200
    assert response.json()["top_opportunity"] == "T"


@pytest.mark.parametrize(
    "change",
    [
        {"hours_per_week": 0},
        {"hours_per_week": 500},
        {"skills": ""},
        {"skills": "x" * 501},
        {"goal": ""},
    ],
)
def test_invalid_input_is_rejected_before_calling_the_ai(monkeypatch, change):
    async def must_not_run(profile):
        raise AssertionError("the AI should not be called for invalid input")

    monkeypatch.setattr(app_module, "generate_plan", must_not_run)
    response = client.post("/api/plan", json={**PROFILE, **change})
    assert response.status_code == 422


def test_plan_error_becomes_readable_message(monkeypatch):
    async def fake_generate(profile):
        raise PlanError("Groq's free-tier rate limit was hit.", 429)

    monkeypatch.setattr(app_module, "generate_plan", fake_generate)
    response = client.post("/api/plan", json=PROFILE)
    assert response.status_code == 429
    assert "rate limit" in response.json()["detail"]


# ---- The agent logic ---------------------------------------------------------

STUDENT = StudentProfile(**PROFILE)


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(PlanError) as error:
        asyncio.run(agent.generate_plan(STUDENT))
    assert "GROQ_API_KEY" in error.value.message


def test_bad_json_is_retried_once_then_succeeds(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    replies = iter(
        ["this is not json", Plan.model_validate(VALID_PLAN).model_dump_json()]
    )

    async def fake_run(agent_obj, profile):
        return next(replies)

    monkeypatch.setattr(agent, "_run_agent", fake_run)
    plan = asyncio.run(agent.generate_plan(STUDENT))
    assert plan.top_opportunity == "T"


def test_bad_json_twice_gives_friendly_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    async def fake_run(agent_obj, profile):
        return "still not json"

    monkeypatch.setattr(agent, "_run_agent", fake_run)
    with pytest.raises(PlanError) as error:
        asyncio.run(agent.generate_plan(STUDENT))
    assert "unexpected format" in error.value.message


def _api_error(cls, status, body=None):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=request), body=body)


def _run_agent_raising(monkeypatch, exc):
    async def fake_runner_run(*args, **kwargs):
        raise exc

    monkeypatch.setattr(agent.Runner, "run", fake_runner_run)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    return asyncio.run(agent._run_agent(agent._build_agent(), STUDENT))


def test_rate_limit_gives_429_message(monkeypatch):
    with pytest.raises(PlanError) as error:
        _run_agent_raising(monkeypatch, _api_error(openai.RateLimitError, 429))
    assert error.value.status_code == 429


def test_invalid_key_gives_clear_message(monkeypatch):
    with pytest.raises(PlanError) as error:
        _run_agent_raising(monkeypatch, _api_error(openai.AuthenticationError, 401))
    assert "GROQ_API_KEY" in error.value.message


def test_groq_json_failure_is_treated_as_bad_answer(monkeypatch):
    exc = _api_error(openai.BadRequestError, 400, {"code": "json_validate_failed"})
    assert _run_agent_raising(monkeypatch, exc) == ""
