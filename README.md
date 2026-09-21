# Skills-to-Income Planner

A class project built with the **OpenAI Agents SDK**. A student describes their skills, experience, weekly hours, goal and constraints. An AI agent returns a realistic plan: opportunities, hypotheses to test, skill gaps and next actions. It never promises income.

It runs locally only. It has no accounts and no database, and it saves nothing.

## Run it

You need [uv](https://docs.astral.sh/uv/) and a free Groq API key from https://console.groq.com.

```
uv sync                        # installs everything into .venv
copy .env.example .env         # then open .env and paste your key after GROQ_API_KEY=
uv run uvicorn app:app --reload
```

Open http://127.0.0.1:8000, fill in the form and press **Build my plan**. A plan takes about 10 to 30 seconds.

Run the tests (the AI is mocked, so no key or internet is needed):

```
uv run pytest
```

## How it works

```
browser form  ->  POST /api/plan  ->  app.py (validates input)
                                        ->  agent.py (Agents SDK agent -> Groq)
                                        ->  validated Plan (Pydantic)  ->  page renders it
```

| File | What it does |
|---|---|
| `static/index.html` | The form and results page (plain HTML, CSS and JS). |
| `app.py` | FastAPI server: serves the page and has one endpoint, `/api/plan`. |
| `agent.py` | The agent: its instructions, the Groq connection and the shape of the answer. |
| `test_app.py` | Tests, with the AI mocked. |

### Talking points for class

- **One `Agent`** with clear instructions, run with `Runner.run(...)`. There are no tools or handoffs because this task doesn't need them.
- **Groq through the Agents SDK.** The SDK is made for OpenAI, but Groq has an OpenAI-compatible API. We point an `AsyncOpenAI` client at `https://api.groq.com/openai/v1` and wrap it in `OpenAIChatCompletionsModel`.
- **Tracing is turned off** (`set_tracing_disabled(True)`). By default the SDK uploads traces to OpenAI, which we don't use. This also means no OpenAI key is needed.
- **Why the SDK's `output_type` isn't used.** We tried it first. Groq rejected the strict JSON-schema request with a 400 error (`json_validate_failed`). The fix was to use Groq's plain JSON mode, put the schema in the instructions, and validate the reply with Pydantic (`Plan`). If the reply is malformed, we retry once and then show a friendly error.
- **Honesty by design.** The instructions forbid income guarantees and invented resources. The output is split into labeled parts (possible opportunities, hypotheses to test, skill gaps, next actions, assumptions, uncertainties). The page always shows a "no income is guaranteed" notice, and that notice is fixed in the page, not written by the AI.
- **Prompt injection.** The user's text is treated as data, and the instructions tell the agent to ignore commands inside it.

## Limits

- The Groq free tier is rate-limited. If you hit the limit, the app says so and you wait a minute.
- Model: `openai/gpt-oss-120b` by default. Change it with `GROQ_MODEL` in `.env`.
- The advice comes from an AI, so treat it as ideas to test, not facts. Prices in the output are examples.
- What the user types is sent to Groq. The page says so.

## Keep your key safe

`.env` is git-ignored. Never paste your key into code, screenshots or chat. If it leaks, delete it in the Groq console and make a new one.
