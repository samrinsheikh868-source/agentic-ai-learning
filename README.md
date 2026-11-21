# CareerLens — AI Job Search & Screening

A Streamlit app that searches live job listings (JSearch / OpenWeb Ninja), scores
them against a list of skills, and lets you shortlist, compare and export them.
**Live demo:** https://career-lens-app.streamlit.app


## Features

- Live job search by role, location, country, posting age and remote-only
- **Skill match score** — deterministic keyword overlap, computed in plain Python
- Shortlist (save jobs) with CSV export
- Side-by-side comparison of 2–4 jobs
- Insights: job count, remote count, salary disclosure, posting freshness
- Optional LangSmith tracing
- Dark, responsive UI

## How the match score works

The score is `matched skills / total skills * 100`, computed on the full job
description by plain Python — no LLM. That makes it cheap, fast and explainable:
the card tells you exactly which skills matched and which did not.

### Caveats (read this before trusting a score)

1. **Keyword matching is shallow.** "Python" in a posting may be a
   nice-to-have, not a requirement. A 100% score means *the words appear*,
   not that you are a strong fit.
2. **It is case-insensitive but literal.** `Power BI`, `PowerBI` and `PBI` are
   different tokens. Synonyms and abbreviations are not resolved.
3. **Descriptions vary in quality.** If a posting omits detail, the score
   under-reports. Some sources publish very short descriptions.
4. **The LLM in the loop is not scoring.** Groq only orchestrates the tool call
   and writes the summary text. All numbers come from Python.
5. **Job data may be stale or duplicated.** Listings are aggregated from
   Google for Jobs; the same role can appear under several publishers.

`publisher` is where the listing was *found* (LinkedIn, Shine, UNjobnet,
Jobrapido…). `apply_link` is where clicking *applies* — often a redirect
domain, not the publisher's own site.

## Setup

Requires Python 3.13.

```bash
# with uv (recommended)
uv sync

# or with pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```bash
GROQ_API_KEY=...
JSEARCH_API_KEY=...

# optional tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=job-search-agent-local
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

## Run

```bash
streamlit run src/agentic_ai_learning/app.py
```

Streamlit's own Runner button will not work — it must be launched as a
Streamlit app, not as a plain Python script.

## Deploy (Streamlit Community Cloud)

1. Push the repo to GitHub (`.env` must stay untracked).
2. On share.streamlit.io: **New app** → repo → branch → main file path
   `src/agentic_ai_learning/app.py`.
3. **Settings → Secrets**:

```toml
GROQ_API_KEY = "..."
JSEARCH_API_KEY = "..."
LANGSMITH_API_KEY = "..."
LANGSMITH_TRACING = "true"
LANGSMITH_PROJECT = "job-search-agent-production"
```

## Architecture

```
app.py
├── build_tool()      JSearch call + skill scoring (scores computed here,
│                     so full descriptions never reach the LLM)
├── get_agent()       cached LangChain agent (ChatGroq) bound to the filters
├── extract_jobs()    pulls the tool's JSON result out of the agent messages
└── UI                sidebar filters → hero → search → 4 result tabs
```

Why scoring happens inside the tool: the tool result is sent back to the LLM,
so returning full descriptions blows the free-tier token limit (HTTP 413).
Scoring in code before truncating keeps the score accurate *and* the payload small.

## Known limitations

- Groq free tier: 8K tokens/minute — 10 jobs with long descriptions can hit it.
- JSearch free tier: 200 requests/month.
- No authentication or multi-user state; shortlist lives in one browser session.

## Live demo

Deployed on Streamlit Community Cloud: results are cached for 15 minutes and
searches are throttled to one every 8 seconds, because the free tiers are small
(JSearch: 200 requests/month, Groq: 8K tokens/minute). Heavy use will hit those
limits — that is a quota limit, not a bug.
