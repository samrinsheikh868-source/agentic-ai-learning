"""CareerLens - job search + screening tool (saved jobs, match score, compare, insights)."""

import html
import json
import os
import textwrap

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY")
JSEARCH_URL = "https://api.openwebninja.com/jsearch/search-v2"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = """You are a job search assistant.

When the user asks for jobs, call search_job exactly once.
Then answer with a short numbered list: title, company, location, apply link.

Rules:
- No tables, no headings, no commentary.
- Use only the data returned by the tool. Never invent a link.
"""

st.set_page_config(
    page_title="CareerLens | Job Screening",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(markup: str) -> None:
    """Dedent HTML so Markdown doesn't render it as a code block."""
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&display=swap');

      .stApp,
      .stApp button,
      .stApp input,
      .stApp textarea,
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stWidgetLabel"] {
          font-family: 'Inter', sans-serif;
      }
      .stApp {
          background:
            radial-gradient(900px 500px at 10% -8%, #10263f 0%, rgba(7,10,19,0) 60%),
            radial-gradient(700px 420px at 92% 0%, #1f1a3d 0%, rgba(7,10,19,0) 55%),
            #070a13;
      }
      .block-container { max-width: 1180px; padding-top: 1.2rem; }

      .hero {
          position: relative; overflow: hidden;
          border-radius: 26px; padding: 40px 42px 36px;
          background: linear-gradient(120deg, #0f3a46 0%, #1c2a63 48%, #3a2a78 100%);
          border: 1px solid rgba(255,255,255,.08);
          box-shadow: 0 24px 60px rgba(0,0,0,.5);
      }
      .hero::after {
          content: ""; position: absolute; inset: 0;
          background: radial-gradient(420px 180px at 88% -20%, rgba(90,220,200,.22), transparent 70%);
          pointer-events: none;
      }
      .hero-badge {
          display: inline-block; font-size: .66rem; letter-spacing: .14em;
          text-transform: uppercase; color: #b8f0e6;
          background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.12);
          padding: 6px 12px; border-radius: 999px;
      }
      .hero h1 {
          font-size: 2.35rem; line-height: 1.16; font-weight: 800;
          color: #fff; margin: 14px 0 10px; letter-spacing: -.02em;
      }
      .hero p { color: #b8c2d6; max-width: 720px; font-size: .94rem; margin: 0; }

      .job-card {
          background: rgba(255,255,255,.035);
          border: 1px solid rgba(255,255,255,.09);
          border-radius: 16px; padding: 18px 20px; margin-bottom: 12px;
      }
      .job-title { font-size: 1.04rem; font-weight: 700; color: #f8fafc; margin-bottom: 3px; }
      .company { color: #9ae6d8; font-weight: 600; font-size: .9rem; margin-bottom: 10px; }
      .job-desc { color: #9aa7bd; font-size: .87rem; line-height: 1.55; margin: 8px 0 2px; }
      .job-source { color: #7c8798; font-size: .76rem; margin: 2px 0 6px; }
      .chip {
          display: inline-block; font-size: .74rem; color: #c7d2fe;
          background: rgba(99,102,241,.16); border: 1px solid rgba(99,102,241,.28);
          padding: 3px 10px; border-radius: 999px; margin: 0 6px 6px 0;
      }
      .score {
          display: inline-block; font-size: .74rem; font-weight: 700;
          padding: 3px 10px; border-radius: 999px; margin-right: 6px;
          color: #06281f; background: #4ade80;
      }
      .score.mid { background: #fbbf24; color: #2b1a00; }
      .score.low { background: #f87171; color: #2b0000; }

      .empty {
          border: 1px solid rgba(255,255,255,.10);
          background: rgba(255,255,255,.02);
          border-radius: 18px; padding: 40px; text-align: center;
          color: #93a0b8; font-size: .92rem;
      }
      .footer-note { text-align: center; color: #5f6b80; font-size: .76rem; margin-top: 26px; }

      section[data-testid="stSidebar"] {
          background: #080c17; border-right: 1px solid rgba(255,255,255,.06);
      }
      .side-brand-title { color: #e5e7eb; font-weight: 700; font-size: .95rem; margin-bottom: 6px; }
      .side-brand-sub { color: #6b7280; font-size: .75rem; line-height: 1.5; margin-bottom: 16px; }
      .side-title { color: #e5e7eb; font-weight: 700; font-size: .9rem; margin: 6px 0 12px; }
      .side-hint { color: #6b7280; font-size: .73rem; margin-top: -8px; margin-bottom: 6px; }

      div.stButton > button { border-radius: 12px; font-weight: 600; }
      div.stButton > button[kind="primary"] {
          background: linear-gradient(135deg, #0e7490 0%, #7c3aed 100%);
          border: none; color: #fff;
      }
      div.stButton > button[kind="primary"]:hover { filter: brightness(1.1); }

      div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ agent
def match_skills(skills: tuple[str, ...], job_title: str, description: str, job_type: str):
    """Deterministic keyword match, computed on the FULL description in code."""
    if not skills:
        return 0, [], []

    text = f"{job_title} {description} {job_type}".lower()
    hits = [s for s in skills if s.lower() in text]
    missing = [s for s in skills if s.lower() not in text]
    score = round(100 * len(hits) / len(skills))
    return score, hits, missing


def build_tool(
    country: str,
    date_posted: str,
    limit: int,
    remote_only: bool,
    skills: tuple[str, ...],
):
    """Tool bound to the sidebar filters. Scores here so the LLM never sees full descriptions."""

    @tool
    def search_job(query: str) -> list:
        """Search live job postings for a role and a location."""
        params = {
            "query": query,
            "country": country,
            "date_posted": date_posted,
            "num_pages": 1,
        }
        if remote_only:
            params["work_from_home"] = "true"

        response = requests.get(
            JSEARCH_URL,
            headers={"X-API-Key": JSEARCH_API_KEY},
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        jobs = response.json()["data"]["jobs"]
        compact = []

        for j in jobs[:limit]:
            full_description = j.get("job_description") or ""
            score, hits, missing = match_skills(
                skills, j["job_title"], full_description, j["job_employment_type"] or ""
            )
            compact.append(
                {
                    "id": j.get("job_id") or j.get("job_uid") or j.get("job_apply_link"),
                    "title": j["job_title"],
                    "company": j["employer_name"],
                    "location": j["job_location"],
                    "type": j["job_employment_type"],
                    "posted": j["job_posted_at"],
                    "posted_ts": j.get("job_posted_at_timestamp"),
                    "salary": j.get("job_salary_string"),
                    "remote": j.get("job_is_remote"),
                    "apply_link": j["job_apply_link"],
                    "description": full_description[:300],
                    "publisher": j.get("job_publisher"),
                    "score": score,
                    "hits": hits,
                    "missing": missing,
                }
            )

        return compact

    return search_job


@st.cache_resource(show_spinner=False)
def get_agent(
    country: str,
    date_posted: str,
    limit: int,
    remote_only: bool,
    skills: tuple[str, ...],
):
    """One cached agent per filter combination."""
    return create_agent(
        model=ChatGroq(model=MODEL, temperature=0),
        tools=[build_tool(country, date_posted, limit, remote_only, skills)],
        system_prompt=SYSTEM_PROMPT,
    )


def extract_jobs(messages) -> list[dict]:
    """Pull the structured job list out of the tool message."""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        data = message.content
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                continue
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data
    return []


# ------------------------------------------------------------------ scoring
def score_class(score: int) -> str:
    if score >= 70:
        return "score"
    if score >= 40:
        return "score mid"
    return "score low"


def jobs_frame(jobs: list[dict], skills: list[str]) -> pd.DataFrame:
    rows = []
    for job in jobs:
        score = job.get("score", 0)
        hits = job.get("hits", [])
        missing = job.get("missing", [])
        rows.append(
            {
                "Title": job.get("title"),
                "Company": job.get("company"),
                "Location": job.get("location"),
                "Type": job.get("type"),
                "Posted": job.get("posted"),
                "Remote": "yes" if job.get("remote") else "",
                "Salary": job.get("salary") or "",
                "Match %": score if skills else "",
                "Matched skills": ", ".join(hits),
                "Missing skills": ", ".join(missing),
                "Source": job.get("publisher") or "",
                "Apply link": job.get("apply_link"),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ state
for key, default in (("jobs", []), ("saved", {}), ("searched", False), ("error", None)):
    if key not in st.session_state:
        st.session_state[key] = default


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    render_html(
        """
        <div class="side-brand-title">🔎 CareerLens</div>
        <div class="side-brand-sub">
            Search live roles, score them against your skills, and keep a
            shortlist you can export.
        </div>
        """
    )

    render_html('<div class="side-title">Search settings</div>')

    location = st.text_input("Preferred location", value="India")
    date_posted = st.selectbox(
        "Posted",
        options=["today", "3days", "week", "month"],
        index=0,
        format_func=lambda d: {
            "today": "Today",
            "3days": "Last 3 days",
            "week": "Last week",
            "month": "Last month",
        }[d],
    )
    remote_only = st.checkbox("Remote jobs only", value=False)

    st.markdown('<div class="side-hint">How many results to fetch</div>', unsafe_allow_html=True)
    limit = st.slider("Results", 5, 20, 10, label_visibility="collapsed")

    country = st.selectbox(
        "Country",
        options=["in", "us", "gb", "ca", "au", "de", "sg", "ae"],
        format_func=lambda c: {
            "in": "India", "us": "United States", "gb": "United Kingdom",
            "ca": "Canada", "au": "Australia", "de": "Germany",
            "sg": "Singapore", "ae": "UAE",
        }[c],
    )

    st.divider()
    render_html('<div class="side-title">Your skills</div>')
    skills_raw = st.text_area(
        "Comma separated",
        value="SQL, Python, Excel, Power BI, Data Visualization",
        height=90,
        label_visibility="collapsed",
    )
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

    st.divider()
    if st.button("Clear results", use_container_width=True):
        st.session_state.jobs = []
        st.session_state.searched = False
        st.session_state.error = None
        st.rerun()

    st.caption(f"Shortlisted: {len(st.session_state.saved)} jobs")

# ------------------------------------------------------------------ hero
render_html(
    """
    <div class="hero">
        <span class="hero-badge">CareerLens · screening workspace</span>
        <h1>Find work that fits your skills, not just your keywords.</h1>
        <p>
            Live listings from JSearch, scored against your skill list by plain
            Python. Shortlist the best ones, compare them side by side, and
            export a clean CSV for your tracker.
        </p>
    </div>
    """
)

# ------------------------------------------------------------------ search
with st.container(border=True):
    col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
    with col1:
        role = st.text_input(
            "Role, skills, or keywords",
            value="Data Analyst",
            placeholder="e.g. Financial Analyst, Data Analyst, Business Analyst",
        )
    with col2:
        run = st.button("Search jobs", type="primary", use_container_width=True)

if run:
    if not role.strip():
        st.warning("Enter a role or keywords first.")
    else:
        prompt = f"Find {role.strip()} jobs in {location.strip() or 'India'}."

        with st.spinner("Searching live listings…"):
            try:
                result = get_agent(
                    country, date_posted, limit, remote_only, tuple(skills)
                ).invoke(
                    {"messages": [HumanMessage(content=prompt)]}
                )
                st.session_state.jobs = extract_jobs(result["messages"])
                st.session_state.error = None
                st.session_state.searched = True
            except Exception as exc:  # noqa: BLE001
                st.session_state.jobs = []
                st.session_state.searched = True
                message = str(exc)
                if "413" in message or "rate_limit" in message:
                    message = (
                        "Groq token limit hit (413). Lower the results slider "
                        "or use a narrower role."
                    )
                st.session_state.error = message

if st.session_state.get("error"):
    st.error(f"Search failed: {st.session_state['error']}")

jobs = st.session_state.jobs
saved: dict = st.session_state.saved

# ------------------------------------------------------------------ tabs
if not jobs:
    render_html(
        """
        <div class="empty">
            No results yet.<br>
            Pick a role and filters, then hit <b>Search jobs</b>.
        </div>
        """
    )
else:
    ranked = sorted(jobs, key=lambda j: j.get("score", 0), reverse=True)
    matches_tab, saved_tab, compare_tab, insights_tab = st.tabs(
        ["🎯 Matches", "⭐ Shortlist", "⚖️ Compare", "📊 Insights"]
    )

    with matches_tab:
        m1, m2, m3 = st.columns(3)
        m1.metric("Matching roles", len(jobs))
        m2.metric("Companies", len({j.get("company") for j in jobs if j.get("company")}))
        m3.metric("Remote roles", sum(1 for j in jobs if j.get("remote")))

        top_left, top_right = st.columns([3, 1])
        with top_left:
            st.markdown(f"**{len(jobs)} jobs** · sorted by match score")
        with top_right:
            frame = jobs_frame(ranked, skills)
            st.download_button(
                "⬇ Export CSV",
                frame.to_csv(index=False).encode("utf-8"),
                file_name="careerlens_jobs.csv",
                mime="text/csv",
                use_container_width=True,
            )

        for job in ranked:
            job_id = job.get("id") or job.get("apply_link")
            score = job.get("score", 0)
            hits = job.get("hits", [])
            missing = job.get("missing", [])

            chips = "".join(
                f'<span class="chip">{html.escape(str(value))}</span>'
                for value in (job.get("location"), job.get("type"), job.get("posted"),
                              f"via {job['publisher']}" if job.get("publisher") else None,
                              "Remote" if job.get("remote") else None)
                if value
            )

            score_html = (
                f'<span class="{score_class(score)}">{score}% match</span>'
                if skills else ""
            )

            with st.container(border=True):
                render_html(
                    f"""
                    <div class="job-title">{html.escape(str(job.get('title') or 'Untitled role'))}</div>
                    <div class="company">{html.escape(str(job.get('company') or 'Unknown company'))}</div>
                    <div class="chips">{score_html}{chips}</div>
                    <div class="job-source">Listed on {html.escape(str(job.get('publisher') or 'unknown source'))}</div>
                    <div class="job-desc">{html.escape(str(job.get('description') or ''))[:280]}</div>
                    """
                )

                foot_left, foot_mid, foot_right = st.columns(
                    [1, 2.2, 1.1], vertical_alignment="center"
                )

                with foot_left:
                    is_saved = job_id in saved
                    if st.button(
                        "★ Saved" if is_saved else "☆ Save",
                        key=f"save_{job_id}",
                        use_container_width=True,
                    ):
                        if is_saved:
                            saved.pop(job_id, None)
                        else:
                            saved[job_id] = job
                        st.rerun()

                with foot_mid:
                    if hits:
                        st.caption(f"✅ Matched: {', '.join(hits)}")
                    if missing:
                        st.caption(f"Gap: {', '.join(missing)}")

                with foot_right:
                    if job.get("apply_link"):
                        st.link_button(
                            "Apply now ↗", job["apply_link"], use_container_width=True
                        )


    with saved_tab:
        if not saved:
            render_html(
                """
                <div class="empty">
                    Nothing shortlisted yet.<br>
                    Hit <b>☆ Save</b> on any job to build your shortlist.
                </div>
                """
            )
        else:
            saved_list = list(saved.values())
            saved_frame = jobs_frame(saved_list, skills)

            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**{len(saved_list)} shortlisted**")
            with col_b:
                st.download_button(
                    "⬇ Export shortlist",
                    saved_frame.to_csv(index=False).encode("utf-8"),
                    file_name="careerlens_shortlist.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.dataframe(saved_frame, use_container_width=True, hide_index=True)

            if st.button("Clear shortlist"):
                st.session_state.saved = {}
                st.rerun()

    with compare_tab:
        options = {job.get("id") or job.get("apply_link"): job for job in jobs}
        picked = st.multiselect(
            "Pick 2-4 jobs to compare",
            options=list(options),
            format_func=lambda jid: f"{options[jid].get('title')} — {options[jid].get('company')}",
            max_selections=4,
        )

        if len(picked) >= 2:
            compare_rows = []
            for jid in picked:
                job = options[jid]
                score = job.get("score", 0)
                missing = job.get("missing", [])
                compare_rows.append(
                    {
                        "Field": job.get("title"),
                        "Company": job.get("company"),
                        "Location": job.get("location"),
                        "Type": job.get("type"),
                        "Posted": job.get("posted"),
                        "Remote": "yes" if job.get("remote") else "no",
                        "Salary": job.get("salary") or "not disclosed",
                        "Match %": score if skills else "n/a",
                        "Missing skills": ", ".join(missing) or "—",
                    }
                )
            st.dataframe(
                pd.DataFrame(compare_rows).set_index("Field").T,
                use_container_width=True,
            )
        else:
            st.info("Select at least two jobs to compare.")

    with insights_tab:
        now = pd.Timestamp.now(tz="UTC")
        stamps = pd.to_datetime(
            [job.get("posted_ts") for job in jobs], unit="s", utc=True, errors="coerce"
        )
        ages = (now - stamps).total_seconds() / 3600

        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("Jobs", len(jobs))
        metric_b.metric("Remote", sum(1 for j in jobs if j.get("remote")))
        metric_c.metric("With salary", sum(1 for j in jobs if j.get("salary")))
        metric_d.metric("Median age", f"{int(pd.Series(ages).median())} h" if len(ages) else "—")

        st.markdown("**Freshness** — how old are these postings?")
        buckets = pd.cut(
            pd.Series(ages).dropna(),
            bins=[-0.1, 6, 12, 24, 48, 168, 100000],
            labels=["<6h", "6-12h", "12-24h", "1-2d", "2-7d", ">7d"],
        )
        counts = buckets.value_counts().sort_index()
        if len(counts):
            st.bar_chart(counts, height=220)

        st.markdown("**Salary**")
        salary_rows = [
            {"Title": j.get("title"), "Company": j.get("company"), "Salary": j["salary"]}
            for j in jobs
            if j.get("salary")
        ]
        if salary_rows:
            st.dataframe(pd.DataFrame(salary_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("None of these postings disclose a salary.")

render_html(
    """
    <div class="footer-note">
        CareerLens · live data may change. Verify every detail on the employer's page.
    </div>
    """
)
