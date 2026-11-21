import os


import requests

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain .tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

JSEARCH_API_KEY = os.getenv ("JSEARCH_API_KEY")


@tool
def search_job (query: str):

    """
    Search for job posting based on the user's query.
    """

    url= "https://api.openwebninja.com/jsearch/search-v2"

    headers = {
        "X-API-Key": JSEARCH_API_KEY
    }

    params = {
        "query": query,
        "country": "in",
        "date_posted": "today"
    }

    response = requests.get(url,headers=headers, params=params)

    response.raise_for_status()

    jobs = response.json()["data"]["jobs"]

    return [
        {
            "title": j["job_title"],
            "company": j["employer_name"],
            "location": j["job_location"],
            "type": j["job_employment_type"],
            "posted": j["job_posted_at"],
            "apply_link": j["job_apply_link"],
            "description": (j["job_description"] or "")[:400],
        }
        for j in jobs[:5]
    ]

# print (search_job.invoke({"query":"Python Developer"}))

llm = ChatGroq (
    model= "openai/gpt-oss-20b", temperature=0
)

tools = [search_job]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are a job search assistant.

When the user asks for jobs, call search_job exactly once.
Then answer with a numbered list, one entry per job, in EXACTLY this format:

1. <title> || <company>
   - Job Title: <title>
   - Company Name: <company>
   - Location: <location>
   - Job Type: <type>
   - Posted Time: <posted>
   - Apply Link: <apply_link>

Rules:
- No markdown tables, no headings, no intro line, no closing commentary.
- Use only the data returned by the tool. Never invent a link.
- If a field is null, write "Not specified".
"""
)
# noinspection PyTypeChecker
user_query = agent.invoke(
    {
    "messages": [
        HumanMessage (
            content= """ Find data analyst jobs in india. for each job provide:
            
            - Job Title
            - Company Name
            - Location
            - Job Type
            - Posted Time
            - Apply Link"""

            )
    ]
})



print(user_query["messages"][-1].content)









