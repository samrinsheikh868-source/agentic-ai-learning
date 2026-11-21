from dotenv import load_dotenv
from langchain_groq import ChatGroq

# load_dotenv()
#
# llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
#
# response = llm.invoke("Explain agentic AI in one simple line")

# print(response.content)

curl -s -o /dev/null -w "%{http_code}\n" \
  -H "x-api-key: <asli_key>" \
  "https://api.smith.langchain.com/sessions?limit=1"
