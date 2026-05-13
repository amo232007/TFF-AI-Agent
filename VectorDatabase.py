from langchain_huggingface import HuggingFaceEmbeddings

# This downloads a small, fast, and popular open-source embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma 

# 1. Initialize your LLM and Vector DB
LM_STUDIO_URL = "http://localhost:1234/v1"
llm = ChatOpenAI(
    openai_api_base=LM_STUDIO_URL,
    openai_api_key="lm-studio",
    temperature=0.3
)

# Assuming 'embeddings' is already defined in your code
vector_db = Chroma(persist_directory="./my_local_vectordb", embedding_function=embeddings)

# 2. Define your list of departments exactly as they appear in your metadata
departments = [
    "marketing", 
    "business development", 
    "production", 
    "accounting", 
    "HR", 
    "QA", 
    "engineering", 
    "warehousing"
]

# 3. Create a dictionary to store a dedicated retriever for each department
department_retrievers = {}

for dept in departments:
    dept_filter = {"department": dept} 
    department_retrievers[dept] = vector_db.as_retriever(search_kwargs={"k": 3, "filter": dept_filter})

# 4. Example Usage:
# If you want to search only within the HR department's documents:
# hr_retriever = department_retrievers["HR"]
# hr_docs = hr_retriever.invoke("What is the standard onboarding process?")
# If you want to search only within Engineering:
# engineering_retriever = department_retrievers["engineering"]
# eng_docs = engineering_retriever.invoke("Where are the deployment logs stored?")

