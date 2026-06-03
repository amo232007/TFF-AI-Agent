from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Point LangChain to LM Studio
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen3-vl-a3b-30b-instruct",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful orchestration assistant."),
    ("user", "{input}")
])

chain = prompt | llm
print(chain.invoke({"input": "Explain how you will manage my tasks."}))

