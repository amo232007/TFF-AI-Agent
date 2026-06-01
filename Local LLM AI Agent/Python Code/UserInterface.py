import json
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- 1. Load Approved Departments from JSON Config ---
with open("config.json", "r", encoding="utf-8") as file:
    config = json.load(file)
departments = config["approved_departments"]

# --- 2. Page Configuration ---
# Using your Postimages direct link ensures the logo works perfectly on any machine
st.set_page_config(page_title="TFF AI Agent", page_icon="https://i.postimg.cc/fW0RysMZ/TFF-Logo.jpg")
st.title("Toronto Fresh Food (TFF) AI Agent")
st.subheader("Ask me any questions about your department at TFF!")

# --- 3. Sidebar: Department Filter ---
st.sidebar.title("Search Settings")
st.sidebar.markdown("Select a department to filter the documents the AI reads.")
selected_dept = st.sidebar.selectbox("Department Filter", departments)

# --- 4. Initialize Models & Database ---
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio", 
        model="qwen3.5-27b", 
        temperature=0.7
    )

@st.cache_resource
def get_vector_db():
    # Must perfectly match the embedding setup used in the ingestion script!
    embeddings = OpenAIEmbeddings(
        openai_api_base="http://localhost:1234/v1",
        openai_api_key="lm-studio", 
        model="nomic-embed-text-v1.5",
        check_embedding_ctx_length=False 
    )
    return Chroma(persist_directory="./local_chroma_db", embedding_function=embeddings)

llm = get_llm()
vector_db = get_vector_db()

# --- 5. Manage Chat History in Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# --- 6. Handle User Input & RAG Logic ---
if prompt := st.chat_input("Ask a question based on your documents..."):
    
    # Add user message to UI and history
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner(f"Searching {selected_dept} documents..."):
            # Retrieve relevant text chunks matching the prompt AND the active department metadata
            retriever = vector_db.as_retriever(
                search_kwargs={
                    "k": 3, 
                    "filter": {"department": selected_dept}
                }
            )
            context_docs = retriever.invoke(prompt)
            
            # Combine retrieved text blocks into one single string
            context_text = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
            
            # Create the hidden system instructions feeding context to the AI
            system_instruction = f"""You are a helpful AI assistant for the {selected_dept} department at Toronto Fresh Food (TFF). 
            Use the following retrieved context from the company database to answer the user's question. 
            If the answer is not contained within the context, simply say that you do not know. 
            
            Retrieved Context:
            {context_text}
            """
            
            # Prepend system instruction to the active ongoing chat conversation
            full_messages = [SystemMessage(content=system_instruction)] + st.session_state.messages
            
        with st.spinner("Generating answer..."):
            # Call LM Studio with the combined prompt, history, and context
            response = llm.invoke(full_messages)
            
            # Render response onto the UI
            response_placeholder.markdown(response.content)
            
            # Optional dropdown feature allowing users to view source documents transparency
            if context_docs:
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(context_docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.write(doc.page_content)
                        st.caption(f"Metadata: {doc.metadata}")
                        st.divider()
        
    # Append the AI's response to the history so it remembers context next turn
    st.session_state.messages.append(AIMessage(content=response.content))
