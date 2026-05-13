import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- 1. Page Configuration ---
st.set_page_config(page_title="TFF AI Agent", page_icon="TFF Logo.jpg")
st.title("Toronto Fresh Food (TFF) AI Agent. Ask me any questions about your department at TFF!")

# --- 2. Sidebar: Department Filter ---
st.sidebar.title("Search Settings")
st.sidebar.markdown("Select a department to filter the documents the AI reads.")
departments = [
    "engineering", 
    "HR", 
    "marketing", 
    "sales", 
    "QA", 
    "accounting", 
    "business Development", 
    "production", 
    "warehousing"
]
selected_dept = st.sidebar.selectbox("Department Filter", departments)

# --- 3. Initialize Models & Database ---
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio", 
        model="qwen2.5-vl-7b-instruct", 
        temperature=0.7
    )

@st.cache_resource
def get_vector_db():
    # Make sure this perfectly matches the embedding setup we built for ingestion!
    embeddings = OpenAIEmbeddings(
        openai_api_base="http://localhost:1234/v1",
        openai_api_key="lm-studio", 
        model="nomic-embed-text-v1.5",
        check_embedding_ctx_length=False 
    )
    # Point this to the folder where your database was saved
    return Chroma(persist_directory="./local_chroma_db", embedding_function=embeddings)

llm = get_llm()
vector_db = get_vector_db()

# --- 4. Manage Chat History in Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# --- 5. Handle User Input & RAG Logic ---
if prompt := st.chat_input("Ask a question based on your documents..."):
    
    # Add user message to UI and session state
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner(f"Searching {selected_dept} documents..."):
            # A. Retrieve relevant document chunks based on the user's prompt and selected department
            retriever = vector_db.as_retriever(
                search_kwargs={
                    "k": 3, 
                    "filter": {"department": selected_dept}
                }
            )
            context_docs = retriever.invoke(prompt)
            
            # Combine the retrieved text chunks into a single string
            context_text = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
            
            # B. Create a hidden System Message that feeds the context to the AI
            system_instruction = f"""You are a helpful AI assistant for the {selected_dept} department. 
            Use the following retrieved context from the company database to answer the user's question. 
            If the answer is not contained within the context, simply say that you do not know. 
            
            Retrieved Context:
            {context_text}
            """
            
            # Combine the system instruction with the ongoing chat history
            full_messages = [SystemMessage(content=system_instruction)] + st.session_state.messages
            
        # C. Call the LLM with the context and history
        with st.spinner("Generating answer..."):
            response = llm.invoke(full_messages)
            
            # Display response
            response_placeholder.markdown(response.content)
            
            # Optional: Add a dropdown menu to let the user see exactly which documents the AI read
            if context_docs:
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(context_docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.write(doc.page_content)
                        st.caption(f"Metadata: {doc.metadata}")
                        st.divider()
        
    # Add AI message to session state to remember it for the next turn
    st.session_state.messages.append(AIMessage(content=response.content))
