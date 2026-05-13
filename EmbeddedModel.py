from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. Load the Data ---
# This will automatically scan the folder, detect the file types, 
# and load all supported files it finds!
loader = DirectoryLoader('./FolderName', loader_kwargs={"strategy": "fast"}) 
documents = loader.load()

# ---> ADD YOUR METADATA HERE <---
# Loop through all the loaded documents from the folder
# and add your custom key-value pair to the metadata dictionary.
for doc in documents:
    doc.metadata["department"] = "DepartmentName"

print(f"Loaded {len(documents)} document(s) and assigned metadata.")

# --- 2. Chunk the Data ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       
    chunk_overlap=50,     
    separators=["\n\n", "\n", " ", ""]
)
# The splitter will automatically attach {"department": "DepartmentName"} to every resulting chunk!
chunks = text_splitter.split_documents(documents)
print(f"Split data into {len(chunks)} chunks.")

# --- 3. Initialize the Embedding Model via LM Studio ---
embeddings = OpenAIEmbeddings(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio", 
    model="nomic-embed-text-v1.5",
    check_embedding_ctx_length=False
)

# --- 4. Generate Vectors and Store in ChromaDB ---
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./local_chroma_db" 
)

print("Data successfully embedded and saved to ChromaDB with metadata!")
