import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --- Multi-Format Document Loaders ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# --- 1. Load Approved Departments from JSON Config ---
with open("config.json", "r", encoding="utf-8") as file:
    config = json.load(file)
departments = config["approved_departments"]

# --- 2. Base Configuration ---
# Update this path if your shared Google Drive has a different folder name!
BASE_DRIVE_PATH = "G:/Shared drives/TFF_Docs" 
PERSIST_DIRECTORY = "./local_chroma_db"

# Initialize your local embedding model via LM Studio
embeddings = OpenAIEmbeddings(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio", 
    model="nomic-embed-text-v1.5",
    check_embedding_ctx_length=False 
)

all_processed_chunks = []

print("Starting automated Google Drive document ingestion...")

# --- 3. Loop Through Each Department Folder ---
for dept in departments:
    # Build the path to the specific department folder (e.g., G:/Shared drives/TFF_Docs/HR)
    dept_folder_path = os.path.join(BASE_DRIVE_PATH, dept)
    
    if os.path.exists(dept_folder_path):
        print(f"\nScanning files for department: '{dept}'...")
        
        dept_documents = []
        
        # Scan the folder for all files
        for filename in os.listdir(dept_folder_path):
            file_path = os.path.join(dept_folder_path, filename)
            
            if os.path.isfile(file_path):
                try:
                    # Route the file to the correct LangChain loader
                    if filename.lower().endswith(".pdf"):
                        dept_documents.extend(PyPDFLoader(file_path).load())
                    
                    elif filename.lower().endswith(".txt"):
                        dept_documents.extend(TextLoader(file_path, encoding="utf-8").load())
                        
                    elif filename.lower().endswith(".docx"):
                        dept_documents.extend(Docx2txtLoader(file_path).load())
                    
                    else:
                        print(f"  -> Skipping unsupported file type: {filename}")
                        
                except Exception as e:
                    print(f"  -> Error reading {filename}: {e}")
        
        if dept_documents:
            print(f"-> Found {len(dept_documents)} document(s) in {dept}. Splitting into chunks...")
            
            # Split documents into small readable blocks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            dept_chunks = text_splitter.split_documents(dept_documents)
            
            # Dynamically tag every chunk with its matching department name
            for chunk in dept_chunks:
                chunk.metadata["department"] = dept
            
            all_processed_chunks.extend(dept_chunks)
            print(f"-> Successfully processed {len(dept_chunks)} chunks for {dept}.")
        else:
            print(f"-> No readable documents found in '{dept}' folder.")
    else:
        print(f"\nWarning: Folder not found for '{dept}' at path: {dept_folder_path}")

# --- 4. Commit Everything to ChromaDB ---
if all_processed_chunks:
    print(f"\nSaving a total of {len(all_processed_chunks)} chunks to ChromaDB...")
    
    vector_db = Chroma.from_documents(
        documents=all_processed_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("Database sync complete! The AI's memory is fully up to date.")
else:
    print("\nNo new documents were found across any departments. Database remains unchanged.")
