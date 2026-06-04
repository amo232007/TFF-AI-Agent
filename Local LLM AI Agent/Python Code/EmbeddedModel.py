import os
import json
import base64
import pandas as pd
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --- 1. System Configurations ---
with open("config.json", "r", encoding="utf-8") as file:
    config = json.load(file)
departments = config["approved_departments"]

BASE_DRIVE_PATH = "G:/Shared drives/TFF_Docs" 
PERSIST_DIRECTORY = "./local_chroma_db"

API_BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"

# Initialize standard embeddings for ChromaDB
embeddings = OpenAIEmbeddings(
    openai_api_base=API_BASE_URL,
    openai_api_key=API_KEY, 
    model="nomic-embed-text-v1.5",
    check_embedding_ctx_length=False 
)

# Initialize OpenAI Client to talk to your Qwen Vision Model
vision_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# --- 2. Helper Functions for New File Types ---

def process_spreadsheet(file_path):
    """Converts a spreadsheet into a Markdown table so the AI can read the grid perfectly."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Convert DataFrame to markdown text (requires the 'tabulate' library)
        return df.to_markdown(index=False)
    except Exception as e:
        print(f"Error reading spreadsheet {file_path}: {e}")
        return ""

def process_image(file_path):
    """Sends the image to the Vision Model and returns a highly detailed text description."""
    try:
        with open(file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = vision_client.chat.completions.create(
            model="qwen3-vl-a3b-30b-instruct", # The exact name in LM Studio doesn't strictly matter
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Describe this image in extreme detail. If it is a spreadsheet, chart, or warehouse schematic, extract all the text, numbers, and data relationships exactly as written."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return ""

# --- 3. The Main Scanning Engine ---
all_processed_chunks = []
print("Starting Multimodal Google Drive ingestion...")

for dept in departments:
    dept_folder_path = os.path.join(BASE_DRIVE_PATH, dept)
    
    if os.path.exists(dept_folder_path):
        print(f"\nScanning files for department: '{dept}'...")
        dept_documents = []
        
        # os.walk allows us to check every file type individually, even in sub-folders
        for root, dirs, files in os.walk(dept_folder_path):
            for file in files:
                ext = file.lower().split('.')[-1]
                file_path = os.path.join(root, file)
                
                # Handle PDFs
                if ext == 'pdf':
                    loader = PyPDFLoader(file_path)
                    dept_documents.extend(loader.load())
                
                # Handle Spreadsheets
                elif ext in ['csv', 'xlsx']:
                    markdown_data = process_spreadsheet(file_path)
                    if markdown_data:
                        doc = Document(page_content=markdown_data, metadata={"source": file_path, "type": "spreadsheet"})
                        dept_documents.append(doc)
                
                # Handle Images
                elif ext in ['jpg', 'jpeg', 'png']:
                    print(f" -> Calling Vision Model to analyze image: {file}...")
                    image_description = process_image(file_path)
                    if image_description:
                        doc = Document(page_content=image_description, metadata={"source": file_path, "type": "image_description"})
                        dept_documents.append(doc)

        # --- 4. Chunk and Tag the Data ---
        if dept_documents:
            print(f"-> Found {len(dept_documents)} valid files in {dept}. Splitting and tagging...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            dept_chunks = text_splitter.split_documents(dept_documents)
            
            for chunk in dept_chunks:
                chunk.metadata["department"] = dept
            
            all_processed_chunks.extend(dept_chunks)
        else:
            print(f"-> No valid files found in '{dept}'.")
    else:
        print(f"\nWarning: Folder not found for '{dept}'")

# --- 5. Commit Everything to ChromaDB ---
if all_processed_chunks:
    print(f"\nSaving a total of {len(all_processed_chunks)} chunks to ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=all_processed_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("Database sync complete! The AI's memory is fully updated with multimodal data.")
else:
    print("\nNo data found across any departments. Database remains unchanged.")
