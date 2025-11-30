from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm  # For progress bar

# Load environment variables
load_dotenv()

print("Loading documents...")
loader = PyPDFDirectoryLoader("data/")
docs = loader.load()
print(f"Loaded {len(docs)} documents")

print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} chunks")

print("Initializing embeddings...")
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name=os.getenv("AWS_REGION")
)

print("Embedding chunks in batches...")
batch_size = 10
all_embeddings = []
for i in tqdm(range(0, len(splits), batch_size)):
    batch = splits[i:i+batch_size]
    try:
        batch_embeddings = embeddings.embed_documents([doc.page_content for doc in batch])
        all_embeddings.extend(batch_embeddings)
    except Exception as e:
        print(f"Error embedding batch {i//batch_size}: {e}")
        time.sleep(5)  # Wait before retrying
    time.sleep(0.5)  # Add a short delay between batches

print("Creating vector store...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"✅ Ingestion complete! Processed {len(docs)} docs into {len(splits)} chunks")