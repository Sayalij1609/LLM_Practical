import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ═══════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR  = "./chroma_db"
COLLECTION  = "pdf_collection"
TOP_K       = 3

# ═══════════════════════════════════════════════
#  STEP 1 — Load & Split PDF
# ═══════════════════════════════════════════════
def load_and_split(pdf_path: str):
    print(f"\n📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"   → {len(docs)} page(s) found")

    # ✅ FIX 1: Smaller chunks = more precise retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # reduced from 1000
        chunk_overlap=50,      # reduced from 200
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"   → {len(chunks)} chunks created")
    return chunks


# ═══════════════════════════════════════════════
#  STEP 2 — Build Vector Store
# ═══════════════════════════════════════════════
def build_vectorstore(chunks):
    print(f"\n🧠 Embedding with Ollama ({EMBED_MODEL})...")
    print("   (This may take a minute...)")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )
    print(f"   ✅ Embeddings saved to {CHROMA_DIR}/")
    return vectorstore


# ═══════════════════════════════════════════════
#  STEP 3 — Load Existing Vector Store
# ═══════════════════════════════════════════════
def load_existing_vectorstore():
    print(f"\n♻️  Loading existing vector store from {CHROMA_DIR}/")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )


# ═══════════════════════════════════════════════
#  STEP 4 — Query PDF
# ═══════════════════════════════════════════════
def query_pdf(vectorstore, query: str):
    """Returns top-k relevant unique text chunks."""

    # ✅ FIX 2: Use MMR (Maximum Marginal Relevance)
    # This ensures diverse results instead of repetitive ones
    retriever = vectorstore.as_retriever(
        search_type="mmr",              # changed from "similarity"
        search_kwargs={
            "k": TOP_K,
            "fetch_k": 10,              # fetch 10, return best 3 diverse ones
            "lambda_mult": 0.7          # 0=max diversity, 1=max similarity
        },
    )
    results = retriever.invoke(query)

    # ✅ FIX 3: Remove duplicate chunks
    seen = set()
    unique_results = []
    for doc in results:
        # Use first 100 chars as key to detect duplicates
        key = doc.page_content[:100].strip()
        if key not in seen:
            seen.add(key)
            unique_results.append(doc.page_content)

    return unique_results


# ═══════════════════════════════════════════════
#  STEP 5 — Interactive Question Loop
# ═══════════════════════════════════════════════
def interactive_loop(vectorstore):
    print("\n" + "=" * 55)
    print("   PDF Query System — Powered by Ollama (Local AI)")
    print("   Ask anything about your PDF document")
    print("   Type 'exit' or 'quit' to stop")
    print("=" * 55)

    while True:
        print()
        query = input("Enter your question: ").strip()

        if query.lower() in ("exit", "quit", "q"):
            print("\n👋 Goodbye!")
            break

        if query == "":
            print("⚠️  Please enter a question.")
            continue

        print(f"\nSearching PDF for: '{query}'")
        results = query_pdf(vectorstore, query)

        if not results:
            print("❌ No relevant chunks found. Try rephrasing.")
            continue

        print(f"\nTop {len(results)} relevant chunks retrieved:")
        for idx, res in enumerate(results):
            print(f"\nChunk {idx + 1}:\n{res}\n")

        print("-" * 55)


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════
if __name__ == "__main__":

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "agentic_ai.pdf"

    if not os.path.exists(pdf_path):
        print(f"\n❌ PDF not found: '{pdf_path}'")
        print("   Usage: python pdf_query_ollama.py your_file.pdf")
        sys.exit(1)

    if os.path.exists(CHROMA_DIR):
        vectorstore = load_existing_vectorstore()
    else:
        chunks = load_and_split(pdf_path)
        vectorstore = build_vectorstore(chunks)

    interactive_loop(vectorstore)