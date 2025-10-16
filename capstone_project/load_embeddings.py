# Load pre-computed embeddings and vector store
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

def load_vector_store(save_dir="vector_store"):
    """
    Load pre-computed vector store from disk.
    
    Returns:
        dict: Contains 'index', 'metadata_list', 'texts', 'embedding_model'
    """
    
    # Paths to your saved files
    faiss_index_path = os.path.join(save_dir, "legal_acts_index.faiss")
    metadata_path = os.path.join(save_dir, "metadata.pkl")
    texts_path = os.path.join(save_dir, "texts.pkl")
    
    # Check if files exist
    missing_files = []
    for path, name in [(faiss_index_path, "FAISS index"), (metadata_path, "metadata"), (texts_path, "texts")]:
        if not os.path.exists(path):
            missing_files.append(f"- {name}: {path}")
    
    if missing_files:
        print("❌ Missing files!")
        for missing in missing_files:
            print(missing)
        print("\nRun the embedding process first to create these files.")
        return None
    
    # Load the vector store
    print("🔄 Loading pre-computed vector store...")
    
    # Load FAISS index
    index = faiss.read_index(faiss_index_path)
    
    # Load metadata and texts
    with open(metadata_path, "rb") as f:
        metadata_list = pickle.load(f)
    
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)
    
    # Load embedding model (same one used for indexing)
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    print(f"✅ Loaded vector store:")
    print(f"- Index: {index.ntotal} vectors")
    print(f"- Metadata: {len(metadata_list)} records")
    print(f"- Texts: {len(texts)} documents")
    
    # Verify everything matches
    if len(metadata_list) != len(texts) or len(texts) != index.ntotal:
        print("⚠️  Warning: Counts don't match! This might cause issues.")
    else:
        print("✅ All counts match - vector store is ready!")
    
    return {
        'index': index,
        'metadata_list': metadata_list,
        'texts': texts,
        'embedding_model': embedding_model
    }

def retrieve_texts(vector_store, query, k=5):
    """
    Retrieve top-k relevant documents from loaded FAISS index.
    
    Args:
        vector_store: Dict returned by load_vector_store()
        query: Search query string
        k: Number of results to return
    
    Returns:
        list: List of results with score, text, and metadata
    """
    if vector_store is None:
        print("❌ Vector store not loaded!")
        return []
    
    index = vector_store['index']
    texts = vector_store['texts']
    metadata_list = vector_store['metadata_list']
    embedding_model = vector_store['embedding_model']
    
    # Encode query and search
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "score": float(distances[0][i]),
            "text": texts[idx],
            "metadata": metadata_list[idx],
        })
    return results

# Example usage
if __name__ == "__main__":
    # Load the vector store
    vs = load_vector_store()
    
    if vs is not None:
        # Test retrieval
        test_query = "acts related to revenue in Bombay"
        print(f"\n🔍 Testing retrieval with: '{test_query}'")
        
        results = retrieve_texts(vs, test_query, k=3)
        
        for i, r in enumerate(results):
            print(f"\n--- Result {i+1} (Score: {r['score']:.3f}) ---")
            print(f"Title: {r['metadata'].get('Short Title', 'N/A')}")
            print(f"Text preview: {r['text'][:200]}...")
        
        print(f"\n✅ Vector store loaded successfully with {vs['index'].ntotal} vectors!")
        print("You can now use 'vs' variable for retrieval in your code.")

