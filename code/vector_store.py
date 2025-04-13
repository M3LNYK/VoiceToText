# vector_store.py
from pathlib import Path
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


class JournalVectorStore:
    def __init__(self, db_path="journal_vectors"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.db_path))

        # Create collections if they don't exist
        self.journal_collection = self.client.get_or_create_collection(
            "journal_entries"
        )
        self.entity_collection = self.client.get_or_create_collection("entities")

        # Load embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_journal_entry(self, entry_text, date_str, metadata=None):
        """Add a journal entry to the vector store"""
        if metadata is None:
            metadata = {}

        metadata["date"] = date_str
        metadata["type"] = "journal_entry"

        # Generate embedding directly with sentence-transformers
        embedding = self.embedding_model.encode(entry_text)

        # Add to collection
        self.journal_collection.add(
            ids=[f"entry_{date_str}"],
            embeddings=[embedding.tolist()],
            documents=[entry_text],
            metadatas=[metadata],
        )

    def find_similar_entries(self, query_text, limit=3):
        """Find entries similar to the query text"""
        # Generate embedding for query
        query_embedding = self.embedding_model.encode(query_text)

        # Query the collection
        results = self.journal_collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=limit
        )

        return results
