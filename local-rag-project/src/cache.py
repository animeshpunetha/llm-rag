import numpy as np 
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from langchain_ollama import OllamaEmbeddings
import config

class SemanticCache:
    def __init__(self, host='localhost', port=6379, threshold=0.15):
        """
        Initializes the Redis connection and sets up the vector search index.
        threshold = 0.05 means questions must be 95% similar to trigger a cache hit.
        """
        self.redis_client = redis.Redis(host=host, port=port, decode_responses=False)
        self.threshold = threshold
        self.index_name = "rag_cache_index"

        # Initialize the exact same embedding model used for the ChromaDB chunks
        self.embeddings = OllamaEmbeddings(model = config.EMBEDDING_MODEL)

        self._setup_index()

    
    def _setup_index(self):
        """Creates the Redis index schema if it doesn't already exist."""

        try:
            self.redis_client.ft(self.index_name).info()
        except redis.exceptions.ResponseError:
            # Define the schema: A text field for the answer, a vector field for the embedded question
            schema = (
                TextField("answer"),
                VectorField(
                    "question_vector",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": 768,  # nomic-embed-text outputs 768 dimensions
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            )
            definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
            self.redis_client.ft(self.index_name).create_index(fields=schema, definition=definition)
            print("[Cache] Initialized new Redis vector index.")

    def check_cache(self, user_query:str):
        """Embeds the query and searches Redis for a highly similar past question."""
        # 1. Embed the incoming question
        query_vector = self.embeddings.embed_query(user_query)
        query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()

        # 2. Build the KNN search query looking for the top 1 closest match
        search_query = (
            Query("*=>[KNN 1 @question_vector $vec AS vector_score]")
            .sort_by("vector_score")
            .return_fields("vector_score", "answer")
            .dialect(2)
        )

        # 3. Execute search
        results = self.redis_client.ft(self.index_name).search(
            search_query, 
            query_params={"vec": query_vector_bytes}
        )

        if results.docs:
            distance = float(results.docs[0].vector_score)
            if distance <= self.threshold:
                print(f"[Cache Hit] Found similar query! (Distance: {distance:.4f})")
                answer = results.docs[0].answer
                return answer if isinstance(answer, str) else answer.decode('utf-8')
        
        return None

    def set_cache(self, user_query: str, answer: str):
        """Embeds the question and stores it alongside the generated answer in Redis."""
        query_vector = self.embeddings.embed_query(user_query)
        query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        
        # Create a unique key using a simple counter or hash
        cache_key = f"cache:{abs(hash(user_query))}"
        
        # Store in Redis
        mapping = {
            "answer": answer,
            "question_vector": query_vector_bytes
        }
        self.redis_client.hset(cache_key, mapping=mapping)

