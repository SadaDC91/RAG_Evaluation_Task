import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


queries = ["What is AI?", "Explain machine learning", "Define deep learning"]
documents = [
    "AI stands for Artificial Intelligence, enabling machines to mimic human intelligence.",
    "Machine learning is a subset of AI that focuses on using algorithms to learn from data.",
    "Deep learning is a subset of machine learning focusing on neural networks."
]
relevant_docs = [[0], [1], [2]]  # Ground truth for evaluation

# TF-IDF-based retrieval
def evaluate_retrieval(queries, documents, relevant_docs):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vectors = vectorizer.transform(queries)
    
    metrics = {"Precision@1": [], "MRR": []}
    for i, query_vector in enumerate(query_vectors):
        scores = cosine_similarity(query_vector, doc_vectors).flatten()
        ranked_indices = np.argsort(-scores)  # Descending order
        
        # Calculating Precision@1
        precision_at_1 = 1 if ranked_indices[0] in relevant_docs[i] else 0
        metrics["Precision@1"].append(precision_at_1)
        
        # Calculating Mean Reciprocal Rank (MRR)
        rr = 0
        for rank, idx in enumerate(ranked_indices, start=1):
            if idx in relevant_docs[i]:
                rr = 1 / rank
                break
        metrics["MRR"].append(rr)
        
        # Logging individual query results
        logger.info(f"Query {i+1}: Precision@1={precision_at_1}, Reciprocal Rank={rr}")
    
    # Aggregating metrics
    avg_precision = np.mean(metrics["Precision@1"])
    avg_mrr = np.mean(metrics["MRR"])
    logger.info(f"Average Precision@1: {avg_precision}")
    logger.info(f"Average MRR: {avg_mrr}")
    return avg_precision, avg_mrr

if __name__ == "__main__":
    evaluate_retrieval(queries, documents, relevant_docs)
