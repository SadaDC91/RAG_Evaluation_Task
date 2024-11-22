# RAG_Evaluation_Task
A short program that performs a basic RAG evaluation

# Retrieval-Augmented Generation (RAG) Evaluation

It evaluates the retrieval component of a Retrieval-Augmented Generation (RAG) system. RAG combines document retrieval with a text generation model. Accurate retrieval is critical for generating contextually relevant responses.

## Types of RAG Evaluation
1. Retrieval Evaluation:
   - Focuses on evaluating the relevance of retrieved documents.
   - Metrics: Precision@k, Recall@k, Mean Reciprocal Rank (MRR), etc.
2. Generation Evaluation:
   - Measures the quality of generated text based on retrieved documents.
   - Metrics: BLEU, ROUGE, BERTScore, etc.
3. End-to-End Evaluation:
   - Directly evaluates the RAG system's output for correctness and relevance.
   - Metrics: Human evaluation or task-specific benchmarks.

## Chosen Evaluation for my model here: Retrieval Evaluation
The evaluation focuses on retrieval because the quality of retrieved documents is foundational to the RAG framework's success. **Precision@1** and **MRR** are used as metrics for this evaluation.


