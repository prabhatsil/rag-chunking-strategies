# RAG Lab

A hands-on lab for building, comparing and evaluating Retrieval-Augmented Generation (RAG) techniques.

## The architecture in one picture

```
                         RAG LAB
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
  01 Hybrid RAG       04 Agentic RAG       05 Graph RAG
        │                   │                   │
        │                   │                   │
        ▼                   ▼                   ▼
  02 Reranking         Agent Tools         Knowledge Graph
        │                   │                   │
        ▼                   │                   │
  03 Parent-Child           │                   │
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                    06 Multimodal RAG
                            │
                            ▼
                    ┌───────────────┐
                    │   Evaluation  │
                    └───────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
         Experiments                Observability
```

## Shared Infrastructure

```
                 SHARED INFRASTRUCTURE
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
    Documents        Embeddings          LLM
        │                │                │
        ▼                ▼                │
     Chunking       Vector Database       │
        │                │                │
        └────────────────┴────────────────┘
                         │
                         ▼
                  RAG Applications
```

## Ingestion Pipeline

```
                 Documents
                     │
                     ▼
              Document Loader
                     │
                     ▼
                  Chunking
                     │
                     ▼
              Text Chunks
                     │
                     ▼
             Embedding Model
                     │
                     ▼
              Vector Embeddings
                     │
                     ▼
                Vector DB
          ┌──────────┴──────────┐
          │                     │
       vector                metadata
       [0.12, ...]           text
                             source
                             page
                             chunk_id
```

## User Query Steps

```
 User Query
    │
    ▼
Embedding Model
    │
    ▼
Query Vector
    │
    ▼
Vector DB
    │
    ▼
Top-K Chunks
    │
    ▼
LLM
    │
    ▼
Answer
```
