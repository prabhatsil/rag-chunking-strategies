# RAG Chunking Strategies

A hands-on lab for comparing document chunking strategies for Retrieval-Augmented Generation (RAG). Each strategy is a standalone script that splits a sample document and prints the resulting chunks, so you can see how the choice of splitter, chunk size and overlap changes what ends up in your vector store.

## Why chunking matters

In a RAG pipeline, chunking happens before embedding:

```
Documents → Loader → Chunking → Embedding Model → Vector DB → Top-K Chunks → LLM → Answer
```

Chunks that are too large dilute the embedding and waste context window. Chunks that are too small lose the surrounding meaning. Splitting in the wrong place cuts a sentence, a table or a list in half. The strategies here trade simplicity for preserving meaning in different ways.

## Strategies

| # | Strategy | Status | How it splits |
|---|----------|--------|---------------|
| 01 | [Fixed size](chunkers/01-fixed_size/) | ✅ Implemented | `CharacterTextSplitter` on newlines up to a fixed character count, with overlap |
| 02 | [Recursive](chunkers/02-recursive/) | ✅ Implemented | `RecursiveCharacterTextSplitter`, trying paragraph → line → sentence → word → character |
| 03 | [Token-aware](chunkers/03-token-aware/) | ✅ Implemented | Recursive splitting measured in **tokens**, using `tiktoken` (`cl100k_base`) or a Hugging Face tokenizer |
| 04 | [Sentence](chunkers/04-sentence/) | 🚧 Planned | Split on sentence boundaries (NLTK) |
| 05 | [Agentic](chunkers/05-agentic/) | 🚧 Planned | An LLM decides where chunk boundaries go |
| 06 | [Context-enriched](chunkers/06-context_enriched/) | 🚧 Planned | Each chunk carries a summary of its neighbours |
| 07 | [Contextual retrieval](chunkers/07-contextual_retrieval/) | 🚧 Planned | An LLM prepends document-level context to each chunk before embedding |
| – | [Document structure](chunkers/document_structure/) | 🚧 Planned | Split on Markdown/HTML headings and sections |
| – | [Semantic](chunkers/semantic/) | 🚧 Planned | Split where embedding similarity between sentences drops |
| – | [Sentence window](chunkers/sentence_window/) | 🚧 Planned | Embed single sentences, retrieve with surrounding sentences |
| – | [Parent-child](chunkers/parent_child/) | 🚧 Planned | Retrieve small chunks, return their larger parent |
| – | [Proposition](chunkers/proposition/) | 🚧 Planned | An LLM rewrites text into atomic, self-contained facts |
| – | [Late chunking](chunkers/late_chunking/) | 🚧 Planned | Embed the whole document first, then pool token embeddings per chunk |

## Project structure

```
rag-chunking-strategies/
├── chunkers/            # One folder per strategy (script + README)
├── data/
│   ├── documents/md/    # Sample Markdown documents used by the chunkers
│   ├── documents/pdf/   # Put your own PDFs here (git-ignored)
│   └── golden_set.jsonl # Question/answer pairs for evaluation (WIP)
├── evaluation/          # Retrieval metrics and retriever (WIP)
├── notebooks/           # Side-by-side comparison notebook (WIP)
├── results/             # Experiment outputs (git-ignored)
├── tests/
├── run_experiments.py   # Runs all strategies and compares them (WIP)
└── pyproject.toml
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <your-repo-url>
cd rag-chunking-strategies
uv sync              # creates .venv and installs the locked dependencies
```

To add a dependency, use `uv add <package>` rather than `pip install`, so it is recorded in `pyproject.toml` and `uv.lock`.

## Running a strategy

Each chunker reads from `data/documents/md/` and prints the chunks to the terminal:

```bash
uv run python chunkers/01-fixed_size/fixed_size.py
uv run python chunkers/02-recursive/recursive.py
uv run python chunkers/03-token-aware/token-aware.py
```

Change `chunk_size` and `chunk_overlap` in each script's `main()` to see how the chunks change.

The Hugging Face tokenizer option in the token-aware chunker downloads the tokenizer from the Hugging Face Hub the first time it runs. The "PyTorch was not found" warning can be ignored, because loading a tokenizer doesn't need PyTorch.

## Environment variables

None are needed yet. The planned LLM-based strategies (agentic, contextual retrieval, proposition) will need API keys. Copy `.env.example` to `.env` and fill in your keys there. `.env` is git-ignored.

## Tech stack

- [LangChain](https://python.langchain.com/) text splitters and document loaders
- [tiktoken](https://github.com/openai/tiktoken) and [🤗 Transformers](https://huggingface.co/docs/transformers) tokenizers for token counting
- [NLTK](https://www.nltk.org/) for sentence splitting
- [pypdf](https://pypdf.readthedocs.io/) for PDF loading
