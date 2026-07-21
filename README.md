# KnowledgeFlow AI

**企业级智能知识库问答系统**  
**Retrieval-Augmented Knowledge Base Assistant**

KnowledgeFlow AI is a learning-oriented, portfolio-ready RAG project. It will turn local PDF, TXT, and Markdown documents into a searchable knowledge base and generate answers grounded in retrieved document content with source citations.

> Current status: project initialization. RAG, document processing, API, and model integration have not been implemented yet.

## Current progress

- [x] Create an isolated Python virtual environment
- [x] Add Git ignore rules for local and sensitive files
- [x] Add an incremental dependency manifest
- [x] Create the initial application and test directories
- [ ] Load and parse documents
- [ ] Split and embed document text
- [ ] Store and retrieve vectors
- [ ] Generate grounded answers with Ollama
- [ ] Expose the workflow through FastAPI
- [ ] Add automated tests and RAG evaluation

## Environment

- Windows
- PowerShell
- Python 3.14.6
- Python virtual environment: `venv`

## Local setup

Run the following commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The project currently has no third-party dependencies. Dependencies will be added only when their corresponding features are implemented.

## Current structure

```text
knowledgeflow-ai/
├── app/
│   └── __init__.py
├── tests/
│   └── .gitkeep
├── .gitignore
├── README.md
└── requirements.txt
```

## Planned core workflow

```text
Document upload
→ text extraction
→ text cleaning and chunking
→ embedding generation
→ vector storage
→ semantic retrieval
→ prompt construction
→ local LLM generation
→ answer with source citations
```

## Project principles

- Implement the core RAG workflow manually before introducing orchestration frameworks.
- Keep configuration, retrieval, generation, and API responsibilities separated.
- Prefer small, verifiable development steps over premature complexity.
- Report unsupported questions instead of generating unsupported claims.
