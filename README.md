# System Savant - CO Tutor Bot using Llama2 RAG Implementation 
Your very own Computer Organization Tutor.

This project is a tutor for Computer Organization and its related domains. The chatbot utilizes state-of-the-art LLM Llama2 and made it domain-specific using Advanced RAG Implementation.

## Tech Stacks Used
1. Llamaindex: LLamaindex is a data framework that connects your applications with large language models (LLMs) by providing tools for data loading, storage, and orchestration.
2. Langchain: Used it for converting texts into embeddings for the vector store.
3. ChromaDB: ChromaDB is an open-source vector database specifically designed for applications powered by Large Language Models (LLMs), allowing developers to efficiently store, manage, and query text data as embeddings for tasks like text search, analysis, and summarization.
4. HuggingFace: Hugging Face offers pre-trained NLP models, a powerful library for building custom models.
5. StreamLit: Streamlit is an open-source Python framework for data scientists and AI/ML engineers to deliver dynamic data apps.

## Project Structure

```
SystemSavant
│   .gitignore
│   app.py
│   Dockerfile
│   LICENSE
│   params.yaml
│   README.md
│   requirements.txt
│   setup.py
│   template.py
│
├───.github
│   └───workflows
│           .gitkeep
│
├───artifacts
│   └───data
│           IODEVICE.pdf
│
├───config
│       config.yaml
│       params.yaml
│
├───logs
│       running_logs.log
│
├───research
│       trails.ipynb
│
└───src
    └───SystemSavant
        │   __init__.py
        │
        ├───components
        │       llm.py
        │       __init__.py
        │
        ├───config
        │       configuration.py
        │       __init__.py
        │
        ├───constants
        │       __init__.py
        │
        ├───entity
        │       __init__.py
        │
        ├───logging
        │       __init__.py
        │
        ├───pipeline
        │       inference.py
        │       __init__.py
        │
        └───utils
                common.py
                __init__.py
```
   
