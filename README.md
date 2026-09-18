# Agentic AI Assistant

An AI assistant built with **LangGraph**, **LangChain**, a local **Ollama LLM**, and a Persian **RAG system**.

The project is designed as a practical Agentic RAG assistant that can:
- answer questions using Persian company documents,
- decide when document retrieval is needed,
- evaluate whether retrieved documents are relevant,
- rewrite unsuccessful search queries and retry retrieval,
- perform calculations,
- retrieve customer debt information,
- send emails immediately,
- schedule emails for future times,
- maintain conversation state using LangGraph memory.

## Architecture

```text
                         ┌──────────────────┐
                         │       User       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   LangGraph      │
                         │      Agent       │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              General request             Document request
                    │                           │
                    ▼                           ▼
              Local LLM                 search_persian_docs
                    │                           │
                    │                           ▼
                    │                   Retrieval Evaluator
                    │                           │
                    │                 ┌─────────┴─────────┐
                    │                 │                   │
                    │              Relevant          Not relevant
                    │                 │                   │
                    │                 ▼                   ▼
                    │          Continue action      Rewrite query
                    │                                      │
                    │                                      ▼
                    │                              Second retrieval
                    │                                      │
                    │                                      ▼
                    │                                  Evaluator
                    │
                    └──────────────────────┬────────────────
                                           │
                                           ▼
                                      Final action
```

## Main Technologies

- **Python**
- **LangGraph**
- **LangChain**
- **Ollama**
- **Qwen2.5 7B**
- **FAISS**
- **BM25**
- **Sentence Transformers**
- **APScheduler**
- **SMTP**
- **InMemorySaver**

## RAG System

The assistant uses the existing Persian RAG project as its document knowledge source.

The RAG pipeline is:

```text
PDF documents
     ↓
PDF Loader
     ↓
Persian Text Cleaner
     ↓
Text Chunker
     ↓
Multilingual Embeddings
     ↓
FAISS + BM25
     ↓
Hybrid Retriever
     ↓
search_persian_docs tool
     ↓
LangGraph Agent
```

The retriever combines semantic and keyword search:

```text
Hybrid Retrieval
    ├── FAISS semantic search
    └── BM25 keyword search
```

This allows the agent to search Persian company documents using both semantic similarity and exact keyword matching.

## Agent Tools

The current agent has several tools.

### Calculator

Performs basic operations:

```text
add
subtract
multiply
divide
```

### Customer Debt

Retrieves the debt amount associated with a customer from the current sample data.

### Persian Document Search

```text
search_persian_docs
```

Searches the Persian company documents and returns relevant document chunks together with metadata such as source and page.

The agent is instructed not to answer company-document questions from its own knowledge when the required information should come from the documents.

### Email

Two email actions are currently supported:

```text
send_email
schedule_email
```

The first sends an email immediately.

The second schedules an email for a future time using APScheduler.

## Retrieval Evaluation

One of the main features of the project is a separate retrieval evaluation step.

After the RAG tool returns documents:

```text
User question
      ↓
Document retrieval
      ↓
Retrieval evaluator
      ↓
YES / NO
```

The evaluator checks whether the retrieved documents contain enough information to answer the **information part** of the user's request.

It does not evaluate email-related actions.

For example, a question such as:

```text
انبار کارش چیه؟
```

can be considered relevant when the retrieved documents contain information such as the responsibilities of an `انباردار`.

The evaluator also distinguishes different roles when they are genuinely different, rather than relying only on keyword matching.

## Query Rewriting and Retry

If retrieval is evaluated as insufficient, the agent can rewrite the search query and perform one additional retrieval attempt.

```text
Initial retrieval
      ↓
Evaluator
      ↓
NO
      ↓
Query rewriting
      ↓
Second retrieval
      ↓
Evaluator
```

The rewritten query focuses only on the information needed from the documents and removes unrelated action details such as email instructions.

If the second retrieval is still insufficient, the agent returns:

```text
اطلاعات کافی برای پاسخ به این سؤال در اسناد موجود پیدا نشد.
```

## Agent Workflow

The current LangGraph workflow is approximately:

```text
START
  ↓
LLM
  ↓
Tools
  ├── search_persian_docs
  │        ↓
  │   evaluate_retrieval
  │        │
  │        ├── relevant → continue_after_retrieval
  │        │
  │        └── not relevant → rewrite_query
  │                                  ↓
  │                           retrieve_again
  │                                  ↓
  │                           evaluate_retrieval
  │
  └── email/action tools → END
```

The important design principle is that **retrieval and actions are separated**.

After successful retrieval, the agent moves to an action stage rather than blindly allowing the LLM to search the documents again. This prevents unnecessary retrieval loops.

## Conversation Memory

The graph uses:

```python
InMemorySaver()
```

with a `thread_id` to maintain conversation state during the running application.

The state currently includes fields such as:

```text
messages
retrieval_relevant
evaluation_reason
rewritten_query
retry_count
email_sent
```

`InMemorySaver` is suitable for the current development stage because the goal is to understand and build the agent workflow. Persistent memory can be added later.

## Current Status

### Implemented

- [x] LangGraph agent
- [x] Local Ollama LLM
- [x] Tool calling
- [x] Persian RAG integration
- [x] FAISS + BM25 hybrid retrieval
- [x] Retrieval relevance evaluation
- [x] Query rewriting
- [x] One retrieval retry
- [x] Retrieval fallback
- [x] Calculator tool
- [x] Customer debt tool
- [x] Immediate email sending
- [x] Scheduled email sending
- [x] Conversation memory with InMemorySaver

### Planned

- [ ] Telegram publishing
- [ ] Scheduled Telegram posts
- [ ] Local image generation
- [ ] More external tools and actions
- [ ] Persistent scheduler storage
- [ ] More robust production deployment

## Why Agentic RAG?

A traditional RAG system mainly follows:

```text
Question
   ↓
Retrieve
   ↓
Generate answer
```

This project adds an agent layer:

```text
Question
   ↓
Agent
   ↓
Decide whether a tool is needed
   ↓
Retrieve / Calculate / Send email / Schedule
   ↓
Evaluate retrieval when necessary
   ↓
Retry or continue
   ↓
Complete the request
```

The goal is to demonstrate how an LLM can work as a decision-making layer around RAG and external tools instead of being used only as a question-answering model.

## Local and Free

The core AI components are designed to run locally:

- LLM: Ollama
- Model: Qwen2.5 7B
- Embeddings: local Sentence Transformers model
- Vector search: FAISS
- Keyword search: BM25
- Workflow: LangGraph
- Scheduler: APScheduler

No paid LLM API is required for the current implementation.

## Project Direction

The project is intentionally being developed incrementally.

Current focus:

```text
Agent + RAG + Tools + Memory
```

Future direction:

```text
Agentic RAG
      +
Content Generation
      +
External Actions
      +
Scheduling
      +
Local Media Generation
```

## Author

**Zahra Hemmati**

AI / BI / Data-oriented developer building practical AI systems with Python, RAG, LangGraph, and local LLMs.
