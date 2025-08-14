# CLAUDE.local.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start using the provided script
chmod +x run.sh
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install/update dependencies (Intel Mac compatible)
uv sync

# Add new dependency
uv add package_name
```

### Environment Setup
- Create `.env` file in root directory with:
  ```
  ANTHROPIC_API_KEY=your_anthropic_api_key_here
  ```

## Intel Mac Compatibility

This version uses Intel Mac compatible dependencies:
- **Python**: 3.11+ (compatible with Intel and ARM Macs)
- **PyTorch**: 2.1.1 (has Intel Mac wheels)
- **Sentence Transformers**: 2.2.2 (compatible with PyTorch 2.1.1)

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials with a modular, service-oriented architecture:

### Core Components

1. **FastAPI Application** (`app.py`)
   - Main web server and API endpoints
   - Serves static frontend files
   - Handles CORS and proxy configuration

2. **RAG System** (`rag_system.py`)
   - Central orchestrator managing all components
   - Handles document ingestion and query processing
   - Coordinates between vector store, AI generator, and tools

3. **Vector Store** (`vector_store.py`)
   - ChromaDB-based semantic search using sentence transformers
   - Two collections: `course_catalog` (metadata) and `course_content` (chunks)
   - Smart course name resolution and filtering capabilities

4. **Document Processor** (`document_processor.py`)
   - Parses structured course documents with metadata
   - Sentence-based chunking with configurable overlap
   - Expected format: Course Title/Link/Instructor followed by "Lesson N:" sections

5. **AI Generator** (`ai_generator.py`)
   - Anthropic Claude integration with tool calling support
   - System prompt optimized for educational content
   - Handles conversation context and tool execution flow

6. **Search Tools** (`search_tools.py`)
   - Tool-based architecture for course content search
   - Semantic course name matching and lesson filtering
   - Extensible tool registry system

### Document Format Requirements

Course documents must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: Introduction
Lesson Link: [optional url]
[lesson content]

Lesson 1: Next Topic
[lesson content]
```

### Configuration

All settings centralized in `config.py`:
- **Chunk size**: 800 characters with 100 character overlap
- **Embedding model**: all-MiniLM-L6-v2
- **AI model**: claude-sonnet-4-20250514
- **Database**: ChromaDB in `./chroma_db`
- **Search**: Max 5 results, 2 conversation history turns

### Key Design Patterns

- **Tool-based search**: AI uses structured tools rather than direct vector queries
- **Session management**: Conversation context tracked by session ID
- **Modular architecture**: Each component has single responsibility
- **Smart name resolution**: Fuzzy matching for course names using vector similarity
- **Context-aware chunking**: Lesson metadata included in content chunks

### API Endpoints

- `POST /api/query`: Process user questions with optional session context
- `GET /api/courses`: Get course statistics and titles
- `/` (static): Frontend interface served from `../frontend/`

### Frontend Integration

Simple HTML/CSS/JS frontend served statically with development-friendly no-cache headers.

**NEW CHAT Button Feature:**
- Located at top of left sidebar above "Courses" section
- Styled to match existing UI elements (uppercase, same font size/color scheme)
- Clears current conversation without page reload
- Resets session state and shows welcome message
- Handles proper cleanup of both frontend and backend state

### Development Notes

- Uses `uv` for Python package management (requires Python 3.11+)
- ChromaDB handles persistence automatically
- Documents loaded on startup from `../docs/` directory
- Vector embeddings cached for performance
- Tool execution happens in single AI call with follow-up response