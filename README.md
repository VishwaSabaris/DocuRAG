# DocuRAG

> An AI-powered document analysis and Retrieval-Augmented Generation (RAG) system for asking questions and retrieving grounded answers from uploaded PDF documents.

DocuRAG combines document processing, vector-based retrieval, structured evidence extraction, answerability checking, and Large Language Models (LLMs) to provide reliable answers based on the contents of uploaded documents.

---

## 🚀 Features

- 📄 Upload and analyze PDF documents
- 🔎 Semantic document retrieval using vector search
- 🤖 Retrieval-Augmented Generation (RAG)
- 🧠 LLM-powered question answering
- 📚 Support for document-grounded answers
- ✅ Answerability checking before generating responses
- 🧩 Structured evidence extraction
- 📌 Document source references and citations
- 💬 ChatGPT-style conversational interface
- 🗂️ Recent chat history
- 📁 Recently uploaded documents
- 🌙 Modern dark-themed frontend
- 📱 Responsive UI for desktop, tablet, and mobile
- 🐳 Docker-based PostgreSQL setup
- 🧪 Evaluation and benchmarking framework
- 🔐 Environment-based configuration
- ⚡ FastAPI backend
- ⚛️ React + TypeScript frontend

---

## 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │      User / UI       │
                         │   React + TypeScript │
                         └──────────┬───────────┘
                                    │
                                    │ HTTP API
                                    ▼
                         ┌──────────────────────┐
                         │    FastAPI Backend   │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
             ┌────────────┐ ┌──────────────┐ ┌───────────────┐
             │ Document   │ │ Retrieval /  │ │ Answerability │
             │ Processing │ │ Vector Search│ │    Service    │
             └─────┬──────┘ └──────┬───────┘ └───────┬───────┘
                   │               │                 │
                   ▼               ▼                 ▼
             ┌────────────────────────────────────────────┐
             │          Structured Evidence Layer          │
             └─────────────────────┬──────────────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │         LLM          │
                         │  Answer Generation   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Grounded Response +  │
                         │ Document References  │
                         └──────────────────────┘
```

---

## 🧠 How DocuRAG Works

DocuRAG follows a Retrieval-Augmented Generation pipeline.

```text
PDF Upload
    │
    ▼
Document Processing
    │
    ▼
Text Extraction
    │
    ▼
Chunking
    │
    ▼
Embedding Generation
    │
    ▼
Vector Database
    │
    ▼
User Question
    │
    ▼
Query Processing
    │
    ▼
Relevant Evidence Retrieval
    │
    ▼
Answerability Check
    │
    ▼
Structured Evidence
    │
    ▼
LLM Answer Generation
    │
    ▼
Grounded Answer + Sources
```

The system is designed to ensure that answers are based on information retrieved from the uploaded documents rather than blindly generating responses from the language model.

---

## 🛠️ Technology Stack

### Backend
- Python
- FastAPI
- Uvicorn
- Pydantic
- PostgreSQL
- Vector Database / Vector Search
- LLM
- RAG architecture

### Frontend
- React
- TypeScript
- Vite
- CSS
- Lucide Icons

### Infrastructure
- Docker
- Docker Compose
- PostgreSQL
- Linux
- Git / GitHub

### Evaluation
- Python-based evaluation framework
- Benchmark datasets
- Evaluation metrics
- Multi-document evaluation

---

## 📂 Project Structure

```text
DocuRAG/
│
├── backend/
│   └── app/
│       ├── core/
│       │
│       ├── services/
│       │   ├── answerability/
│       │   │   └── answerability_service.py
│       │   │
│       │   └── structured_evidence/
│       │       └── structured_evidence_service.py
│       │
│       └── main.py
│
├── frontend/
│   ├── public/
│   │
│   └── src/
│       ├── components/
│       │   ├── ChatInput.tsx
│       │   ├── ChatWindow.tsx
│       │   ├── SettingsModal.tsx
│       │   ├── Sidebar.tsx
│       │   └── UploadZone.tsx
│       │
│       ├── services/
│       │   ├── api.ts
│       │   └── chatStorage.ts
│       │
│       ├── App.tsx
│       ├── App.css
│       ├── chat.ts
│       ├── index.css
│       └── main.tsx
│
├── data/
│   └── processed/
│
├── evaluation/
│   ├── datasets/
│   │   ├── multi_document.json
│   │   └── nptel_exam.json
│   │
│   ├── fixtures/
│   │   ├── database_exam_2.pdf
│   │   └── database_exam_2.txt
│   │
│   ├── benchmark1_results.json
│   ├── benchmark2_results.json
│   ├── dataset.json
│   ├── latest_results.json
│   ├── metrics.py
│   ├── multi_document_results.json
│   ├── run_evaluation.py
│   └── runner.py
│
├── docker-compose.yml
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

Before running DocuRAG, make sure the following are installed:

- Python 3.12+
- Node.js 18+
- npm
- Docker
- Docker Compose
- Git

---

## 📥 Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DocuRAG.git
cd DocuRAG
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## 🐍 Backend Setup

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration

Create your local environment file:

```bash
cp .env.example .env
```

Then configure the required environment variables inside `.env`.

Example:

```env
DATABASE_URL=your_database_url
LLM_API_KEY=your_llm_api_key
VECTOR_DB_URL=your_vector_database_url
```

**Never commit `.env` to GitHub.**

The repository contains `.env.example` so that other developers know which configuration variables are required.

---

## 🐘 Start PostgreSQL

DocuRAG uses Docker for the database environment.

Start the required services:

```bash
docker compose up -d
```

Check running containers:

```bash
docker ps
```

To stop the services:

```bash
docker compose down
```

---

## 🚀 Start the Backend

From the project root directory:

```bash
cd ~/DocuRAG
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Start FastAPI:

```bash
uvicorn backend.app.main:app --reload
```

The backend should be available at:

```
http://127.0.0.1:8000
```

---

## 📚 API Documentation

FastAPI automatically provides interactive API documentation.

Open:

```
http://127.0.0.1:8000/docs
```

Alternative ReDoc documentation:

```
http://127.0.0.1:8000/redoc
```

The Swagger interface can be used to test the backend APIs directly.

---

## ⚛️ Start the Frontend

Open another terminal.

Navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```
http://localhost:5173
```

---

## 🔄 Running the Complete Application

You need two development processes.

**Terminal 1 — Backend**
```bash
cd ~/DocuRAG
source .venv/bin/activate
uvicorn backend.app.main:app --reload
```

**Terminal 2 — Frontend**
```bash
cd ~/DocuRAG/frontend
npm run dev
```

**Database**

Make sure PostgreSQL is running:

```bash
docker compose up -d
```

Then open:

```
http://localhost:5173
```

---

## 💬 Using DocuRAG

### Step 1 — Upload a Document

Upload a supported PDF document through the frontend.

The document goes through the document processing pipeline.

```text
PDF
 ↓
Text Extraction
 ↓
Document Processing
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Storage
```

### Step 2 — Ask a Question

After processing the document, ask a question through the chat interface.

Example:

> What is the main objective of this document?

or:

> What are the important concepts discussed in Chapter 3?

### Step 3 — Retrieval

DocuRAG retrieves relevant document evidence for the question.

Instead of passing the entire document to the LLM, relevant pieces of information are selected.

```text
User Question
      ↓
Query Representation
      ↓
Vector Retrieval
      ↓
Relevant Chunks
```

### Step 4 — Answerability

The system checks whether the retrieved evidence contains enough information to answer the question.

Conceptually:

```text
Question
   +
Retrieved Evidence
   │
   ▼
Answerability Service
   │
   ├── Sufficient Evidence
   │         ↓
   │      Generate Answer
   │
   └── Insufficient Evidence
             ↓
       Avoid Unsupported Answer
```

This helps reduce unsupported or hallucinated responses.

### Step 5 — Structured Evidence

The retrieved information is processed into structured evidence before answer generation.

This creates a controlled boundary between:

```text
Retrieved Document Evidence
            ↓
Structured Evidence
            ↓
LLM
            ↓
Final Answer
```

---

## 🔎 RAG Pipeline

The core RAG process can be represented as:

```text
                 ┌──────────────┐
                 │   Documents  │
                 └──────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Text Extraction│
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │    Chunking   │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │   Embeddings  │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Vector Store  │
                └───────┬───────┘
                        │
                        │
User Question ──────────┤
                        ▼
                ┌───────────────┐
                │   Retrieval   │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Answerability │
                └───────┬───────┘
                        │
                        ▼
                ┌──────────────────┐
                │ Structured       │
                │ Evidence         │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │       LLM        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Grounded Answer  │
                └──────────────────┘
```

---

## 🧪 Evaluation

DocuRAG contains an evaluation framework for testing the quality of document question answering.

The evaluation directory contains:

```text
evaluation/
├── datasets/
├── fixtures/
├── metrics.py
├── runner.py
├── run_evaluation.py
└── benchmark results
```

Evaluation datasets include document-based question-answering scenarios and multi-document testing.

Run the evaluation according to the configured evaluation scripts:

```bash
python -m evaluation.run_evaluation
```

Make sure the required backend services and dependencies are available before running evaluations.

---

## 🧪 Testing

Run the project's test suite with:

```bash
pytest
```

For more verbose output:

```bash
pytest -v
```

---

## 🖥️ Frontend Architecture

The frontend is organized into reusable React components.

```text
App
│
├── Sidebar
│   ├── Recent Chats
│   ├── Documents
│   └── User Settings
│
├── ChatWindow
│   ├── User Messages
│   ├── AI Messages
│   ├── Sources
│   └── Loading State
│
├── ChatInput
│   ├── Text Input
│   ├── Document Upload
│   └── Send Button
│
├── UploadZone
│   ├── Drag & Drop
│   └── Document Processing
│
└── SettingsModal
    ├── Theme
    └── Configuration
```

---

## 🎨 UI Features

The frontend provides a modern document-chat experience with:

- Dark interface
- Glass-style panels
- Responsive layout
- Recent conversations
- Document history
- PDF upload
- Drag-and-drop support
- Chat interface
- Markdown rendering
- Code block rendering
- Source references
- Loading indicators
- Settings interface
- Responsive mobile layout

---

## 🗃️ Chat History

DocuRAG maintains recent conversations through the frontend chat storage layer.

The interface allows users to:

- Create a new conversation
- Continue previous conversations
- View recent chats
- Delete conversations
- Manage uploaded documents
- Restore previous conversation context

---

## 📄 Document Handling

Documents are treated as the primary knowledge source for the RAG system.

The system is designed to support general PDF/document analysis rather than being limited to a specific document type.

Example document categories include:

- Academic documents
- Examination documents
- Technical documentation
- Reports
- Research papers
- Business documents
- Manuals
- Multi-document collections

---

## 🔐 Security Considerations

Do not commit sensitive credentials.

The following files should remain local:

```
.env
```

API keys, database passwords, authentication secrets, and other credentials should be provided through environment variables.

The repository provides:

```
.env.example
```

as a configuration template.

---

## 🐳 Docker

Start services:

```bash
docker compose up -d
```

View running containers:

```bash
docker ps
```

View logs:

```bash
docker compose logs
```

Stop services:

```bash
docker compose down
```

Stop and remove containers:

```bash
docker compose down -v
```

---

## 📊 Project Goals

The primary goals of DocuRAG are:

- Provide document-grounded question answering.
- Retrieve relevant information from documents efficiently.
- Reduce unsupported LLM responses.
- Introduce an explicit answerability layer.
- Structure retrieved evidence before generation.
- Provide document references with generated answers.
- Support multiple document-analysis workflows.
- Provide a modern conversational user experience.
- Evaluate RAG performance using benchmark datasets.

---

## 🧩 Key Components

### Answerability Service

The answerability layer determines whether the available document evidence is sufficient to attempt an answer.

It helps distinguish between:

- Question can be answered
- Document does not provide enough information

### Structured Evidence Service

The structured evidence layer organizes retrieved information before it reaches the final answer-generation stage.

This provides a more controlled interface between retrieval and generation.

### Retrieval Layer

The retrieval layer identifies document sections relevant to the user's query.

The goal is to provide the LLM with relevant evidence instead of the entire document.

### LLM Layer

The LLM uses the retrieved and structured evidence to generate the final natural-language response.

The architecture is designed around document-grounded generation.

---

## 👥 Team

DocuRAG is developed as a team project.

**Team Member 1**
Responsible for:
- Document processing
- Text extraction
- Chunking
- Embeddings
- Retrieval pipeline

**Team Member 2**
Responsible for:
- Answerability
- Structured evidence
- RAG response pipeline
- Evaluation

**Team Member 3**
Responsible for:
- Frontend
- Application integration
- Cloud / DevOps
- Deployment
- Infrastructure

---

## ☁️ Deployment

The application can be extended toward cloud deployment using services such as:

```text
Frontend
   │
   ▼
Cloud Hosting
   │
   ▼
FastAPI Backend
   │
   ├── PostgreSQL
   │
   ├── Vector Database
   │
   └── LLM Service
```

The infrastructure can be containerized using Docker and deployed using cloud infrastructure.

---

## 🚧 Future Improvements

Potential future improvements include:

- Multi-user authentication
- Advanced document management
- Improved retrieval strategies
- Hybrid search
- Reranking
- Better citation verification
- Streaming LLM responses
- Conversation persistence on the backend
- Cloud deployment
- Monitoring and observability
- Distributed processing
- Improved evaluation metrics
- Additional document formats
- OCR support for scanned documents

---

## 📌 Development Status

DocuRAG is an actively developed RAG-based document analysis project.

Current development focuses on:

```text
Document Processing
        +
RAG Retrieval
        +
Answerability
        +
Structured Evidence
        +
LLM Generation
        +
Evaluation
        +
Modern Frontend
        +
Cloud / DevOps
```

---

## 🤝 Contributing

Contributions are welcome.

To contribute:

```bash
git clone https://github.com/YOUR_USERNAME/DocuRAG.git
cd DocuRAG
```

Create a new branch:

```bash
git checkout -b feature/your-feature
```

Make your changes and commit:

```bash
git add .
git commit -m "Add your feature"
```

Push the branch:

```bash
git push origin feature/your-feature
```

Then create a Pull Request on GitHub.

---

## 📜 License

Add the project's selected license here.

For example:

```
MIT License
```

---

## ⭐ Acknowledgements

DocuRAG is built using open-source technologies and frameworks including:

- Python
- FastAPI
- React
- TypeScript
- Vite
- PostgreSQL
- Docker
- RAG techniques
- Large Language Models
- Vector search technologies

---

## 📬 Contact

**Vishwa Sabaris V**

GitHub: [https://github.com/VishwaSabaris](https://github.com/VishwaSabaris)

LinkedIn: [https://linkedin.com/in/vishwa-sabaris-v](https://linkedin.com/in/vishwa-sabaris-v)

---

## ⭐ If you find this project useful

Give the repository a ⭐ on GitHub and feel free to explore, fork, and contribute.
