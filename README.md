# KZ Admission Helper Bot

A GPT-4o powered Telegram assistant that guides prospective students through the
full journey of enrolling at a university in Kazakhstan (Almaty) — choosing a
university, assembling documents, handling relocation, military-registration
questions, and staying on top of admission deadlines.

The bot combines a **per-user long-term memory (RAG)** layer, **multimodal
document ingestion** (photos, PDFs, DOCX, voice), and a conversational
**onboarding flow**, all backed by OpenAI's GPT-4o, Vision, and Whisper.

---

## Features

### Per-user RAG memory
- Every user has an isolated profile and memory store in SQLite.
- Conversations, uploaded documents, and profile facts are persisted as
  categorized "memories".
- Retrieval uses **SQLite FTS5** full-text search (`unicode61` tokenizer, with
  Russian support) plus a recency feed, so each GPT-4o call is grounded in the
  user's relevant history and current context.
- Context assembly merges profile + relevant memories + recent messages +
  live document-checklist status into the system prompt.

### Multimodal document & voice ingestion
- **Photos** → OCR via Tesseract (`rus+eng`) with automatic fallback to
  **OpenAI GPT-4o Vision** for scanned/low-quality images.
- **PDF** → text extraction via PyPDF2.
- **DOCX** → text extraction via `python-docx`.
- **Voice messages** → transcription via **OpenAI Whisper** (`whisper-1`), then
  answered like any text message.
- Uploaded documents are auto-classified (passport, diploma, apostille, medical
  certificate, etc.) from caption/filename and folded into a live document
  checklist.

### Onboarding FSM
- A `ConversationHandler` state machine walks new users through a short
  interview — name → age → interests → budget → study language — and stores each
  answer in the profile and memory.
- Finishes with a personalized GPT-4o recommendation based on the collected
  profile. Returning users are recognized and skip onboarding.

### Domain command set
- `/docs` — live document checklist with upload progress (✅/❌).
- `/plan` — deadline-aware step-by-step action plan.
- `/universities` — profile-based university comparison and recommendation.
- `/army` — military registration / deferral guidance.
- `/move` — relocation plan to Almaty (uses the user's stored origin city).
- `/status` — overall readiness snapshot with a computed countdown to the
  admission window.

### OpenAI integration
- **GPT-4o** for chat and vision, **Whisper** for speech-to-text.
- Async client (`AsyncOpenAI`) throughout for non-blocking Telegram handling.
- Long responses are automatically chunked to respect Telegram's 4096-char
  message limit.

---

## Architecture

| File | Responsibility |
|------|----------------|
| `bot.py` | Telegram wiring: command/message handlers, onboarding FSM, context builder, GPT-4o & Whisper calls |
| `rag.py` | SQLite schema, FTS5 memory store/search, user profiles, documents, checklist |
| `knowledge.py` | System prompt / knowledge base + document-type detection |
| `documents.py` | Text extraction from PDF, DOCX, and images (Tesseract + Vision fallback) |

Data lives in `store/memory.db` (created on first run). Tables: `memories`
(+ `memories_fts` virtual table and sync triggers), `documents`, and
`user_profiles`.

---

## Setup

### Requirements
- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on the
  host, with Russian + English language data (`tesseract-ocr-rus`,
  `tesseract-ocr-eng`). Without it, image OCR falls back to GPT-4o Vision.

### Install
```bash
git clone <your-repo-url>
cd kz-admission-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configure
```bash
cp .env.example .env
# then edit .env and fill in your tokens
```

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Bot token from [@BotFather](https://t.me/BotFather) |
| `OPENAI_API_KEY` | OpenAI API key (GPT-4o, Vision, Whisper) |

### Run
```bash
python bot.py
```

Run the retrieval-isolation check with `python -m unittest discover -s tests`.

The bot uses long polling. On start it registers its command menu and
initializes the local database automatically.

---

## Tech stack

- **python-telegram-bot 21.x** (async)
- **OpenAI Python SDK** — GPT-4o, GPT-4o Vision, Whisper
- **SQLite + FTS5** — per-user memory & full-text retrieval
- **PyPDF2**, **python-docx**, **Pillow**, **pytesseract** — document ingestion

---

## Notes

This is a portfolio/showcase build. Secrets, the runtime database, and cached
bytecode are excluded via `.gitignore`. Provide your own credentials in `.env`.
The knowledge base reflects general Kazakhstan-admission information and is not
legal advice — always verify dates and requirements with official sources.
