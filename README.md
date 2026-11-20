# PDF Reader Chat

A small PDF question-answering app with a browser UI and a Python/FastAPI backend. Register or log in, upload a PDF, process its text into page-aware chunks, embed and store those chunks in Neon/PostgreSQL (or local SQLite when no Neon URL is configured), then ask questions and see the cited pages. PDF libraries are scoped to the signed-in account.

## Run locally

1. Install Python 3.10 or later.
2. In a terminal, enter the `server` folder and create a virtual environment:

   ```powershell
   cd server
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. For AI generated answers, copy `.env.example` to `.env` inside `server` (`Copy-Item .env.example .env` in PowerShell), then set `GEMINI_API_KEY` to your Google Gemini API key. The backend loads `server/.env` automatically. With a key, Gemini generates grounded answers and Gemini embeddings improve retrieval. Without a key, the app uses local keyword/vector matching and returns matching PDF text; that fallback cannot reason or synthesize information like a language model.
4. To use Neon, open `server/.env` (not `.env.example`) and add your copied Neon connection string on its own line as `DATABASE_URL=postgresql://...`. Keep the password private. If `DATABASE_URL` is empty or missing, the app continues to use `server/pdf_chat.db` (SQLite). Neon starts with an empty set of app tables; existing SQLite accounts, PDFs and chat data are not copied automatically.
5. Install dependencies after setting up Neon support, then start the API:

   ```powershell
   pip install -r requirements.txt
   py -m uvicorn main:app --port 8011
   ```

6. Open **http://localhost:8011** in your browser. FastAPI serves the UI and API together, avoiding browser cross-origin errors. You can also open `index.html` directly; it will try `http://localhost:8011`.

API health check: `http://localhost:8011/api/health`. Interactive API docs: `http://localhost:8011/docs`.

Each PDF in the library has an actions menu to archive or permanently delete it. Archived PDFs can be restored from the Archived section. Registration asks for a username, email, contact number, and password.

## Processing workflow

1. Validate the uploaded file and save it under `server/uploads/`.
2. Extract selectable text one page at a time with `pypdf`.
3. Normalize whitespace, remove null/soft-hyphen artifacts, and join line-break hyphenation.
4. Split text into overlapping chunks while preserving page ranges.
5. Generate embeddings with Gemini when configured, otherwise use deterministic local feature hashing.
6. Store document metadata, chunk text, page ranges, and vectors in Neon/PostgreSQL when `DATABASE_URL` is configured; otherwise use local SQLite.
7. Rank chunks with BM25 keyword relevance plus available vector similarity. When configured, Gemini resolves follow-up questions and answers from the retrieved excerpts with page citations; otherwise the app uses basic pronoun resolution and shows the most relevant matching sentences.
8. Handle document-wide quantity and roster questions by scanning all extracted pages, so totals such as question counts and team-member names do not depend on the top retrieved chunks.

Scanned/image-only PDFs are reported as unsupported because OCR is not included in this MVP. SQLite and local uploads are intended for development; there is no user authentication or multi-user isolation yet.
