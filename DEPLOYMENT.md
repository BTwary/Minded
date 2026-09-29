# DEPLOYMENT & ARCHITECTURE GUIDE

This document outlines the architecture, environment variables, local development setup, containerized configuration, database migration, and security model for **Minded (AA-OS)**.

---

## 🏛️ System Architecture

```
                               ┌────────────────────────────────┐
                               │   Next.js 14 Frontend / UI     │
                               │   (DataLens + Mining Cockpit)   │
                               └───────────────┬────────────────┘
                                               │ HTTP / WebSockets / SSE
                                               ▼
                               ┌────────────────────────────────┐
                               │       FastAPI Backend API      │
                               │   (Authentication & Routing)   │
                               └───────┬───────────────┬────────┘
                                       │               │
                     ┌─────────────────▼──┐     ┌──────▼─────────────────┐
                     │ Investigation       │     │ In-Process DuckDB      │
                     │ Controller & State  │     │ Deterministic OLAP     │
                     └─────────┬──────────┘     └────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
  ┌───────────────────┐ ┌─────────────┐ ┌───────────────────┐
  │ PostgreSQL DB     │ │ SciPy/SymPy │ │ Bounded LLM Plan  │
  │ (Supabase/Neon/   │ │ Statistical │ │ (Gemini/Claude/   │
  │  Self-Hosted)     │ │ Verifier    │ │  Ollama)          │
  └───────────────────┘ └─────────────┘ └───────────────────┘
```

---

## 🔐 Environment Variables

| Variable | Scope | Description | Default / Example |
| :--- | :--- | :--- | :--- |
| `AI_ENABLED` | Backend | Enable optional AI augmentation (Mode 2) or run 100% Free (Mode 1) | `false` (Mode 1) or `true` (Mode 2) |
| `AI_PROVIDER` | Backend | LLM planning engine (`none`, `gemini`, `openai`, `claude`, `ollama`, `groq`) | `none` |
| `GEMINI_API_KEY` | Backend | Google Gemini API key (server-side only) | `AIzaSy...` |
| `ANTHROPIC_API_KEY`| Backend | Anthropic Claude API key (server-side only) | `sk-ant-...` |
| `OPENAI_API_KEY` | Backend | OpenAI API key (server-side only) | `sk-...` |
| `DATABASE_URL` | Backend | PostgreSQL / SQLite connection string | `sqlite:///./autonomous_analyst.db` or `postgresql://...` |
| `SECRET_KEY` | Backend | JWT signing key | `[RANDOM_SECURE_KEY]` |
| `CORS_ORIGINS` | Backend | Permitted frontend origins (comma-separated or JSON) | `http://localhost:3000,*` |
| `DATA_STORAGE_DIR` | Backend | Directory for Parquet/CSV dataset storage | `./data_store` |
| `NEXT_PUBLIC_API_URL` | Frontend | Public backend URL accessible from browser | `http://localhost:8000/api/v1` |

> **Security Invariant**: No backend secrets (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DATABASE_URL`, `SECRET_KEY`) are ever bundled or exposed to frontend client code. Only `NEXT_PUBLIC_*` variables are exposed to the browser. Under Mode 1 (`AI_ENABLED=false`), zero API keys are required.

---

## 💻 Local Development Setup

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.11 or 3.12)
- Node.js 18+ (Node 20 / 22 recommended)
- npm or yarn

### 2. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run FastAPI backend with Uvicorn
python -m uvicorn apps.api.src.main:app --host 127.0.0.1 --port 8000 --reload
```
*API is accessible at `http://127.0.0.1:8000` with Swagger docs at `http://127.0.0.1:8000/api/v1/docs`.*

### 3. Frontend Setup
```bash
# Navigate to web directory
cd apps/web

# Install dependencies
npm install

# Run Next.js development server
npm run dev
```
*Frontend is accessible at `http://localhost:3000`.*

---

## 🐳 Docker & Containerized Deployment

AA-OS has two deliberate Docker paths.

### Zero-cost local path (default autonomy)

This is the canonical client installation. It requires no paid AI, hosted database, Redis, MinIO, cloud account, or internet service.

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

The local path uses SQLite and local disk-backed data. `AI_ENABLED=false` and `AI_PROVIDER=none` keep the analytical engine deterministic and provider-independent.

### Optional provider-backed production path

Clients who choose enterprise persistence, object storage, distributed queueing, or AI augmentation may use:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

This path requires client-supplied credentials for PostgreSQL, Redis, MinIO and the production JWT/bootstrap secrets. Those services and any AI APIs are optional integrations; their costs belong to the client.

### Internet-facing production requirement

`docker-compose.prod.yml` exposes ports `8000` (API) and `3000` (Web) to the host for deployment flexibility. This is **not by itself an internet security boundary**. Internet-facing deployments must place the services behind a TLS-terminating reverse proxy/WAF/rate limiter such as Nginx, Traefik, HAProxy or an equivalent client-managed gateway.

### Dependency and migration fail-closed behavior

The production API container validates required secrets before startup and exits on missing/weak/default production credentials. Alembic/custom migration failures also abort container startup; the application is never intentionally started against a failed migration state.

### Provider independence invariant

Removing the production providers must not remove the autonomous analytical capability. PostgreSQL/Redis/MinIO and external AI providers are adapters for scale, persistence, collaboration or augmentation, not prerequisites for the scientific analytical core.

## 🛡️ Security & Zero-Hallucination Guarantees

1. **Deterministic Execution Authority:** LLMs are optional and never have authority over metric arithmetic, aggregation or statistical verification.
2. **Local-first autonomy:** the core can execute with SQLite/local storage and AI disabled.
3. **Immutable Provenance:** investigations retain reproducible analytical evidence and provenance metadata.
4. **Fail-Closed State Consistency:** detected state/evidence integrity violations must stop the relevant operation rather than silently fabricating a result.
