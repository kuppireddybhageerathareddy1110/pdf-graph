# PDF → Graph → GAT

> **Open-source research pipeline:** Upload any PDF → extract page-wise knowledge graphs → run Graph Attention Networks → visualize interactively.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org)

> https://pdf-graph.netlify.app/
> 
> https://pdf-graph.onrender.com/ 
## Pipeline

```
PDF → Page Extraction (PyMuPDF)
    → NLP: NER + SVO Relations (spaCy)
    → Knowledge Graph (NetworkX DiGraph)
    → Adjacency Matrix [N×N]
    → Feature Matrix [N×384] (SentenceTransformers)
    → Graph Attention Network (PyTorch Geometric)
    → Interactive Visualization (PyVis + Next.js)
```

Each page = independent graph. All results inspectable per-page.

## Quick Start

```bash
git clone https://github.com/your-repo/pdf-graph-gat.git
cd pdf-graph-gat/backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --reload        # → http://localhost:8000/docs
```

```bash
cd frontend && npm install && npm run dev   # → http://localhost:3000
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload PDF → returns `job_id` |
| GET | `/status/{job_id}` | Poll progress (0-100%) |
| GET | `/results/{job_id}` | Full per-page results |
| GET | `/graph/{job_id}/{page}` | Interactive PyVis HTML |
| GET | `/adjacency/{job_id}/{page}` | Adjacency matrix JSON |

## Modules

- `pdf_loader.py` — PyMuPDF page extraction + metadata
- `nlp_processor.py` — spaCy NER + SVO dependency triples
- `graph_builder.py` — NetworkX DiGraph + stats
- `adjacency_builder.py` — Binary/weighted/normalized adj matrix + edge_index
- `feature_builder.py` — SentenceTransformer embeddings + TF-IDF fallback
- `gat_model.py` — 2-layer GAT (multi-head attention) + attention weight extraction
- `visualizer.py` — PyVis interactive HTML + matplotlib static export

## Tests

```bash
cd backend && pytest tests/ -v
```

## Deploy

- **Backend** → Render (Python, `uvicorn main:app --host 0.0.0.0 --port 10000`)
- **Frontend** → Netlify (Next.js, base dir: `frontend`)
- **CI/CD** → GitHub Actions (`.github/workflows/ci.yml`)

## Extend This

| Research Direction | Where |
|---|---|
| Co-reference resolution | `nlp_processor.py` |
| Multi-doc graph merging | New `modules/merger.py` |
| Link prediction | `gat_model.py` — add edge decoder |
| KG embeddings (TransE) | New `modules/kg_embeddings.py` |
| OCR for scanned PDFs | `pdf_loader.py` — Tesseract fallback |
| Neo4j / RDF export | New `modules/exporter.py` |

## License

MIT — see [LICENSE](LICENSE)
