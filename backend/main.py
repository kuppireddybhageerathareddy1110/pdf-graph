"""
PDF → Graph → GAT Pipeline
Production FastAPI Backend
"""

import os
import uuid
import json
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from modules.pdf_loader import extract_pdf_pages
from modules.nlp_processor import extract_entities_relations
from modules.graph_builder import build_graph, merge_page_graphs
from modules.adjacency_builder import create_adjacency_matrix
from modules.feature_builder import create_feature_matrix
from modules.gat_model import run_gat_inference
from modules.visualizer import visualize_graph

import networkx as nx


def graph_to_dict(G: nx.DiGraph) -> dict:
    """Convert a NetworkX graph to a JSON-serialisable dict."""
    return {
        "nodes": [{"id": n, **{k: str(v) for k, v in d.items()}}
                  for n, d in G.nodes(data=True)],
        "edges": [{"source": u, "target": v,
                   "weight": float(data.get("weight", 1.0)),
                   "relations": data.get("relations", [])}
                  for u, v, data in G.edges(data=True)],
    }

# ─── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PDF Graph GAT API",
    description="""
    ## PDF → Page-wise NLP → Knowledge Graph → GAT → Visualization

    A production-level research pipeline that:
    1. Extracts pages from PDFs
    2. Runs NLP entity/relation extraction on each page
    3. Builds a Knowledge Graph (NetworkX)
    4. Creates Adjacency + Feature matrices
    5. Processes through a Graph Attention Network (GAT)
    6. Returns interactive visualizations

    **Open Source** | MIT License | [GitHub](https://github.com/kuppireddybhageerathareddy1110/pdf-graph)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for generated visualizations
os.makedirs("data/graphs", exist_ok=True)
app.mount("/graphs", StaticFiles(directory="data/graphs"), name="graphs")

# ─── In-memory job store (replace with Redis/DB in production) ─────────────────
jobs: dict = {}


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "PDF Graph GAT API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/upload", tags=["Pipeline"])
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_pages: Optional[int] = None,
):
    """
    Upload a PDF file to start the graph extraction pipeline.

    Returns a `job_id` — poll `/status/{job_id}` for progress.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    job_id = str(uuid.uuid4())
    pdf_path = f"data/pdf/{job_id}.pdf"

    os.makedirs("data/pdf", exist_ok=True)
    contents = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(contents)

    jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "progress": 0,
        "pages_total": 0,
        "pages_done": 0,
        "error": None,
    }

    background_tasks.add_task(run_pipeline, job_id, pdf_path, max_pages)

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}", tags=["Pipeline"])
def get_status(job_id: str):
    """Poll pipeline progress for a given job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs[job_id]


@app.get("/results/{job_id}", tags=["Results"])
def get_results(job_id: str):
    """
    Retrieve full pipeline results once job is complete.

    Returns per-page graph data, matrices, GAT embeddings, and visualization URLs.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(status_code=400, detail=f"Job not complete. Status: {job['status']}")
    return job.get("results", {})


@app.get("/graph/{job_id}/{page_num}", tags=["Visualization"])
def get_graph_html(job_id: str, page_num: int):
    """Return the interactive PyVis HTML visualization for a specific page."""
    path = f"data/graphs/{job_id}_page_{page_num}.html"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Graph visualization not found.")
    return FileResponse(path, media_type="text/html")


@app.get("/adjacency/{job_id}/{page_num}", tags=["Matrices"])
def get_adjacency(job_id: str, page_num: int):
    """Return the adjacency matrix for a specific page as JSON."""
    path = f"data/adjacency/{job_id}_page_{page_num}.npy"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Adjacency matrix not found.")
    matrix = np.load(path).tolist()
    return {"page": page_num, "adjacency_matrix": matrix}


@app.get("/features/{job_id}/{page_num}", tags=["Matrices"])
def get_features(job_id: str, page_num: int):
    """Return the feature (embedding) matrix for a specific page."""
    path = f"data/features/{job_id}_page_{page_num}.npy"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Feature matrix not found.")
    matrix = np.load(path).tolist()
    return {"page": page_num, "feature_matrix_shape": [len(matrix), len(matrix[0]) if matrix else 0]}


# ─── Background Pipeline ───────────────────────────────────────────────────────

def run_pipeline(job_id: str, pdf_path: str, max_pages: Optional[int]):
    """Full pipeline: PDF → NLP → Graph → Matrices → GAT → Visualization."""
    try:
        jobs[job_id]["status"] = "processing"

        # Phase 1: Extract pages
        pages = extract_pdf_pages(pdf_path)
        if max_pages:
            pages = pages[:max_pages]

        total = len(pages)
        jobs[job_id]["pages_total"] = total

        results = {"pages": []}

        for i, page in enumerate(pages):
            page_num = page["page_number"]
            text = page["text"].strip()

            if not text:
                jobs[job_id]["pages_done"] += 1
                continue

            # Phase 2: NLP
            entities, relations = extract_entities_relations(text)

            if not entities:
                jobs[job_id]["pages_done"] += 1
                continue

            # Phase 3: Build graph
            G = build_graph(entities, relations)
            graph_dict = graph_to_dict(G)

            # Phase 4: Adjacency matrix
            os.makedirs("data/adjacency", exist_ok=True)
            adj_matrix, node_map = create_adjacency_matrix(G)
            np.save(f"data/adjacency/{job_id}_page_{page_num}.npy", adj_matrix)

            # Phase 5: Feature matrix
            os.makedirs("data/features", exist_ok=True)
            nodes = list(G.nodes)
            features = create_feature_matrix(nodes)
            np.save(f"data/features/{job_id}_page_{page_num}.npy", features)

            # Phase 6: GAT
            gat_embeddings = run_gat_inference(features, adj_matrix)

            # Phase 7: Visualize
            graph_html_path = f"data/graphs/{job_id}_page_{page_num}.html"
            visualize_graph(G, output_path=graph_html_path)

            results["pages"].append({
                "page_number": page_num,
                "num_entities": len(entities),
                "num_relations": len(relations),
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "top_entities": entities[:10],
                "graph_data": graph_dict,
                "adj_matrix_shape": list(adj_matrix.shape),
                "feature_matrix_shape": list(features.shape),
                "gat_output_shape": list(gat_embeddings.shape),
                "graph_url": f"/graphs/{job_id}_page_{page_num}.html",
            })

            jobs[job_id]["pages_done"] += 1
            jobs[job_id]["progress"] = int(((i + 1) / total) * 100)

        jobs[job_id]["status"] = "complete"
        jobs[job_id]["results"] = results

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["traceback"] = traceback.format_exc()


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
