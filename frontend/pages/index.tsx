"use client";

import { useState, useRef } from "react";
import axios from "axios";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type JobStatus = {
  status: string;
  filename: string;
  progress: number;
  pages_total: number;
  pages_done: number;
  error: string | null;
  results?: PipelineResults;
};

type PageResult = {
  page_number: number;
  num_entities: number;
  num_relations: number;
  num_nodes: number;
  num_edges: number;
  top_entities: string[];
  adj_matrix_shape: number[];
  feature_matrix_shape: number[];
  gat_output_shape: number[];
  graph_url: string;
};

type PipelineResults = {
  pages: PageResult[];
};

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [selectedPage, setSelectedPage] = useState<PageResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const pollRef = useRef<NodeJS.Timeout | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (f: File | null) => {
    if (f && f.type === "application/pdf") setFile(f);
  };

  const startPipeline = async () => {
    if (!file) return;
    setLoading(true);
    setJobStatus(null);
    setSelectedPage(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API}/upload`, formData);
      const id = res.data.job_id;
      setJobId(id);
      pollStatus(id);
    } catch (e) {
      setLoading(false);
      alert("Upload failed. Is the backend running?");
    }
  };

  const pollStatus = (id: string) => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`${API}/status/${id}`);
        const status: JobStatus = res.data;
        setJobStatus(status);

        if (status.status === "complete") {
          clearInterval(pollRef.current!);
          setLoading(false);
          const results = await axios.get(`${API}/results/${id}`);
          setJobStatus((prev) => prev ? { ...prev, results: results.data } : prev);
        } else if (status.status === "failed") {
          clearInterval(pollRef.current!);
          setLoading(false);
        }
      } catch {}
    }, 1500);
  };

  const stageColors: Record<string, string> = {
    queued: "#888",
    processing: "#ffaa00",
    complete: "#00ff88",
    failed: "#ff4466",
  };

  return (
    <main style={{ background: "#0a0a0f", minHeight: "100vh", color: "#e0e0f0", fontFamily: "'Space Mono', monospace" }}>
      {/* Header */}
      <header style={{ borderBottom: "1px solid #1e1e2e", padding: "20px 32px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <h1 style={{ fontFamily: "Syne, sans-serif", fontSize: 22, fontWeight: 800, margin: 0 }}>
            PDF <span style={{ color: "#00ff88" }}>→</span> GRAPH <span style={{ color: "#00ff88" }}>→</span> GAT
          </h1>
          <p style={{ fontSize: 11, color: "#555570", margin: "4px 0 0", letterSpacing: "0.1em" }}>
            PAGE-WISE NLP · KNOWLEDGE GRAPH · GRAPH ATTENTION NETWORK
          </p>
        </div>
        <div style={{ fontSize: 11, color: "#555570" }}>
          <a href={`${API}/docs`} target="_blank" rel="noopener noreferrer" style={{ color: "#4488ff", textDecoration: "none" }}>API DOCS ↗</a>
          {" · "}
          <a href="https://github.com/your-repo/pdf-graph-gat" target="_blank" rel="noopener noreferrer" style={{ color: "#4488ff", textDecoration: "none" }}>GITHUB ↗</a>
        </div>
      </header>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "32px" }}>

        {/* Pipeline diagram */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 32, overflowX: "auto", paddingBottom: 8 }}>
          {["PDF Upload", "Page Extract", "NLP / NER", "Build Graph", "Adj Matrix", "Feature Matrix", "GAT Model", "Visualize"].map((s, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ background: "#111118", border: "1px solid #1e1e2e", padding: "8px 14px", fontSize: 10, whiteSpace: "nowrap", letterSpacing: "0.05em" }}>
                {s}
              </div>
              {i < 7 && <span style={{ color: "#555570" }}>→</span>}
            </div>
          ))}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 24 }}>

          {/* Left: Upload + Status */}
          <div>
            {/* Drop zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); handleFileChange(e.dataTransfer.files[0]); }}
              style={{
                border: `2px dashed ${dragging ? "#00ff88" : file ? "#4488ff" : "#1e1e2e"}`,
                background: dragging ? "rgba(0,255,136,0.04)" : "#111118",
                padding: 32,
                textAlign: "center",
                cursor: "pointer",
                marginBottom: 16,
                transition: "all 0.2s",
              }}
            >
              <div style={{ fontSize: 32, marginBottom: 12 }}>📄</div>
              {file ? (
                <div>
                  <div style={{ color: "#00ff88", fontSize: 13, marginBottom: 4 }}>{file.name}</div>
                  <div style={{ color: "#555570", fontSize: 11 }}>{(file.size / 1024).toFixed(1)} KB</div>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: 13, marginBottom: 4 }}>Drop PDF or click to upload</div>
                  <div style={{ color: "#555570", fontSize: 11 }}>Supports multi-page PDFs</div>
                </div>
              )}
              <input ref={fileInputRef} type="file" accept=".pdf" style={{ display: "none" }} onChange={(e) => handleFileChange(e.target.files?.[0] || null)} />
            </div>

            <button
              onClick={startPipeline}
              disabled={!file || loading}
              style={{
                width: "100%",
                padding: "14px",
                background: file && !loading ? "#00ff88" : "#1e1e2e",
                color: file && !loading ? "#0a0a0f" : "#555570",
                border: "none",
                cursor: file && !loading ? "pointer" : "not-allowed",
                fontFamily: "Space Mono, monospace",
                fontWeight: 700,
                fontSize: 13,
                letterSpacing: "0.1em",
                marginBottom: 16,
              }}
            >
              {loading ? "⏳ PROCESSING..." : "▶ RUN PIPELINE"}
            </button>

            {/* Status card */}
            {jobStatus && (
              <div style={{ background: "#111118", border: "1px solid #1e1e2e", padding: 20 }}>
                <div style={{ fontSize: 10, color: "#555570", letterSpacing: "0.1em", marginBottom: 12 }}>PIPELINE STATUS</div>

                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                  <span style={{ fontSize: 11 }}>{jobStatus.filename}</span>
                  <span style={{ fontSize: 11, color: stageColors[jobStatus.status] || "#888", fontWeight: 700 }}>
                    {jobStatus.status.toUpperCase()}
                  </span>
                </div>

                {/* Progress bar */}
                <div style={{ background: "#1e1e2e", height: 4, marginBottom: 8 }}>
                  <div style={{ height: "100%", background: "#00ff88", width: `${jobStatus.progress}%`, transition: "width 0.5s" }} />
                </div>

                <div style={{ fontSize: 10, color: "#555570", display: "flex", justifyContent: "space-between" }}>
                  <span>{jobStatus.pages_done} / {jobStatus.pages_total} pages</span>
                  <span>{jobStatus.progress}%</span>
                </div>

                {jobStatus.error && (
                  <div style={{ marginTop: 12, fontSize: 11, color: "#ff4466", background: "rgba(255,68,102,0.1)", padding: 8 }}>
                    {jobStatus.error}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right: Results */}
          <div>
            {jobStatus?.results ? (
              <div>
                {/* Page selector */}
                <div style={{ fontSize: 10, color: "#555570", letterSpacing: "0.1em", marginBottom: 12 }}>
                  SELECT PAGE — {jobStatus.results.pages.length} PROCESSED
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 20 }}>
                  {jobStatus.results.pages.map((page) => (
                    <button
                      key={page.page_number}
                      onClick={() => setSelectedPage(page)}
                      style={{
                        padding: "6px 14px",
                        background: selectedPage?.page_number === page.page_number ? "#4488ff" : "#111118",
                        border: `1px solid ${selectedPage?.page_number === page.page_number ? "#4488ff" : "#1e1e2e"}`,
                        color: "#e0e0f0",
                        cursor: "pointer",
                        fontFamily: "Space Mono, monospace",
                        fontSize: 11,
                      }}
                    >
                      PG {page.page_number + 1}
                    </button>
                  ))}
                </div>

                {/* Page detail */}
                {selectedPage && (
                  <div>
                    {/* Metrics */}
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 20 }}>
                      {[
                        { label: "Entities", value: selectedPage.num_entities, color: "#00ff88" },
                        { label: "Relations", value: selectedPage.num_relations, color: "#4488ff" },
                        { label: "Nodes", value: selectedPage.num_nodes, color: "#ffaa00" },
                        { label: "Edges", value: selectedPage.num_edges, color: "#cc44ff" },
                      ].map((m) => (
                        <div key={m.label} style={{ background: "#111118", border: "1px solid #1e1e2e", padding: 16 }}>
                          <div style={{ fontSize: 9, color: "#555570", letterSpacing: "0.1em", marginBottom: 6 }}>{m.label.toUpperCase()}</div>
                          <div style={{ fontSize: 28, fontFamily: "Syne, sans-serif", fontWeight: 800, color: m.color }}>{m.value}</div>
                        </div>
                      ))}
                    </div>

                    {/* Matrix shapes */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 20 }}>
                      {[
                        { label: "Adj Matrix", shape: selectedPage.adj_matrix_shape },
                        { label: "Feature Matrix", shape: selectedPage.feature_matrix_shape },
                        { label: "GAT Output", shape: selectedPage.gat_output_shape },
                      ].map((m) => (
                        <div key={m.label} style={{ background: "#111118", border: "1px solid #1e1e2e", padding: 14 }}>
                          <div style={{ fontSize: 9, color: "#555570", letterSpacing: "0.1em", marginBottom: 6 }}>{m.label}</div>
                          <div style={{ fontSize: 13, color: "#00ff88", fontFamily: "Space Mono" }}>
                            [{m.shape?.join(" × ")}]
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Top entities */}
                    <div style={{ background: "#111118", border: "1px solid #1e1e2e", padding: 16, marginBottom: 20 }}>
                      <div style={{ fontSize: 10, color: "#555570", letterSpacing: "0.1em", marginBottom: 10 }}>TOP ENTITIES</div>
                      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                        {selectedPage.top_entities.map((e, i) => (
                          <span key={i} style={{ fontSize: 11, padding: "3px 10px", background: "rgba(68,136,255,0.15)", border: "1px solid rgba(68,136,255,0.3)", color: "#4488ff" }}>
                            {e}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Graph viewer */}
                    <div style={{ background: "#111118", border: "1px solid #1e1e2e" }}>
                      <div style={{ padding: "12px 16px", borderBottom: "1px solid #1e1e2e", fontSize: 10, color: "#555570", letterSpacing: "0.1em" }}>
                        INTERACTIVE GRAPH — PAGE {selectedPage.page_number + 1}
                      </div>
                      <iframe
                        src={`${API}${selectedPage.graph_url}`}
                        style={{ width: "100%", height: 500, border: "none", background: "#0a0a0f" }}
                        title={`Graph page ${selectedPage.page_number}`}
                      />
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ background: "#111118", border: "1px solid #1e1e2e", padding: 48, textAlign: "center", color: "#555570" }}>
                <div style={{ fontSize: 32, marginBottom: 16 }}>⬡</div>
                <div style={{ fontSize: 13, marginBottom: 8 }}>Upload a PDF to start</div>
                <div style={{ fontSize: 11 }}>The pipeline will extract entities, build graphs,<br />compute matrices, and run GAT on each page.</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
