"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/* ───────────────────────── Types ──────────────────────────── */

interface Source {
  pmcid: string;
  title: string;
  journal: string;
  authors: string[] | { name: string }[];
  year: number;
  doi?: string;
  pdf_url?: string;
  article_type?: string;
  dailymed_url?: string;
  source?: string;
  evidence_grade?: "A" | "B" | "C" | "D";
  evidence_level?: 1 | 2 | 3 | 4;
  evidence_term?: string;
  evidence_source?: string;
  citation_index?: number;
}

interface EvidenceLevel {
  grade: "A" | "B" | "C" | "D";
  level: 1 | 2 | 3 | 4;
  label: string;
  terms: string[];
}

interface EvidenceHierarchy {
  levels: EvidenceLevel[];
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  evidenceHierarchy?: EvidenceHierarchy;
  steps?: { title: string; status: "pending" | "loading" | "complete" }[];
  activeTab?: "answer" | "drugs" | "references";
}

/* ───────────────────────── Constants ─────────────────────── */

const CITATION_COLORS = [
  "#f97316", "#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899",
  "#06b6d4", "#ef4444",
];

const EVIDENCE_COLORS: Record<string, string> = {
  A: "#10b981", B: "#3b82f6", C: "#f59e0b", D: "#ef4444",
};

const ARTICLE_TYPE_COLORS: Record<
  string,
  { border: string; background: string; text: string }
> = {
  systematic_review: {
    border: "rgba(20, 184, 166, 0.65)",
    background: "rgba(20, 184, 166, 0.12)",
    text: "#2dd4bf",
  },
  meta_analysis: {
    border: "rgba(59, 130, 246, 0.7)",
    background: "rgba(59, 130, 246, 0.12)",
    text: "#60a5fa",
  },
  clinical_trial: {
    border: "rgba(16, 185, 129, 0.75)",
    background: "rgba(16, 185, 129, 0.12)",
    text: "#34d399",
  },
  guideline: {
    border: "rgba(245, 158, 11, 0.75)",
    background: "rgba(245, 158, 11, 0.12)",
    text: "#fbbf24",
  },
  review_article: {
    border: "rgba(168, 85, 247, 0.75)",
    background: "rgba(168, 85, 247, 0.12)",
    text: "#c084fc",
  },
  drug_label: {
    border: "rgba(249, 115, 22, 0.75)",
    background: "rgba(249, 115, 22, 0.12)",
    text: "#fb923c",
  },
};

const SUGGESTED_QUERIES = [
  "Best treatments for rheumatoid arthritis?",
  "SGLT2 inhibitors cardiovascular benefits",
  "Management of IgG4-related disease",
  "Neurobrucellosis diagnosis and treatment",
];

/* ───────────────────────── Helpers ───────────────────────── */

function formatAuthors(
  authors: (string | { name: string })[] | undefined,
): string {
  if (!authors || authors.length === 0) return "Unknown Authors";
  const strs = authors.map((a) =>
    typeof a === "string" ? a : a?.name || "Unknown",
  );
  if (strs.length === 1) return strs[0];
  if (strs.length === 2) return `${strs[0]}, ${strs[1]}`;
  return `${strs[0]}, ${strs[1]}, et al.`;
}

function getArticleUrl(s: Source): string {
  if (s.dailymed_url) return s.dailymed_url;
  if (s.doi) {
    const clean = s.doi.replace(/^https?:\/\/doi\.org\//, "");
    return `https://doi.org/${clean}`;
  }
  if (s.pmcid && !s.pmcid.toLowerCase().includes("dailymed")) {
    const id = s.pmcid.toUpperCase().startsWith("PMC")
      ? s.pmcid
      : `PMC${s.pmcid}`;
    return `https://www.ncbi.nlm.nih.gov/pmc/articles/${id}/`;
  }
  return "#";
}

function getCitationIndex(s: Source, fallbackIndex: number): number {
  if (typeof s.citation_index === "number" && Number.isFinite(s.citation_index)) {
    return s.citation_index;
  }
  return fallbackIndex + 1;
}

function normalizeArticleType(v?: string): string {
  return (v || "")
    .trim()
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/[-\s]+/g, "_");
}

function toTitleCase(v: string): string {
  return v
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function getArticleTypeBadge(source: Source): {
  label: string;
  style: React.CSSProperties;
} | null {
  const rawType = source.article_type || source.evidence_term;
  if (!rawType) return null;

  const normalized = normalizeArticleType(rawType);
  const palette = ARTICLE_TYPE_COLORS[normalized] || {
    border: "rgba(100, 116, 139, 0.75)",
    background: "rgba(100, 116, 139, 0.12)",
    text: "#94a3b8",
  };

  return {
    label: toTitleCase(rawType),
    style: {
      borderColor: palette.border,
      backgroundColor: palette.background,
      color: palette.text,
    },
  };
}

/* ───────────────────── Markdown Renderer ─────────────────── */

/**
 * Splits markdown content so that `[n]` citation markers become
 * interleaved with the surrounding text. We then render each
 * segment: plain text → ReactMarkdown, citation → badge.
 */
function MarkdownWithCitations({
  content,
  sources,
  onCitationClick,
}: {
  content: string;
  sources: Source[];
  onCitationClick: () => void;
}) {
  const sourcesByCitation = React.useMemo(() => {
    const m = new Map<number, Source>();
    sources.forEach((s, i) => {
      const n = getCitationIndex(s, i);
      if (!m.has(n)) m.set(n, s);
    });
    return m;
  }, [sources]);

  // Split on citation markers like [1], [2], [12] etc.
  const parts = content.split(/(\[\d+\])/g);

  return (
    <div className="md-content">
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+)\]$/);
        if (m) {
          const num = parseInt(m[1]);
          const color = CITATION_COLORS[(num - 1) % CITATION_COLORS.length];
          return (
            <span
              key={i}
              className="citation-badge"
              style={{ backgroundColor: color }}
              onClick={onCitationClick}
              title={sourcesByCitation.get(num)?.title || `Reference ${num}`}
            >
              {num}
            </span>
          );
        }
        if (!part) return null;
        return (
          <ReactMarkdown key={i} remarkPlugins={[remarkGfm]}>
            {part}
          </ReactMarkdown>
        );
      })}
    </div>
  );
}

/* ════════════════════════ MAIN PAGE ════════════════════════ */

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  /* Auto-scroll on new content */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /* ── Set active tab for a specific message ── */
  const setActiveTab = useCallback(
    (idx: number, tab: "answer" | "drugs" | "references") => {
      setMessages((prev) => {
        const next = [...prev];
        next[idx] = { ...next[idx], activeTab: tab };
        return next;
      });
    },
    [],
  );

  /* ── Submit handler (SSE streaming) ── */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = input.trim();
    if (!q || isLoading) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: q },
      {
        role: "assistant",
        content: "",
        steps: [
          { title: "Analyzing Query", status: "loading" },
          { title: "Retrieving Articles", status: "pending" },
          { title: "Reranking Evidence", status: "pending" },
          { title: "Checking Source PDFs", status: "pending" },
          { title: "Synthesizing Answer", status: "pending" },
        ],
      },
    ]);
    setInput("");
    setIsLoading(true);
    setPdfUrl(null);

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, stream: true }),
      });
      if (!res.ok) throw new Error("API request failed");

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let answer = "";
      let sources: Source[] = [];
      let hierarchy: EvidenceHierarchy | undefined;
      let buf = "";

      const updateMsg = (fn: (m: Message) => Message) =>
        setMessages((prev) => {
          const n = [...prev];
          n[n.length - 1] = fn(n[n.length - 1]);
          return n;
        });

      // Mark step i as loading, all before it as complete
      const activateStep = (i: number) =>
        updateMsg((m) => ({
          ...m,
          steps: m.steps?.map((s, si) => ({
            ...s,
            status: si < i ? "complete" : si === i ? "loading" : s.status,
          })),
        }));

      // Mark steps 0..upTo as complete
      const completeUpTo = (upTo: number) =>
        updateMsg((m) => ({
          ...m,
          steps: m.steps?.map((s, si) => ({
            ...s,
            status: si <= upTo ? "complete" : s.status,
          })),
        }));

      while (true) {
        const { done, value } = (await reader?.read()) || {
          done: true,
          value: undefined,
        };
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") break;
          if (!raw) continue;
          try {
            const d = JSON.parse(raw);
            if (d.evidence_hierarchy?.levels) hierarchy = d.evidence_hierarchy;

            // ── Step 0: Query Analysis ──
            if (d.step === "query_expansion" && d.status === "running") {
              activateStep(0);
            } else if (d.step === "query_expansion" && d.status === "complete") {
              completeUpTo(0);

              // ── Step 1: Retrieval ──
            } else if (d.step === "retrieval" && d.status === "running") {
              activateStep(1);
            } else if (d.step === "retrieval" && d.status === "complete") {
              completeUpTo(1);

              // ── Step 2: Reranking ──
            } else if (d.step === "reranking" && d.status === "running") {
              activateStep(2);
            } else if (d.step === "reranking" && d.status === "complete") {
              if (d.sources?.length) sources = d.sources;
              if (d.evidence_hierarchy?.levels) hierarchy = d.evidence_hierarchy;
              updateMsg((m) => ({
                ...m,
                sources,
                evidenceHierarchy: hierarchy,
                steps: m.steps?.map((s, si) => ({
                  ...s,
                  status: si <= 2 ? "complete" : s.status,
                })),
              }));

              // ── Step 3: PDF Check (runs in parallel with generation) ──
            } else if (d.step === "pdf_check" && d.status === "running") {
              activateStep(3);
            } else if (d.step === "pdf_check" && d.status === "complete") {
              if (d.sources?.length) sources = d.sources;
              if (d.evidence_hierarchy?.levels) hierarchy = d.evidence_hierarchy;
              updateMsg((m) => ({
                ...m,
                sources,
                evidenceHierarchy: hierarchy,
                steps: m.steps?.map((s, si) => ({
                  ...s,
                  status: si <= 3 ? "complete" : s.status,
                })),
              }));

              // ── Step 4: Generation (streaming tokens) ──
            } else if (d.step === "generation" && d.status === "running") {
              if (d.token) {
                answer += d.token;
                updateMsg((m) => ({
                  ...m,
                  content: answer,
                  steps: m.steps?.map((s, si) => ({
                    ...s,
                    status: si <= 3 ? "complete" : si === 4 ? "loading" : s.status,
                  })),
                }));
              } else {
                // generation started but no token yet
                activateStep(4);
              }
            } else if (d.step === "generation" && d.status === "complete") {
              completeUpTo(4);

              // ── Final complete event ──
            } else if (d.step === "complete") {
              if (d.answer) answer = d.answer;
              if (d.sources?.length) sources = d.sources;
              if (d.evidence_hierarchy?.levels) hierarchy = d.evidence_hierarchy;
              updateMsg((m) => ({
                ...m,
                content: answer,
                sources,
                evidenceHierarchy: hierarchy || m.evidenceHierarchy,
                steps: m.steps?.map((s) => ({ ...s, status: "complete" as const })),
              }));
            }
          } catch {
            /* skip bad SSE lines */
          }
        }
      }
    } catch (err) {
      console.error("Chat error:", err);
      setMessages((prev) => {
        const n = [...prev];
        n[n.length - 1].content =
          "⚠️ Failed to get a response. Check the API rewrite target and backend availability.";
        return n;
      });
    } finally {
      setIsLoading(false);
    }
  };

  /* ═══════════════════════ RENDER ═══════════════════════════ */

  return (
    <div className="app-container">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-logo">Elixir AI</div>
        <div className="sidebar-subtitle">Medical Research</div>

        <div className="sidebar-section-title">Suggested</div>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
          {SUGGESTED_QUERIES.map((q, i) => (
            <button
              key={i}
              className="suggestion-chip"
              style={{ textAlign: "left", fontSize: "0.78rem" }}
              onClick={() => {
                setInput(q);
              }}
            >
              {q}
            </button>
          ))}
        </div>

        <div className="sidebar-footer">
          Grounded in 1.2M+ peer-reviewed articles from PMC, PubMed &amp; DailyMed
        </div>
      </aside>

      {/* ── Main content area ── */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <main
          className="chat-main"
          style={{
            flex: pdfUrl ? "0 0 50%" : "1",
            transition: "flex 0.3s ease",
          }}
        >
          {/* Scrollable chat */}
          <div className="chat-scroll">
            <div className="chat-inner">
              {messages.length === 0 ? (
                <div className="empty-state">
                  <h1>How can I assist your research?</h1>
                  <p>
                    Ask complex medical questions backed by peer-reviewed
                    evidence from PMC, PubMed, and DailyMed.
                  </p>
                  <div className="suggestions">
                    {SUGGESTED_QUERIES.map((q, i) => (
                      <button
                        key={i}
                        className="suggestion-chip"
                        onClick={() => setInput(q)}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                messages.map((msg, idx) => {
                  const allSources = (msg.sources || [])
                    .map((s, i) => ({ source: s, citation: getCitationIndex(s, i) }))
                    .sort((a, b) => a.citation - b.citation)
                    .map((x) => x.source);
                  const drugSources = allSources.filter(
                    (s) =>
                      s.source === "dailymed" ||
                      s.pmcid?.startsWith("dailymed_"),
                  );
                  const hierarchy = msg.evidenceHierarchy;
                  const tab = msg.activeTab || "answer";

                  return (
                    <div key={idx} className="msg-block">
                      <div
                        className={`msg-role ${msg.role === "user"
                          ? "msg-role-user"
                          : "msg-role-assistant"
                          }`}
                      >
                        {msg.role === "user" ? "You" : "Elixir AI"}
                      </div>

                      {/* ── User message ── */}
                      {msg.role === "user" && (
                        <div className="msg-user-content">{msg.content}</div>
                      )}

                      {/* ── Assistant message ── */}
                      {msg.role === "assistant" && (
                        <div className="msg-assistant-card">
                          {/* Tabs */}
                          {(msg.content || allSources.length > 0) && (
                            <div className="tab-bar">
                              <button
                                className={tab === "answer" ? "active" : ""}
                                onClick={() => setActiveTab(idx, "answer")}
                              >
                                ○ Answer
                              </button>
                              {drugSources.length > 0 && (
                                <button
                                  className={tab === "drugs" ? "active" : ""}
                                  onClick={() => setActiveTab(idx, "drugs")}
                                >
                                  💊 Drugs
                                  <span className="tab-count">
                                    {drugSources.length}
                                  </span>
                                </button>
                              )}
                              {allSources.length > 0 && (
                                <button
                                  className={
                                    tab === "references" ? "active" : ""
                                  }
                                  onClick={() =>
                                    setActiveTab(idx, "references")
                                  }
                                >
                                  📄 References
                                  <span className="tab-count">
                                    {allSources.length}
                                  </span>
                                </button>
                              )}
                            </div>
                          )}

                          {/* ── ANSWER TAB ── */}
                          {tab === "answer" && (
                            <>
                              {/* Steps indicator */}
                              {msg.steps && !msg.content && (
                                <div className="steps-grid">
                                  {msg.steps.map((step, si) => (
                                    <div
                                      key={si}
                                      className={`step-item ${step.status}`}
                                    >
                                      <div
                                        className={`step-dot ${step.status}`}
                                      >
                                        {step.status === "loading" && (
                                          <div className="spinner" />
                                        )}
                                        {step.status === "complete" && (
                                          <div className="complete-dot" />
                                        )}
                                      </div>
                                      <span>{step.title}</span>
                                    </div>
                                  ))}
                                </div>
                              )}

                              {/* Rendered markdown */}
                              {msg.content && (
                                <MarkdownWithCitations
                                  content={msg.content}
                                  sources={allSources}
                                  onCitationClick={() =>
                                    setActiveTab(idx, "references")
                                  }
                                />
                              )}
                            </>
                          )}

                          {/* ── DRUGS TAB ── */}
                          {tab === "drugs" && drugSources.length > 0 && (
                            <div
                              style={{
                                display: "flex",
                                flexDirection: "column",
                                gap: "0.75rem",
                              }}
                            >
                              {drugSources.map((s, si) => {
                                const dmCitation = getCitationIndex(s, si);
                                return (
                                <div key={si} className="drug-card">
                                  <h3>💊 [{dmCitation}] {s.title}</h3>
                                  <div
                                    style={{
                                      fontSize: "0.8rem",
                                      color: "var(--text-secondary)",
                                      marginBottom: "0.35rem",
                                    }}
                                  >
                                    {formatAuthors(s.authors)}
                                  </div>
                                  <a
                                    href={
                                      s.dailymed_url || getArticleUrl(s)
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    📋 View on DailyMed →
                                  </a>
                                </div>
                                );
                              })}
                            </div>
                          )}

                          {/* ── REFERENCES TAB ── */}
                          {tab === "references" && allSources.length > 0 && (
                            <div
                              style={{
                                display: "flex",
                                flexDirection: "column",
                                gap: "1rem",
                              }}
                            >
                              {/* Evidence hierarchy */}
                              {hierarchy?.levels?.length ? (
                                <div className="evidence-hierarchy-box">
                                  <div className="evidence-hierarchy-label">
                                    Evidence Hierarchy
                                  </div>
                                  <div className="evidence-pills">
                                    {hierarchy.levels.map((lvl) => (
                                      <span
                                        key={lvl.grade}
                                        className="evidence-badge"
                                        style={{
                                          color:
                                            EVIDENCE_COLORS[lvl.grade] ||
                                            "#64748b",
                                          borderColor:
                                            EVIDENCE_COLORS[lvl.grade] ||
                                            "#64748b",
                                        }}
                                        title={lvl.terms.join(", ")}
                                      >
                                        {lvl.grade} · L{lvl.level}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              ) : null}

                              {/* Source cards */}
                              {allSources.map((s, si) => {
                                const citationNumber = getCitationIndex(s, si);
                                const articleTypeBadge = getArticleTypeBadge(s);
                                return (
                                  <div
                                    key={`${s.pmcid || s.doi || s.title || "ref"}-${si}`}
                                    className="ref-card"
                                  >
                                    <div
                                      className="ref-number"
                                      style={{
                                        backgroundColor:
                                          CITATION_COLORS[
                                            (citationNumber - 1) % CITATION_COLORS.length
                                          ],
                                      }}
                                    >
                                      {citationNumber}
                                    </div>
                                    <div style={{ flex: 1, minWidth: 0 }}>
                                      <a
                                        href={getArticleUrl(s)}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="ref-title"
                                        style={{
                                          color:
                                            CITATION_COLORS[
                                              (citationNumber - 1) % CITATION_COLORS.length
                                            ],
                                        }}
                                      >
                                        {s.title}
                                      </a>
                                      <div className="ref-meta">
                                        {s.journal && (
                                          <span>{s.journal}. </span>
                                        )}
                                        {s.year && <span>{s.year}; </span>}
                                        {s.doi && (
                                          <a
                                            href={`https://doi.org/${s.doi.replace(/^https?:\/\/doi\.org\//, "")}`}
                                            target="_blank"
                                            rel="noreferrer"
                                          >
                                            doi:{s.doi}
                                          </a>
                                        )}
                                      </div>

                                      <div className="ref-badges">
                                        {s.evidence_grade && (
                                          <span
                                            className="evidence-badge"
                                            style={{
                                              color:
                                                EVIDENCE_COLORS[
                                                  s.evidence_grade
                                                ] || "#64748b",
                                              borderColor:
                                                EVIDENCE_COLORS[
                                                  s.evidence_grade
                                                ] || "#64748b",
                                            }}
                                            title={
                                              s.evidence_source
                                                ? `From ${s.evidence_source}`
                                                : "Evidence grade"
                                            }
                                          >
                                            {s.evidence_grade}
                                            {s.evidence_level
                                              ? ` · L${s.evidence_level}`
                                              : ""}
                                          </span>
                                        )}
                                        {articleTypeBadge && (
                                          <span
                                            className="badge-article-type"
                                            style={articleTypeBadge.style}
                                          >
                                            📋 {articleTypeBadge.label}
                                          </span>
                                        )}
                                        {s.pdf_url && (
                                          <button
                                            className="pdf-badge"
                                            onClick={() =>
                                              setPdfUrl(s.pdf_url!)
                                            }
                                          >
                                            📕 PDF
                                          </button>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* ── Input ── */}
          <div className="input-area">
            <form onSubmit={handleSubmit} className="input-form">
              <input
                type="text"
                className="input-field"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a medical research question…"
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="submit-btn"
              >
                {isLoading ? "…" : "Ask"}
              </button>
            </form>
            <div className="input-disclaimer">
              Elixir AI can make mistakes. Always verify with original sources.
            </div>
          </div>
        </main>

        {/* ── PDF Viewer ── */}
        {pdfUrl && (
          <div className="pdf-panel">
            <div className="pdf-header">
              <span>📄 PDF Viewer</span>
              <div className="pdf-actions">
                <a
                  href={pdfUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="pdf-open-btn"
                >
                  Open in Tab
                </a>
                <button
                  className="pdf-close-btn"
                  onClick={() => setPdfUrl(null)}
                >
                  ✕ Close
                </button>
              </div>
            </div>
            <iframe
              src={pdfUrl}
              style={{ flex: 1, width: "100%", border: "none" }}
              title="PDF Viewer"
            />
          </div>
        )}
      </div>
    </div>
  );
}
