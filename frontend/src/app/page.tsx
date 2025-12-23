"use client";

import React, { useState, useEffect, useRef } from 'react';

// Types for medical RAG
interface Source {
  pmcid: string;
  title: string;
  journal: string;
  authors: string[];
  year: number;
  doi?: string;
  pdf_url?: string;
  article_type?: string;
  dailymed_url?: string;  // DailyMed drug label URL
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  steps?: { title: string; status: 'pending' | 'loading' | 'complete' }[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null); // For embedded PDF viewer
  const [referencesExpanded, setReferencesExpanded] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setPdfUrl(null); // Close PDF viewer on new query

    // Add placeholder assistant message with steps
    const assistantMessage: Message = {
      role: 'assistant',
      content: '',
      steps: [
        { title: 'Analyzing Query', status: 'loading' },
        { title: 'Retrieving Articles', status: 'pending' },
        { title: 'Reranking Evidence', status: 'pending' },
        { title: 'Checking Source PDFs', status: 'pending' },
        { title: 'Synthesizing Answer', status: 'pending' }
      ]
    };
    setMessages(prev => [...prev, assistantMessage]);

    try {
      const response = await fetch('http://localhost:8000/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input, stream: true }),
      });

      if (!response.ok) throw new Error('API request failed');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      let currentAnswer = '';
      let currentSources: Source[] = [];
      let buffer = '';

      while (true) {
        const { done, value } = await reader?.read() || { done: true, value: undefined };
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') break;
            if (!dataStr) continue;

            try {
              const data = JSON.parse(dataStr);

              if (data.step === 'query_expansion' && data.status === 'running') {
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastIdx = newMessages.length - 1;
                  if (newMessages[lastIdx]?.steps) {
                    newMessages[lastIdx].steps = newMessages[lastIdx].steps?.map((s, i) =>
                      i === 0 ? { ...s, status: 'loading' } : s
                    );
                  }
                  return newMessages;
                });
              } else if (data.step === 'retrieval' && data.status === 'running') {
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastIdx = newMessages.length - 1;
                  if (newMessages[lastIdx]?.steps) {
                    newMessages[lastIdx].steps = newMessages[lastIdx].steps?.map((s, i) =>
                      i <= 1 ? { ...s, status: i === 1 ? 'loading' : 'complete' } : s
                    );
                  }
                  return newMessages;
                });
              } else if (data.step === 'reranking' && data.status === 'complete') {
                if (data.sources) {
                  currentSources = data.sources;
                }
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastIdx = newMessages.length - 1;
                  if (newMessages[lastIdx]?.steps) {
                    newMessages[lastIdx].steps = newMessages[lastIdx].steps?.map((s, i) =>
                      i <= 2 ? { ...s, status: i === 2 ? 'complete' : s.status } : s
                    );
                    newMessages[lastIdx].sources = currentSources;
                  }
                  return newMessages;
                });
              } else if (data.step === 'generation' && data.status === 'running') {
                if (data.token) {
                  currentAnswer += data.token;
                  setMessages(prev => {
                    const newMessages = [...prev];
                    const lastIdx = newMessages.length - 1;
                    newMessages[lastIdx] = {
                      ...newMessages[lastIdx],
                      content: currentAnswer,
                      steps: newMessages[lastIdx].steps?.map((s, i) =>
                        i <= 4 ? { ...s, status: i === 4 ? 'loading' : 'complete' } : s
                      )
                    };
                    return newMessages;
                  });
                } else {
                  setMessages(prev => {
                    const newMessages = [...prev];
                    const lastIdx = newMessages.length - 1;
                    if (newMessages[lastIdx]?.steps) {
                      newMessages[lastIdx].steps = newMessages[lastIdx].steps?.map((s, i) =>
                        i <= 4 ? { ...s, status: i === 4 ? 'loading' : 'complete' } : s
                      );
                    }
                    return newMessages;
                  });
                }
              } else if (data.step === 'pdf_check' && data.status === 'running') {
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastIdx = newMessages.length - 1;
                  if (newMessages[lastIdx]?.steps) {
                    newMessages[lastIdx].steps = newMessages[lastIdx].steps?.map((s, i) =>
                      i <= 3 ? { ...s, status: i === 3 ? 'loading' : 'complete' } : s
                    );
                  }
                  return newMessages;
                });
              } else if (data.step === 'pdf_check' && data.status === 'complete') {
                if (data.sources) {
                  currentSources = data.sources;
                }
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastIdx = newMessages.length - 1;
                  if (newMessages[lastIdx]?.steps) {
                    newMessages[lastIdx].steps = newMessages[lastIdx].steps?.map((s, i) =>
                      i === 3 ? { ...s, status: 'complete' } : s
                    );
                    newMessages[lastIdx].sources = currentSources;
                  }
                  return newMessages;
                });
              } else if (data.step === 'generation' && data.status === 'complete') {
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastIdx = newMessages.length - 1;
                  if (newMessages[lastIdx]?.steps) {
                    newMessages[lastIdx].steps = newMessages[lastIdx].steps?.map((s, i) =>
                      i === 4 ? { ...s, status: 'complete' } : s
                    );
                  }
                  return newMessages;
                });
              } else if (data.step === 'complete') {
                currentAnswer = data.answer || currentAnswer;
                if (data.sources && data.sources.length > 0) {
                  currentSources = data.sources;
                }

                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastIdx = newMessages.length - 1;
                  newMessages[lastIdx] = {
                    ...newMessages[lastIdx],
                    content: currentAnswer,
                    sources: currentSources,
                    steps: newMessages[lastIdx].steps?.map(s => ({ ...s, status: 'complete' }))
                  };
                  return newMessages;
                });
              }
            } catch (e) {
              if (dataStr.length > 10) {
                console.warn('SSE parse warning:', dataStr.substring(0, 100) + '...');
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => {
        const newMessages = [...prev];
        const lastIdx = newMessages.length - 1;
        newMessages[lastIdx].content = 'Error: Failed to get response from medical assistant.';
        return newMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Format authors for display - handles both string[] and {name: string}[]
  const formatAuthors = (authors: (string | { name: string })[]): string => {
    if (!authors || authors.length === 0) return 'Unknown Authors';

    // Normalize authors to strings
    const authorStrings = authors.map(a => typeof a === 'string' ? a : a?.name || 'Unknown');

    if (authorStrings.length === 1) return authorStrings[0];
    if (authorStrings.length === 2) return `${authorStrings[0]}, ${authorStrings[1]}`;
    return `${authorStrings[0]}, ${authorStrings[1]}, et al.`;
  };


  // Get article type badge color
  const getTypeBadgeColor = (type: string): string => {
    switch (type?.toLowerCase()) {
      case 'practice guideline':
      case 'guideline':
        return '#10b981'; // green
      case 'systematic_review':
      case 'meta_analysis':
        return '#8b5cf6'; // purple
      case 'clinical_trial':
        return '#f59e0b'; // amber
      case 'review_article':
        return '#3b82f6'; // blue
      default:
        return '#64748b'; // gray
    }
  };

  // Get the latest assistant message with sources
  const lastAssistantMessage = [...messages].reverse().find(m => m.role === 'assistant' && m.sources && m.sources.length > 0);

  return (
    <div className="app-container" style={{ display: 'flex', height: '100vh' }}>
      {/* Sidebar */}
      <aside className="sidebar" style={{ width: '200px', flexShrink: 0 }}>
        <h2 className="title-gradient" style={{ marginBottom: '2rem', fontSize: '1.5rem' }}>Elixir AI</h2>
        <nav style={{ flex: 1 }}>
          <div style={{ color: 'var(--secondary)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1rem' }}>
            History
          </div>
          <div style={{ fontSize: '0.9rem', color: '#64748b' }}>No recent searches</div>
        </nav>
        <div style={{ padding: '1rem', borderTop: '1px solid var(--border)', fontSize: '0.8rem', color: 'var(--secondary)' }}>
          Grounded in 1.2M+ PMC Articles
        </div>
      </aside>

      {/* Main Content Area - Flex container for chat and PDF */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Chat Main - Adjusts width based on PDF viewer */}
        <main className="chat-main" style={{ flex: pdfUrl ? '0 0 50%' : '1', display: 'flex', flexDirection: 'column', overflow: 'hidden', transition: 'flex 0.3s ease' }}>
          {/* Scrollable chat area */}
          <div
            ref={chatContainerRef}
            style={{ flex: 1, overflowY: 'auto', padding: '2rem' }}
          >
            <div style={{ maxWidth: '800px', margin: '0 auto' }}>
              {messages.length === 0 ? (
                <div style={{ marginTop: '20vh', textAlign: 'center' }}>
                  <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }} className="title-gradient">How can I assist your research today?</h1>
                  <p style={{ color: 'var(--secondary)', fontSize: '1.1rem' }}>Ask complex medical questions backed by peer-reviewed evidence.</p>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div key={idx} style={{ marginBottom: '2rem', animation: 'fadeIn 0.3s ease-in-out' }}>
                    <div style={{ fontWeight: 600, color: msg.role === 'user' ? '#1e293b' : 'var(--primary)', marginBottom: '0.5rem' }}>
                      {msg.role === 'user' ? 'You' : 'Medical Assistant'}
                    </div>
                    <div className={msg.role === 'assistant' ? 'glass' : ''} style={{
                      padding: msg.role === 'assistant' ? '1.5rem' : '0',
                      borderRadius: '1rem',
                      whiteSpace: 'pre-wrap'
                    }}>
                      {msg.content}

                      {/* Steps indicator */}
                      {msg.role === 'assistant' && msg.steps && !msg.content && (
                        <div style={{ display: 'grid', gap: '0.75rem', marginTop: '1rem' }}>
                          {msg.steps.map((step, sIdx) => (
                            <div key={sIdx} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: step.status === 'pending' ? '#94a3b8' : '#1e293b' }}>
                              <div style={{
                                width: '18px', height: '18px', borderRadius: '50%',
                                border: `2px solid ${step.status === 'loading' ? 'var(--primary)' : step.status === 'complete' ? 'var(--medical-teal)' : '#e2e8f0'}`,
                                display: 'flex', alignItems: 'center', justifyContent: 'center'
                              }}>
                                {step.status === 'loading' && <div className="spinner" style={{ width: '8px', height: '8px', border: '2px solid var(--primary)', borderRightColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />}
                                {step.status === 'complete' && <div style={{ width: '8px', height: '8px', backgroundColor: 'var(--medical-teal)', borderRadius: '50%' }} />}
                              </div>
                              <span style={{ fontSize: '0.9rem' }}>{step.title}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* References Section - Numbered List Format */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
                          <div
                            onClick={() => setReferencesExpanded(!referencesExpanded)}
                            style={{
                              display: 'flex', alignItems: 'center', gap: '0.5rem',
                              cursor: 'pointer', userSelect: 'none',
                              marginBottom: referencesExpanded ? '1rem' : 0
                            }}
                          >
                            <span style={{ fontSize: '1.1rem' }}>📋</span>
                            <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>References</span>
                            <span style={{ color: 'var(--secondary)', fontSize: '0.85rem' }}>({msg.sources.length})</span>
                            <span style={{ marginLeft: 'auto', color: 'var(--secondary)' }}>{referencesExpanded ? '▲' : '▼'}</span>
                          </div>

                          {referencesExpanded && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                              {msg.sources.map((source, sIdx) => {
                                const getArticleUrl = () => {
                                  if (source.dailymed_url) return source.dailymed_url;
                                  if (source.doi) {
                                    const cleanDoi = source.doi.replace(/^https?:\/\/doi\.org\//, '');
                                    return `https://doi.org/${cleanDoi}`;
                                  }
                                  if (source.pmcid && !source.pmcid.toLowerCase().includes('dailymed')) {
                                    const pmcid = source.pmcid.toUpperCase().startsWith('PMC') ? source.pmcid : `PMC${source.pmcid}`;
                                    return `https://www.ncbi.nlm.nih.gov/pmc/articles/${pmcid}/`;
                                  }
                                  return '#';
                                };

                                return (
                                  <div key={sIdx} style={{ display: 'flex', gap: '1rem', position: 'relative' }}>
                                    {/* Number and Likes */}
                                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem', width: '24px', flexShrink: 0 }}>
                                      <span style={{ fontSize: '0.9rem', color: '#64748b', fontWeight: 500 }}>{sIdx + 1}.</span>
                                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginTop: '0.2rem' }}>
                                        <button className="icon-btn-small" style={{ opacity: 0.5 }}>👍</button>
                                        <button className="icon-btn-small" style={{ opacity: 0.5 }}>👎</button>
                                      </div>
                                    </div>

                                    {/* Content */}
                                    <div style={{ flex: 1 }}>
                                      {/* Title */}
                                      <a
                                        href={getArticleUrl()}
                                        target="_blank"
                                        rel="noreferrer"
                                        style={{
                                          color: '#f97316', // Orange from screenshot
                                          textDecoration: 'none',
                                          fontWeight: 600,
                                          fontSize: '0.95rem',
                                          lineHeight: '1.4',
                                          display: 'inline-block',
                                          marginBottom: '0.25rem'
                                        }}
                                      >
                                        {source.title}
                                      </a>

                                      {/* Authors */}
                                      <div style={{ color: '#475569', fontSize: '0.85rem', marginBottom: '0.15rem' }}>
                                        {formatAuthors(source.authors)}
                                      </div>

                                      {/* Journal and DOI */}
                                      <div style={{ color: '#64748b', fontSize: '0.85rem', lineHeight: '1.4' }}>
                                        {source.journal && (
                                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
                                            {source.journal.toLowerCase().includes('cochrane') && <span>🌐</span>}
                                            {source.journal}.
                                          </span>
                                        )}
                                        {source.year && <span> {source.year}; </span>}
                                        {source.doi && (
                                          <a
                                            href={`https://doi.org/${source.doi.replace(/^https?:\/\/doi\.org\//, '')}`}
                                            target="_blank"
                                            rel="noreferrer"
                                            style={{ color: '#64748b', textDecoration: 'none' }}
                                          >
                                            doi:{source.doi}
                                          </a>
                                        )}
                                      </div>

                                      {/* Badge and PDF */}
                                      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: '0.5rem' }}>
                                        {source.article_type && (
                                          <span style={{
                                            display: 'inline-flex', alignItems: 'center', gap: '0.35rem',
                                            color: '#64748b',
                                            fontSize: '0.8rem',
                                            fontWeight: 500
                                          }}>
                                            <span style={{ fontSize: '1rem' }}>📋</span>
                                            {source.article_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                                          </span>
                                        )}

                                        {source.pdf_url && (
                                          <button
                                            onClick={() => setPdfUrl(source.pdf_url!)}
                                            style={{
                                              display: 'inline-flex', alignItems: 'center', gap: '0.35rem',
                                              backgroundColor: '#dc2626',
                                              color: 'white',
                                              fontSize: '0.75rem',
                                              padding: '0.2rem 0.6rem',
                                              borderRadius: '0.375rem',
                                              fontWeight: 600,
                                              border: 'none',
                                              cursor: 'pointer',
                                              boxShadow: '0 1px 2px rgba(0,0,0,0.1)'
                                            }}
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
                  </div>
                ))
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Area */}
          <div style={{ padding: '1.5rem 2rem', borderTop: '1px solid var(--border)', background: 'rgba(255,255,255,0.8)', flexShrink: 0 }}>
            <form onSubmit={handleSubmit} style={{ maxWidth: '800px', margin: '0 auto', position: 'relative' }}>


              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Query medical literature..."
                style={{
                  width: '100%',
                  padding: '1.25rem 4rem 1.25rem 1.5rem',
                  borderRadius: '1rem',
                  border: '1px solid var(--border)',
                  outline: 'none',
                  fontSize: '1rem',
                  boxShadow: 'var(--soft-shadow)'
                }}
                className="glass"
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                style={{
                  position: 'absolute',
                  right: '0.75rem',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  padding: '0.6rem 1.25rem',
                  backgroundColor: 'var(--primary)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '0.75rem',
                  cursor: 'pointer',
                  opacity: (isLoading || !input.trim()) ? 0.5 : 1
                }}
              >
                Ask
              </button>
            </form>
            <div style={{ textAlign: 'center', marginTop: '0.75rem', fontSize: '0.75rem', color: 'var(--secondary)' }}>
              Medical AI can make mistakes. Verify critical information with original sources.
            </div>
          </div>
        </main>

        {/* PDF Viewer Panel */}
        {pdfUrl && (
          <div style={{
            flex: '0 0 50%',
            borderLeft: '1px solid var(--border)',
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: '#f8fafc'
          }}>
            {/* PDF Header */}
            <div style={{
              padding: '0.75rem 1rem',
              borderBottom: '1px solid var(--border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              backgroundColor: 'white'
            }}>
              <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>📄 PDF Viewer</span>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <a
                  href={pdfUrl}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    padding: '0.35rem 0.75rem',
                    fontSize: '0.8rem',
                    backgroundColor: '#3b82f6',
                    color: 'white',
                    borderRadius: '0.375rem',
                    textDecoration: 'none'
                  }}
                >
                  Open in New Tab
                </a>
                <button
                  onClick={() => setPdfUrl(null)}
                  style={{
                    padding: '0.35rem 0.75rem',
                    fontSize: '0.8rem',
                    backgroundColor: '#ef4444',
                    color: 'white',
                    border: 'none',
                    borderRadius: '0.375rem',
                    cursor: 'pointer'
                  }}
                >
                  ✕ Close
                </button>
              </div>
            </div>

            {/* PDF Embed */}
            <iframe
              src={pdfUrl}
              style={{
                flex: 1,
                width: '100%',
                border: 'none'
              }}
              title="PDF Viewer"
            />
          </div>
        )}
      </div>

      <style jsx global>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        .icon-btn-small {
          background: none;
          border: none;
          cursor: pointer;
          padding: 2px;
          font-size: 0.8rem;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: transform 0.1s ease;
        }
        .icon-btn-small:hover {
          transform: scale(1.2);
        }
      `}</style>
    </div>
  );
}
