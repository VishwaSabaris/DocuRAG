import {
  ArrowRight,
  BookOpen,
  FileSearch,
  Layers3,
  Search,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

import "./LandingPage.css";

interface LandingPageProps {
  onExplore: () => void;
}

export function LandingPage({ onExplore }: LandingPageProps) {
  return (
    <div className="landing-page">
      <header className="landing-nav">
        <button className="landing-brand" onClick={onExplore} aria-label="Open DocuRAG">
          <span className="landing-logo-mark" aria-hidden="true"><span /><span /><span /></span>
          <span>DocuRAG<span className="landing-brand-dot">.</span></span>
        </button>
        <nav className="landing-nav-links" aria-label="Main navigation">
          <a href="#how-it-works">How it works</a>
          <a href="#capabilities">Capabilities</a>
          <a href="#workflow">Workflow</a>
        </nav>
        <button className="landing-nav-cta" onClick={onExplore}>Explore DocuRAG <ArrowRight size={16} /></button>
      </header>

      <main>
        <section className="landing-hero">
          <div className="landing-hero-copy">
            <span className="landing-eyebrow"><Sparkles size={14} /> Document-grounded intelligence</span>
            <h1>Understand your documents.<span> Ask better questions.</span></h1>
            <p>Upload a PDF, ask natural-language questions, retrieve the relevant evidence, and inspect the source behind every grounded answer.</p>
            <div className="landing-hero-actions">
              <button className="landing-primary-button" onClick={onExplore}>Explore DocuRAG <ArrowRight size={18} /></button>
              <a className="landing-secondary-button" href="#how-it-works">See how it works</a>
            </div>
            <div className="landing-trust">
              <span><ShieldCheck size={15} /> Grounded answers</span>
              <span><Search size={15} /> Semantic retrieval</span>
              <span><BookOpen size={15} /> Source references</span>
            </div>
          </div>

          <div className="landing-hero-visual" aria-label="DocuRAG product preview">
            <div className="landing-orbit landing-orbit-one" />
            <div className="landing-orbit landing-orbit-two" />
            <div className="landing-preview-window">
              <div className="landing-preview-header">
                <div className="landing-window-dots"><i /><i /><i /></div>
                <span>DocuRAG workspace</span>
                <span className="landing-live">LIVE</span>
              </div>
              <div className="landing-preview-body">
                <aside className="landing-preview-sidebar">
                  <div className="landing-preview-mini-brand">DocuRAG</div>
                  <div className="landing-preview-new-chat">+ New chat</div>
                  <div className="landing-preview-label">DOCUMENTS</div>
                  <div className="landing-preview-document active"><span className="landing-doc-icon">PDF</span><span>research-paper.pdf</span></div>
                  <div className="landing-preview-document"><span className="landing-doc-icon">PDF</span><span>project-report.pdf</span></div>
                </aside>
                <div className="landing-preview-chat">
                  <div className="landing-preview-chat-top"><span>research-paper.pdf</span><small>Ready</small></div>
                  <div className="landing-preview-message user">What is the main contribution?</div>
                  <div className="landing-preview-message assistant">
                    <strong>DocuRAG</strong>
                    <p>The document proposes a retrieval-driven approach for improving grounded responses.</p>
                    <div className="landing-preview-source"><span>Source</span><b>Page 12</b><span>Reranked evidence</span></div>
                  </div>
                  <div className="landing-preview-input"><span>Ask anything about this document...</span><b>↑</b></div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="landing-proof">
          <div><strong>PDF</strong><span>Document-first workflow</span></div>
          <div><strong>RAG</strong><span>Retrieve before generation</span></div>
          <div><strong>pgvector</strong><span>Semantic document search</span></div>
          <div><strong>Gemma 3</strong><span>Local model inference</span></div>
        </section>

        <section id="how-it-works" className="landing-section">
          <div className="landing-section-heading">
            <span className="landing-section-kicker">HOW IT WORKS</span>
            <h2>From PDF to grounded answer<span> in three steps.</span></h2>
            <p>A simple experience on the surface, with retrieval and evidence processing underneath.</p>
          </div>
          <div className="landing-step-grid">
            <article className="landing-step-card"><span className="landing-step-number">01</span><FileSearch size={24} /><h3>Upload & index</h3><p>Upload a PDF. DocuRAG extracts its content, chunks it and creates searchable embeddings.</p></article>
            <article className="landing-step-card featured"><span className="landing-step-number">02</span><Search size={24} /><h3>Retrieve evidence</h3><p>Ask a question and the retrieval pipeline finds the document sections that matter.</p></article>
            <article className="landing-step-card"><span className="landing-step-number">03</span><Layers3 size={24} /><h3>Answer with sources</h3><p>Receive a grounded response with source references so you can inspect the evidence.</p></article>
          </div>
        </section>

        <section id="capabilities" className="landing-capabilities">
          <div className="landing-capability-copy">
            <span className="landing-section-kicker">DOCUMENT INTELLIGENCE</span>
            <h2>One workspace to<span> search, read and ask.</span></h2>
            <p>DocuRAG brings your uploaded documents, conversations and source evidence together instead of making you jump between tools.</p>
            <button className="landing-primary-button" onClick={onExplore}>Open the workspace <ArrowRight size={18} /></button>
          </div>
          <div className="landing-capability-panel">
            <div className="landing-capability-row"><div className="landing-capability-icon"><Search size={19} /></div><div><strong>Search conversations</strong><span>Find previous chats by title or message content.</span></div></div>
            <div className="landing-capability-row"><div className="landing-capability-icon"><BookOpen size={19} /></div><div><strong>Open the original PDF</strong><span>View an uploaded document without leaving the workspace.</span></div></div>
            <div className="landing-capability-row"><div className="landing-capability-icon"><Layers3 size={19} /></div><div><strong>Trace the answer</strong><span>Keep source pages visible beside the conversation.</span></div></div>
          </div>
        </section>

        <section id="workflow" className="landing-workflow">
          <div className="landing-section-heading compact"><span className="landing-section-kicker">BUILT FOR REAL DOCUMENTS</span><h2>Upload. Ask. Verify.<span> Repeat.</span></h2></div>
          <div className="landing-workflow-frame">
            <div className="landing-workflow-item"><span>Document</span><strong>Research paper.pdf</strong><small>20 pages · indexed</small></div>
            <div className="landing-workflow-arrow">→</div>
            <div className="landing-workflow-item"><span>Question</span><strong>What is the proposed method?</strong><small>Natural-language query</small></div>
            <div className="landing-workflow-arrow">→</div>
            <div className="landing-workflow-item accent"><span>Grounded answer</span><strong>Response + source pages</strong><small>Evidence you can inspect</small></div>
          </div>
        </section>

        <section className="landing-final-cta">
          <span className="landing-section-kicker">READY TO EXPLORE?</span>
          <h2>Turn your next PDF<span> into a conversation.</span></h2>
          <p>Open DocuRAG and start with a document.</p>
          <button className="landing-primary-button light" onClick={onExplore}>Explore DocuRAG <ArrowRight size={18} /></button>
        </section>
      </main>

      <footer className="landing-footer"><span>DocuRAG<span>.</span></span><small>Document-grounded AI workspace</small></footer>
    </div>
  );
}
