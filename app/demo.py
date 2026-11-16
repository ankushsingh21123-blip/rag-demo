"""Self-contained live-preview page served at ``GET /demo``.

Inline CSS/JS, no build step, no external CDN dependency. The page calls
the same public API endpoints (``/health``, ``/documents``, ``/query``)
a visitor could call with curl. It makes no claims about any person,
client, or venture — the banner states plainly that this is a demo.
"""

DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rag-demo &mdash; live preview</title>
<style>
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0; background: #f6f7f9; color: #1a1d21; }
  main { max-width: 720px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  h2 { font-size: 1.1rem; margin: 0 0 10px; }
  .sub { color: #5b6470; margin: 0 0 16px; }
  .banner { background: #fff8e1; border: 1px solid #f0d98c; border-radius: 8px; padding: 12px 14px; font-size: 0.9rem; margin-bottom: 20px; }
  .card { background: #fff; border: 1px solid #e2e5ea; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  label { display: block; font-weight: 600; margin-bottom: 6px; }
  input[type="text"], textarea { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #ccd1d8; border-radius: 6px; font: inherit; margin-bottom: 10px; }
  textarea { min-height: 88px; resize: vertical; }
  button { background: #1f6feb; color: #fff; border: 0; border-radius: 6px; padding: 10px 18px; font: inherit; cursor: pointer; }
  button:disabled { opacity: 0.6; cursor: wait; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
  .chip { background: #eef2f7; color: #1a1d21; border: 1px solid #d6dde6; border-radius: 999px; padding: 6px 12px; font-size: 0.85rem; cursor: pointer; }
  .answer { white-space: pre-wrap; margin: 0 0 12px; }
  .citation { border-top: 1px solid #eef0f3; padding: 10px 0; font-size: 0.9rem; }
  .citation .meta { color: #5b6470; font-size: 0.8rem; margin-bottom: 2px; }
  .error { color: #b3261e; }
  .note { color: #5b6470; font-size: 0.85rem; }
  .status { font-size: 0.85rem; color: #5b6470; margin-bottom: 16px; }
  code { background: #eef0f3; padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<main>
  <h1>rag-demo &mdash; live preview</h1>
  <p class="sub">Ask a question. Answers are grounded in retrieved documents, with citations.</p>
  <div class="banner"><strong>Preview demo.</strong> Answers are grounded in pre-loaded sample documents, with citations and similarity scores. The status line below shows the active store and embedding provider; with no API key configured, a demo-only fallback embedder is used (word overlap only, no semantic understanding). Documents you add are stored in memory only and are lost when the server restarts.</div>
  <p class="status" id="status">Checking server status&hellip;</p>

  <section class="card">
    <label for="question">Question</label>
    <div class="chips">
      <button type="button" class="chip" data-q="Where is the Eiffel Tower?">Where is the Eiffel Tower?</button>
      <button type="button" class="chip" data-q="What is FastAPI?">What is FastAPI?</button>
      <button type="button" class="chip" data-q="How far away is the Moon?">How far away is the Moon?</button>
      <button type="button" class="chip" data-q="What is photosynthesis?">What is photosynthesis?</button>
    </div>
    <input type="text" id="question" placeholder="Ask about the sample documents&hellip;" autocomplete="off">
    <button id="ask-button" type="button">Ask</button>
  </section>

  <section class="card" id="results" hidden>
    <h2>Answer</h2>
    <p class="answer" id="answer"></p>
    <p class="note" id="providers"></p>
    <h2>Citations</h2>
    <div id="citations"></div>
  </section>
  <p class="error" id="query-error" hidden></p>

  <section class="card">
    <h2>Add a document</h2>
    <form id="ingest-form">
      <label for="doc-title">Title</label>
      <input type="text" id="doc-title" required>
      <label for="doc-source">Source</label>
      <input type="text" id="doc-source" required>
      <label for="doc-text">Text</label>
      <textarea id="doc-text" required></textarea>
      <button id="ingest-button" type="submit">Add document</button>
    </form>
    <p class="note">Uploads are session-only on serverless hosting: they live in the server&rsquo;s memory and disappear when the instance restarts.</p>
    <p id="ingest-status" class="note"></p>
  </section>

  <section class="card">
    <h2>Documents in the store</h2>
    <ul id="doc-list"></ul>
  </section>
  <p class="note">API: <code>GET /health</code> &middot; <code>POST /documents</code> &middot; <code>GET /documents</code> &middot; <code>POST /query</code> &middot; <code>GET /demo</code>.</p>
</main>
<script>
const $ = (id) => document.getElementById(id);

async function refreshStatus() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    $("status").textContent = "Store: " + data.store + " \\u00b7 Embeddings: " + data.embedding_provider;
  } catch (e) {
    $("status").textContent = "Server status unavailable.";
  }
}

async function refreshDocs() {
  try {
    const res = await fetch("/documents");
    const docs = await res.json();
    const list = $("doc-list");
    list.innerHTML = "";
    for (const doc of docs) {
      const li = document.createElement("li");
      li.textContent = doc.title + " (" + doc.source + ", " + doc.chunk_count + " chunk" + (doc.chunk_count === 1 ? "" : "s") + ")";
      list.appendChild(li);
    }
  } catch (e) {
    $("doc-list").textContent = "Could not load documents.";
  }
}

async function ask() {
  const question = $("question").value.trim();
  if (!question) return;
  const button = $("ask-button");
  button.disabled = true;
  $("query-error").hidden = true;
  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: question, top_k: 3})
    });
    if (!res.ok) throw new Error("Query failed (" + res.status + ")");
    const data = await res.json();
    $("answer").textContent = data.answer;
    $("providers").textContent = "Answer provider: " + data.answer_provider + " \\u00b7 Embedding provider: " + data.embedding_provider;
    const box = $("citations");
    box.innerHTML = "";
    for (const c of data.citations) {
      const div = document.createElement("div");
      div.className = "citation";
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = "[" + c.rank + "] " + c.title + " \\u2014 " + c.source + " \\u00b7 score " + c.score.toFixed(3);
      const text = document.createElement("div");
      text.textContent = c.text;
      div.appendChild(meta);
      div.appendChild(text);
      box.appendChild(div);
    }
    if (data.citations.length === 0) {
      box.textContent = "No citations \\u2014 the store is empty.";
    }
    $("results").hidden = false;
  } catch (e) {
    const err = $("query-error");
    err.textContent = "Something went wrong: " + e.message;
    err.hidden = false;
  } finally {
    button.disabled = false;
  }
}

$("ask-button").addEventListener("click", ask);
$("question").addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => { $("question").value = chip.dataset.q; ask(); });
});

$("ingest-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = $("ingest-button");
  button.disabled = true;
  try {
    const res = await fetch("/documents", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        title: $("doc-title").value,
        source: $("doc-source").value,
        text: $("doc-text").value
      })
    });
    if (!res.ok) throw new Error("Ingest failed (" + res.status + ")");
    const data = await res.json();
    $("ingest-status").textContent = "Added \\u201c" + data.title + "\\u201d (" + data.chunk_count + " chunk" + (data.chunk_count === 1 ? "" : "s") + ").";
    e.target.reset();
    refreshDocs();
  } catch (err) {
    $("ingest-status").textContent = "Something went wrong: " + err.message;
  } finally {
    button.disabled = false;
  }
});

refreshStatus();
refreshDocs();
</script>
</body>
</html>
"""
