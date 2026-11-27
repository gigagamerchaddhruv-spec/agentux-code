/* ──────────────────────────────────────────────────────────────
   Agentux Web Console — client script
   Vanilla JS, SSE streaming, zero dependencies.
   ────────────────────────────────────────────────────────────── */

"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  busy: false,
  view: "chat",
  files: [],
  tools: [],
  checkpoints: [],
  sessions: [],
  currentFile: null,
};

/* ── Bootstrap ─────────────────────────────────────────────── */
async function boot() {
  try {
    const status = await api("/api/status");
    const modelLine = $("#model-line");
    modelLine.textContent = status.model || "—";
    modelLine.classList.remove("muted");
    const wsLine = $("#workspace-line");
    wsLine.textContent = shortPath(status.workspace);
    wsLine.title = status.workspace;
  } catch (e) {
    console.warn("status fetch failed", e);
  }
  loadTools();
  loadCheckpoints();
  loadSessions();
  switchView("chat");
}

/* ── API helper ────────────────────────────────────────────── */
async function api(path, opts = {}) {
  const res = await fetch(path, {
    method: opts.method || "GET",
    headers: { "Content-Type": "application/json" },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    const t = await res.text().catch(() => "");
    throw new Error(`${res.status} ${t}`);
  }
  return res.json();
}

function shortPath(p) {
  if (!p) return "—";
  const parts = p.split("/").filter(Boolean);
  return parts.length <= 3 ? p : "…" + parts.slice(-3).join("/");
}

/* ── Views ─────────────────────────────────────────────────── */
function switchView(name) {
  state.view = name;
  $$(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.view === name));
  $$(".view").forEach(v => v.classList.remove("active"));
  const el = $("#view-" + name);
  if (el) el.classList.add("active");

  const titles = {
    chat: ["Chat", "Ask anything. Tools run inline."],
    files: ["Workspace", "Browse and edit project files."],
    tools: ["Tools", "Available tools and their descriptions."],
    checkpoints: ["Checkpoints", "Snapshots of the working tree."],
    sessions: ["Sessions", "Saved conversation sessions."],
  };
  const [h, sub] = titles[name] || ["", ""];
  $("#view-title").textContent = h;
  $("#view-subtitle").textContent = sub;

  if (name === "files") renderFileTree();
  if (name === "checkpoints") loadCheckpoints();
  if (name === "sessions") loadSessions();
}

/* ── Chat ──────────────────────────────────────────────────── */
function addMessage(role, content = "") {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "You" : role === "assistant" ? "Agentux" : "system";
  const body = document.createElement("div");
  body.className = "body";
  if (content) renderMarkdown(body, content);
  wrap.append(who, body);
  $("#messages").append(wrap);
  scrollBottom();
  return { wrap, body };
}

function renderMarkdown(el, text) {
  // Very small markdown: code fences, inline code, tables, lists, bold/italic.
  const lines = text.split("\n");
  let i = 0;
  let block = null;

  function closeBlock() {
    if (block) { el.append(block); block = null; }
  }

  function ensureP() {
    closeBlock();
    block = document.createElement("p");
  }

  while (i < lines.length) {
    const line = lines[i];

    // Code fence
    const fence = line.match(/^```(\w*)$/);
    if (fence) {
      ensureP();
      closeBlock();
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      let j = i + 1;
      while (j < lines.length && !lines[j].match(/^```$/)) j++;
      code.textContent = lines.slice(i + 1, j).join("\n");
      pre.append(code);
      el.append(pre);
      i = j + 1;
      continue;
    }

    // Table
    if (line.includes("|") && i + 1 < lines.length && lines[i + 1].match(/^\|[\s:|-]+\|$/)) {
      ensureP();
      closeBlock();
      const table = document.createElement("table");
      const head = document.createElement("tr");
      line.split("|").slice(1, -1).forEach(c => {
        const th = document.createElement("th");
        th.textContent = c.trim();
        head.append(th);
      });
      table.append(head);
      i += 2;
      while (i < lines.length && lines[i].includes("|")) {
        const tr = document.createElement("tr");
        lines[i].split("|").slice(1, -1).forEach(c => {
          const td = document.createElement("td");
          td.innerHTML = inline(c.trim());
          tr.append(td);
        });
        table.append(tr);
        i++;
      }
      el.append(table);
      continue;
    }

    // Unordered list
    if (/^\s*[-*]\s/.test(line)) {
      ensureP();
      closeBlock();
      const ul = document.createElement("ul");
      while (i < lines.length && /^\s*[-*]\s/.test(lines[i])) {
        const li = document.createElement("li");
        li.innerHTML = inline(lines[i].replace(/^\s*[-*]\s/, ""));
        ul.append(li);
        i++;
      }
      el.append(ul);
      continue;
    }

    // Ordered list
    if (/^\s*\d+\.\s/.test(line)) {
      ensureP();
      closeBlock();
      const ol = document.createElement("ol");
      while (i < lines.length && /^\s*\d+\.\s/.test(lines[i])) {
        const li = document.createElement("li");
        li.innerHTML = inline(lines[i].replace(/^\s*\d+\.\s/, ""));
        ol.append(li);
        i++;
      }
      el.append(ol);
      continue;
    }

    // Blank line -> paragraph break
    if (line.trim() === "") {
      ensureP();
      closeBlock();
      i++;
      continue;
    }

    ensureP();
    block.append(document.createElement("span"));
    block.lastChild.innerHTML = inline(line);
    i++;
  }
  closeBlock();
}

function inline(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

function scrollBottom() {
  const m = $("#messages");
  m.scrollTop = m.scrollHeight;
}

function addToolCard(name, args, result, error) {
  const card = document.createElement("div");
  card.className = "tool-card" + (error ? " error" : "");
  const tag = error ? "error" : result ? "result" : "running";
  const argsStr = args ? JSON.stringify(args) : "";
  card.innerHTML = `
    <div class="tool-head">
      <span class="tool-tag">${tag}</span>
      <span class="tool-name">${escapeHtml(name)}</span>
    </div>
    ${argsStr ? `<div class="tool-args">${escapeHtml(argsStr)}</div>` : ""}
    ${result !== undefined ? `<div class="tool-result">${escapeHtml(result)}</div>` : ""}
  `;
  $("#messages").append(card);
  scrollBottom();
  return card;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

async function sendPrompt() {
  const input = $("#input");
  const prompt = input.value.trim();
  if (!prompt || state.busy) return;

  state.busy = true;
  $("#send-btn").disabled = true;
  input.value = "";
  autoGrow(input);

  addMessage("user", prompt);
  const assistant = addMessage("assistant");
  let buffer = "";

  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let done = false;

  while (!done) {
    const { value, done: d } = await reader.read();
    done = d;
    if (value) {
      const chunk = decoder.decode(value, { stream: true });
      const events = parseSSE(chunk);
      for (const ev of events) {
        handleSSE(ev, assistant, buffer);
      }
    }
  }

  state.busy = false;
  $("#send-btn").disabled = false;
  input.focus();
}

function parseSSE(chunk) {
  const events = [];
  const blocks = chunk.split(/\n\n/);
  for (const block of blocks) {
    if (!block.trim()) continue;
    let data = "";
    let event = "message";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (data) {
      try { events.push({ event, data: JSON.parse(data) }); }
      catch { events.push({ event, data: { raw: data } }); }
    }
  }
  return events;
}

function handleSSE(ev, assistant, buffer) {
  const d = ev.data;
  switch (ev.event) {
    case "chunk":
      buffer += d.delta || "";
      renderMarkdown(assistant.body, buffer);
      scrollBottom();
      break;
    case "tool_call":
      addToolCard(d.name, d.arguments);
      break;
    case "tool_result":
      const cards = $("#messages .tool-card.running");
      if (cards.length) {
        const last = cards[cards.length - 1];
        last.classList.remove("running");
        const res = document.createElement("div");
        res.className = "tool-result";
        res.textContent = d.result;
        last.append(res);
      } else {
        addToolCard(d.name, null, d.result);
      }
      scrollBottom();
      break;
    case "narration":
    case "reflection": {
      // Agent narration / reflection nudge. Show as a system line.
      const sys = addMessage("system");
      renderMarkdown(sys.body, d.message || "");
      scrollBottom();
      break;
    }
    case "assistant":
      // Final answer. The streaming already populated the assistant bubble.
      break;
    case "done":
      addMessage("system", "✓ Completed");
      break;
    case "error":
      const err = document.createElement("div");
      err.className = "tool-card error";
      err.innerHTML = `<div class="tool-head"><span class="tool-tag">error</span></div><div class="tool-result">${escapeHtml(d.error)}</div>`;
      $("#messages").append(err);
      scrollBottom();
      break;
    case "nudge":
      const n = document.createElement("div");
      n.className = "tool-card";
      n.style.borderColor = "var(--purple)";
      n.innerHTML = `<div class="tool-head" style="color:var(--purple)"><span class="tool-tag">enforced</span></div><div class="tool-args">${escapeHtml(d.message)}</div>`;
      $("#messages").append(n);
      scrollBottom();
      break;
  }
}

/* ── Files ─────────────────────────────────────────────────── */
async function loadFiles() {
  try {
    const data = await api("/api/files");
    state.files = data.files || [];
  } catch (e) {
    console.warn(e);
  }
}

function renderFileTree() {
  loadFiles().then(() => {
    const tree = $("#file-tree");
    tree.innerHTML = "";
    if (!state.files.length) {
      tree.innerHTML = '<div class="empty">No files found</div>';
      return;
    }
    const root = { name: "", children: {} };
    for (const f of state.files) {
      const parts = f.split("/");
      let node = root;
      for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        if (i === parts.length - 1) {
          node.files = node.files || [];
          node.files.push({ name: part, path: f });
        } else {
          node.children = node.children || {};
          node.children[part] = node.children[part] || { name: part, children: {}, files: [] };
          node = node.children[part];
        }
      }
    }
    renderNode(tree, root, 0);
  });
}

function renderNode(parent, node, depth) {
  const keys = Object.keys(node.children || {}).sort();
  for (const name of keys) {
    const child = node.children[name];
    const div = document.createElement("div");
    div.className = "tree-item";
    div.style.paddingLeft = (8 + depth * 14) + "px";
    div.innerHTML = `<span class="f-icon">📁</span><span class="indent"></span>${escapeHtml(name)}`;
    div.onclick = () => {
      $$(".tree-item").forEach(t => t.classList.remove("active"));
      div.classList.add("active");
    };
    parent.append(div);
    renderNode(div, child, depth + 1);
  }
  for (const f of node.files || []) {
    const div = document.createElement("div");
    div.className = "tree-item";
    div.style.paddingLeft = (8 + depth * 14) + "px";
    const icon = fileIcon(f.name);
    div.innerHTML = `<span class="f-icon">${icon}</span>${escapeHtml(f.name)}`;
    div.onclick = () => openFile(f.path, div);
    parent.append(div);
  }
}

function fileIcon(name) {
  const ext = name.split(".").pop().toLowerCase();
  const map = {
    js: "📄", mjs: "📄", cjs: "📄", ts: "📘", tsx: "📘", jsx: "📘",
    py: "🐍", rb: "💎", go: "🔵", rs: "🟠", java: "☕", kt: "🟣",
    c: "🧩", h: "🧩", cpp: "🧩", hpp: "🧩",
    md: "📝", txt: "📝", json: "🧾", yaml: "⚙️", yml: "⚙️", toml: "⚙️",
    html: "🌐", css: "🎨", scss: "🎨", sass: "🎨",
    sh: "⌨️", bash: "⌨️", zsh: "⌨️",
    sql: "🗄️", db: "🗄️", sqlite: "🗄️",
    png: "🖼️", jpg: "🖼️", svg: "🖼️", gif: "🖼️",
    pdf: "📕",
  };
  return map[ext] || "📄";
}

async function openFile(path, el) {
  if (el) {
    $$(".tree-item").forEach(t => t.classList.remove("active"));
    el.classList.add("active");
  }
  state.currentFile = path;
  $("#file-path").textContent = path;
  $("#file-content").textContent = "Loading…";
  $("#view-files").classList.add("active");
  try {
    const data = await api("/api/file?path=" + encodeURIComponent(path));
    $("#file-content").textContent = data.content || "(empty)";
  } catch (e) {
    $("#file-content").textContent = "Error: " + e.message;
  }
}

/* ── Tools ─────────────────────────────────────────────────── */
async function loadTools() {
  try {
    const data = await api("/api/tools");
    state.tools = data.tools || [];
    renderTools();
  } catch (e) {
    console.warn(e);
  }
}

function renderTools() {
  const grid = $("#tools-grid");
  grid.innerHTML = "";
  if (!state.tools.length) {
    grid.innerHTML = '<div class="empty">No tools registered</div>';
    return;
  }
  for (const t of state.tools) {
    const card = document.createElement("div");
    card.className = "tool-chip";
    card.innerHTML = `
      <div class="tool-chip-name">${escapeHtml(t.name)}</div>
      <div class="tool-chip-desc">${escapeHtml(t.description || "")}</div>
    `;
    grid.append(card);
  }
}

/* ── Checkpoints ───────────────────────────────────────────── */
async function loadCheckpoints() {
  try {
    const data = await api("/api/checkpoints");
    state.checkpoints = data.checkpoints || [];
    renderCheckpoints();
  } catch (e) {
    console.warn(e);
  }
}

function renderCheckpoints() {
  const box = $("#checkpoints");
  box.innerHTML = "";
  if (!state.checkpoints.length) {
    box.innerHTML = '<div class="empty">No checkpoints yet</div>';
    return;
  }
  for (const cp of state.checkpoints) {
    const card = document.createElement("div");
    card.className = "cp-card";
    card.innerHTML = `
      <div class="cp-meta">
        <span class="cp-id">${escapeHtml(cp.id)}</span>
        <span class="cp-desc">${escapeHtml(cp.description || "")}</span>
        <span class="cp-time">${escapeHtml(cp.time || "")} · ${cp.files_count || 0} files${cp.git_sha ? " · " + escapeHtml(cp.git_sha.slice(0,7)) : ""}</span>
      </div>
      <button class="btn btn-ghost btn-sm" data-id="${escapeHtml(cp.id)}">Rollback</button>
    `;
    card.querySelector("button").onclick = () => rollback(cp.id);
    box.append(card);
  }
}

async function rollback(id) {
  if (!confirm("Roll back to checkpoint " + id + "?")) return;
  try {
    const res = await api("/api/checkpoints/rollback", { method: "POST", body: { checkpoint_id: id } });
    alert(res.message);
    loadCheckpoints();
  } catch (e) {
    alert("Rollback failed: " + e.message);
  }
}

/* ── Sessions ──────────────────────────────────────────────── */
async function loadSessions() {
  try {
    const data = await api("/api/sessions");
    state.sessions = data.sessions || [];
    renderSessions();
  } catch (e) {
    console.warn(e);
  }
}

function renderSessions() {
  const box = $("#sessions");
  box.innerHTML = "";
  if (!state.sessions.length) {
    box.innerHTML = '<div class="empty">No saved sessions</div>';
    return;
  }
  for (const s of state.sessions) {
    const card = document.createElement("div");
    card.className = "sess-card";
    card.innerHTML = `
      <div class="sess-meta">
        <span class="sess-id">${escapeHtml(s.id || "")}</span>
        <span class="sess-title">${escapeHtml(s.title || "Untitled")}</span>
        <span class="sess-time">${escapeHtml(s.time || "")}</span>
      </div>
    `;
    box.append(card);
  }
}

/* ── Composer ──────────────────────────────────────────────── */
function autoGrow(ta) {
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
}

/* ── Wire up ───────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  $$(".nav-item").forEach(b => b.onclick = () => switchView(b.dataset.view));
  $("#clear-btn").onclick = () => {
    $("#messages").innerHTML = "";
  };
  $("#close-file-btn").onclick = () => switchView("chat");

  const form = $("#composer");
  const input = $("#input");
  form.onsubmit = (e) => { e.preventDefault(); sendPrompt(); };
  input.oninput = () => autoGrow(input);
  input.onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendPrompt();
    }
  };

  boot();
});