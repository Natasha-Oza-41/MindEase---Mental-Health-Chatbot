/**
 * MindEase — Frontend
 *
 * Pure vanilla JS. No frameworks, no build step.
 * Talks to the FastAPI backend at /api/chat.
 */

// ── State ──────────────────────────────────────────────────────────
let sessionId = loadOrCreateSession();
let isWaiting  = false;

// ── DOM refs ───────────────────────────────────────────────────────
const chatArea       = document.getElementById('chatArea');
const messageInput   = document.getElementById('messageInput');
const sendBtn        = document.getElementById('sendBtn');
const sendIcon       = document.getElementById('sendIcon');
const sendSpinner    = document.getElementById('sendSpinner');
const clearBtn       = document.getElementById('clearBtn');
const sessionShort   = document.getElementById('sessionShort');
const crisisBanner   = document.getElementById('crisisBanner');
const crisisTitle    = document.getElementById('crisisTitle');
const crisisSubtitle = document.getElementById('crisisSubtitle');
const crisisResources = document.getElementById('crisisResources');
const crisisClose    = document.getElementById('crisisClose');
const charCount      = document.getElementById('charCount');
const themeToggle    = document.getElementById('themeToggle');

// ── Init ───────────────────────────────────────────────────────────
(function init() {
  initTheme();
  updateSessionDisplay();
  bindQuickPrompts();

  messageInput.addEventListener('input', onInputChange);
  messageInput.addEventListener('keydown', onKeyDown);
  sendBtn.addEventListener('click', sendMessage);
  clearBtn.addEventListener('click', newSession);
  crisisClose.addEventListener('click', () => crisisBanner.classList.add('hidden'));
  themeToggle.addEventListener('change', toggleTheme);
})();

// ── Theme ──────────────────────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem('mse_theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  themeToggle.checked = (saved === 'dark');
}

function toggleTheme() {
  const isDark = themeToggle.checked;
  const theme  = isDark ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('mse_theme', theme);
}

// ── Session helpers ────────────────────────────────────────────────
function loadOrCreateSession() {
  let id = sessionStorage.getItem('msc_session_id');
  if (!id) {
    id = 'sess-' + Math.random().toString(36).slice(2, 10) + '-' + Date.now().toString(36);
    sessionStorage.setItem('msc_session_id', id);
  }
  return id;
}

function updateSessionDisplay() {
  sessionShort.textContent = sessionId.slice(-6).toUpperCase();
}

async function newSession() {
  try {
    await fetch(`/api/chat/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  } catch (_) { /* ignore */ }

  sessionStorage.removeItem('msc_session_id');
  sessionId = loadOrCreateSession();
  updateSessionDisplay();

  chatArea.innerHTML = '';
  hideCrisisBanner();
  appendWelcome();
}

// ── Quick prompts ──────────────────────────────────────────────────
function bindQuickPrompts() {
  document.querySelectorAll('.qp-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (isWaiting) return;
      messageInput.value = btn.dataset.prompt;
      autoResize(messageInput);
      onInputChange();
      sendMessage();
    });
  });
}

// ── Input handling ─────────────────────────────────────────────────
function onInputChange() {
  autoResize(messageInput);
  const len = messageInput.value.length;
  charCount.textContent = `${len} / 2000`;
  charCount.classList.toggle('warn', len > 1800);
}

function onKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

// ── Core send flow ─────────────────────────────────────────────────
async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text || isWaiting) return;

  setWaiting(true);

  appendUserBubble(text);
  messageInput.value = '';
  autoResize(messageInput);
  charCount.textContent = '0 / 2000';

  const typingId = appendTypingIndicator();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });

    removeTypingIndicator(typingId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      appendBotBubble(
        err.detail || `Server error (${res.status}). Please try again.`,
        null, null, true
      );
      return;
    }

    const data = await res.json();
    appendBotBubble(data.response, data.sources, data.crisis_info);
    handleCrisisBanner(data.crisis_info);

  } catch (err) {
    removeTypingIndicator(typingId);
    appendBotBubble(
      'Could not reach the server. Make sure the backend is running and try again.',
      null, null, true
    );
  } finally {
    setWaiting(false);
    messageInput.focus();
  }
}

// ── Render helpers ─────────────────────────────────────────────────
function timeLabel() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function appendUserBubble(text) {
  const row = document.createElement('div');
  row.className = 'msg-row user-row';
  row.innerHTML = `
    <div class="avatar user-avatar" aria-hidden="true">You</div>
    <div style="display:flex;flex-direction:column;align-items:flex-end">
      <div class="bubble user-bubble">${escapeHtml(text).replace(/\n/g, '<br>')}</div>
      <span class="msg-time">${timeLabel()}</span>
    </div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
}

function appendBotBubble(text, sources, crisisInfo, isError = false) {
  const row = document.createElement('div');
  row.className = 'msg-row bot-row';

  let badgeHtml = '';
  if (crisisInfo && ['medium', 'low'].includes(crisisInfo.severity)) {
    const cls = crisisInfo.severity === 'medium' ? 'badge-medium' : 'badge-low';
    badgeHtml = `<span class="severity-badge ${cls}">${crisisInfo.severity} concern detected</span><br>`;
  }

  let sourcesHtml = '';
  if (sources && sources.length > 0) {
    const tags = sources.map(s =>
      `<span class="source-tag">&#128196; ${escapeHtml(s)}</span>`
    ).join('');
    sourcesHtml = `<div class="sources-row">${tags}</div>`;
  }

  const bubbleCls = 'bubble bot-bubble' + (isError ? ' error-bubble' : '');
  const formatted = formatText(text);

  row.innerHTML = `
    <div class="avatar bot-avatar" aria-hidden="true">S</div>
    <div style="display:flex;flex-direction:column">
      <div class="${bubbleCls}">
        ${badgeHtml}${formatted}${sourcesHtml}
      </div>
      <span class="msg-time">${timeLabel()}</span>
    </div>
  `;

  chatArea.appendChild(row);
  scrollToBottom();
}

function appendTypingIndicator() {
  const id = 'typing-' + Date.now();
  const row = document.createElement('div');
  row.className = 'msg-row bot-row';
  row.id = id;
  row.innerHTML = `
    <div class="avatar bot-avatar" aria-hidden="true">S</div>
    <div class="bubble bot-bubble typing-bubble">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    </div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  document.getElementById(id)?.remove();
}

function appendWelcome() {
  const row = document.createElement('div');
  row.className = 'msg-row bot-row';
  row.id = 'welcomeMsg';
  row.innerHTML = `
    <div class="avatar bot-avatar" aria-hidden="true">S</div>
    <div class="bubble bot-bubble welcome-bubble">
      <p class="welcome-greeting">Hi, I'm <strong>Sage</strong> &#x1F331;</p>
      <p>I'm here to listen, support, and guide you — with empathy and evidence-based techniques. Everything stays private on your device.</p>
      <div class="quick-prompts">
        <span class="qp-label">Try asking:</span>
        <button class="qp-btn" data-prompt="I've been feeling anxious about work lately">Anxious about work</button>
        <button class="qp-btn" data-prompt="I can't sleep and feel overwhelmed">Can't sleep</button>
        <button class="qp-btn" data-prompt="I feel really down and hopeless">Feeling down</button>
      </div>
    </div>
  `;
  chatArea.appendChild(row);

  // Re-bind quick prompts in the new welcome message
  row.querySelectorAll('.qp-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (isWaiting) return;
      messageInput.value = btn.dataset.prompt;
      autoResize(messageInput);
      onInputChange();
      sendMessage();
    });
  });
}

// ── Crisis banner ──────────────────────────────────────────────────
function handleCrisisBanner(crisisInfo) {
  if (!crisisInfo || crisisInfo.severity === 'none') return;
  if (!['critical', 'high'].includes(crisisInfo.severity)) return;

  const isCritical = crisisInfo.severity === 'critical';

  crisisBanner.classList.remove('hidden', 'severity-high');
  if (!isCritical) crisisBanner.classList.add('severity-high');

  crisisTitle.textContent = isCritical
    ? 'Immediate Crisis Support Available'
    : 'Crisis Support Resources';

  crisisSubtitle.textContent = isCritical
    ? 'Please reach out to a crisis line right now — trained counselors are available 24/7.'
    : "It sounds like you're going through something really difficult. Real support is available.";

  crisisResources.innerHTML = (crisisInfo.resources || []).map(r => `
    <a class="resource-card" href="${escapeHtml(r.url)}" target="_blank" rel="noopener">
      <span class="resource-name">${escapeHtml(r.name)}</span>
      <span class="resource-contact">${escapeHtml(r.contact)}</span>
      <span class="resource-avail">${escapeHtml(r.available)}</span>
    </a>
  `).join('');

  crisisBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideCrisisBanner() {
  crisisBanner.classList.add('hidden');
  crisisBanner.classList.remove('severity-high');
}

// ── UI state ───────────────────────────────────────────────────────
function setWaiting(on) {
  isWaiting = on;
  sendBtn.disabled = on;
  messageInput.disabled = on;
  sendIcon.classList.toggle('hidden', on);
  sendSpinner.classList.toggle('hidden', !on);
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

// ── Text utilities ─────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g,  '&amp;')
    .replace(/</g,  '&lt;')
    .replace(/>/g,  '&gt;')
    .replace(/"/g,  '&quot;')
    .replace(/'/g,  '&#39;');
}

function formatText(text) {
  let html = escapeHtml(text);

  // **bold**
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // *italic*
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // `code`
  html = html.replace(/`(.+?)`/g, '<code>$1</code>');
  // double newline = paragraph
  html = html.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
  // horizontal rule
  html = html.replace(/---/g, '<hr style="border:none;border-top:1px solid var(--c-border);margin:10px 0">');

  if (!html.startsWith('<')) html = '<p>' + html + '</p>';

  return html;
}
