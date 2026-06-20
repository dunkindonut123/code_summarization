const docBody = document.body;
window.APP_STATE = {
  ready: docBody.dataset.appReady === "true",
  loading: docBody.dataset.appLoading === "true",
  error: docBody.dataset.appError || null,
};

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const fileMeta = document.getElementById("file-meta");
const submitBtn = document.getElementById("submit-btn");
const loadingEl = document.getElementById("loading");
const errorEl = document.getElementById("error");
const resultsEl = document.getElementById("results");
const resultsMeta = document.getElementById("results-meta");
const resultsGrid = document.getElementById("results-grid");
const statusBadge = document.getElementById("status-badge");
const header = document.querySelector(".site-header");

let selectedFile = null;

/* ---------- Header shadow on scroll ---------- */
function onScroll() {
  if (window.scrollY > 10) header.classList.add("scrolled");
  else header.classList.remove("scrolled");
}
window.addEventListener("scroll", onScroll, { passive: true });
onScroll();

/* ---------- Scroll reveal ---------- */
const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("in");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 }
);
document.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el));

/* ---------- Active nav link on scroll ---------- */
const navLinks = Array.from(document.querySelectorAll(".nav-link"));
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

const navObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute("id");
        navLinks.forEach((l) =>
          l.classList.toggle("active", l.getAttribute("href") === `#${id}`)
        );
      }
    });
  },
  { rootMargin: "-45% 0px -50% 0px" }
);
sections.forEach((s) => navObserver.observe(s));

/* ---------- Expandable model cards ---------- */
document.querySelectorAll(".model-card-head").forEach((head) => {
  head.addEventListener("click", () => {
    const card = head.closest(".model-card");
    const open = card.classList.toggle("open");
    head.setAttribute("aria-expanded", String(open));
  });
});

/* ---------- Upload helpers ---------- */
function setError(message) {
  if (!message) {
    errorEl.classList.add("hidden");
    errorEl.textContent = "";
    return;
  }
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function updateFileMeta(file) {
  selectedFile = file;
  fileMeta.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  fileMeta.classList.remove("hidden");
  submitBtn.disabled = !window.APP_STATE.ready;
}

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) updateFileMeta(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) updateFileMeta(fileInput.files[0]);
});

/* ---------- Submit ---------- */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;

  if (!window.APP_STATE.ready) {
    setError("Models are not ready yet. Please wait for startup to finish.");
    return;
  }

  setError("");
  resultsEl.classList.add("hidden");
  loadingEl.classList.remove("hidden");
  submitBtn.disabled = true;

  const payload = new FormData();
  payload.append("file", selectedFile);

  try {
    const res = await fetch("/api/summarize", { method: "POST", body: payload });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Summarization failed.");
    renderResults(data);
    resultsEl.classList.remove("hidden");
    resultsEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    setError(err.message);
  } finally {
    loadingEl.classList.add("hidden");
    submitBtn.disabled = !window.APP_STATE.ready;
  }
});

/* ---------- Render results ---------- */
function renderResults(data) {
  resultsMeta.innerHTML = [
    ["File", data.filename],
    ["Methods", data.method_count],
    ["Statements", data.statement_count],
    ["Tokens", data.token_count],
    ["Total time", `${data.total_elapsed_ms} ms`],
  ]
    .map(([k, v]) => `<span class="chip"><strong>${k}:</strong> ${escapeHtml(String(v))}</span>`)
    .join("");

  const summaries = data.summaries;
  const withText = summaries.filter((s) => s.summary && s.summary.trim());
  const fastest = summaries.reduce((a, b) => (b.elapsed_ms < a.elapsed_ms ? b : a));
  const longest = withText.reduce(
    (a, b) => (b.summary.length > (a ? a.summary.length : -1) ? b : a),
    null
  );

  resultsGrid.innerHTML = summaries
    .map((s, i) => {
      const accent = s.accent || "#5b9cff";
      const isEmpty = !s.summary || !s.summary.trim();
      const badges = [];
      if (s === fastest) badges.push(`<span class="result-badge fastest">⚡ Fastest</span>`);
      if (longest && s === longest) badges.push(`<span class="result-badge longest">Most detail</span>`);

      const isCodet5 = s.model_id === "codet5" && s.methods && s.methods.length > 0;
      const bodyHtml = isEmpty
        ? "(no output produced)"
        : isCodet5
          ? renderMethodSummaries(s.methods)
          : escapeHtml(s.summary);
      const bodyClass = isEmpty ? "empty" : isCodet5 ? "method-list" : "";

      return `
        <article class="result-card" style="--accent:${accent}; animation-delay:${i * 70}ms">
          <div class="result-header">
            <div class="result-header-left">
              <span class="model-glyph" style="--c:${accent}">${escapeHtml(glyphFor(s.model_id, s.model))}</span>
              <div>
                <h3>${escapeHtml(s.model)}</h3>
                <div class="result-tier">${escapeHtml(s.tier)}</div>
              </div>
            </div>
            <span class="tag tag-${s.approach.toLowerCase()}">${escapeHtml(s.approach)}</span>
          </div>
          <div class="result-body ${bodyClass}">${bodyHtml}</div>
          <div class="result-footer">
            <span class="result-latency">⏱ ${s.elapsed_ms} ms ${badges.join(" ")}</span>
            <button class="copy-btn" type="button" ${isEmpty ? "disabled" : ""}>Copy</button>
          </div>
        </article>
      `;
    })
    .join("");

  resultsGrid.querySelectorAll(".copy-btn").forEach((btn, idx) => {
    btn.addEventListener("click", () => {
      const text = summaries[idx].summary || "";
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = "Copied";
        btn.classList.add("copied");
        setTimeout(() => {
          btn.textContent = "Copy";
          btn.classList.remove("copied");
        }, 1600);
      });
    });
  });
}

function renderMethodSummaries(methods) {
  if (!methods.length) return "";
  return methods
    .map((m) => {
      const text = (m.summary || "").trim();
      const summaryText = text || "(no output)";
      const emptyClass = text ? "" : " empty-method";
      return `
        <div class="method-summary${emptyClass}">
          <div class="method-name">${escapeHtml(m.name)}</div>
          <div class="method-text">${escapeHtml(summaryText)}</div>
        </div>
      `;
    })
    .join("");
}

function glyphFor(id, name) {
  const map = {
    tfidf: "TF",
    lexrank: "LR",
    sentence_transformers: "ST",
    codet5: "T5",
  };
  return map[id] || (name || "?").slice(0, 2).toUpperCase();
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ---------- Health polling ---------- */
function setStatus(state, text) {
  statusBadge.className = `status-badge ${state}`;
  statusBadge.querySelector(".status-text").textContent = text;
}

async function pollHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    window.APP_STATE.ready = data.ready;
    window.APP_STATE.loading = data.loading;
    window.APP_STATE.error = data.error;

    if (data.ready) {
      setStatus("ready", "Models ready");
      if (selectedFile) submitBtn.disabled = false;
    } else if (data.loading) {
      setStatus("loading", "Loading models…");
    } else {
      setStatus("error", data.error ? "Load failed" : "Not ready");
      if (data.error) setError(data.error);
    }

    if (!data.ready && !data.error) setTimeout(pollHealth, 3000);
  } catch {
    setTimeout(pollHealth, 5000);
  }
}

if (!window.APP_STATE.ready) {
  pollHealth();
} else if (selectedFile) {
  submitBtn.disabled = false;
}
