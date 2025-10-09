// ---- Config ----
const ROOT = "part1/matrix/";
// Set your panels here (you can add/remove freely)
const PANELS = ["pan1", "pan2", "pan3", "pan4"];

// ---- DOM ----
const panelSelect = document.getElementById("panel");
const statusEl = document.getElementById("status");
const augEl = document.getElementById("aug");
const HEl = document.getElementById("H");

// Populate dropdown
(function initPanelSelect() {
  PANELS.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p; opt.textContent = p;
    panelSelect.appendChild(opt);
  });
  // default to pan1 if present
  if (PANELS.includes("pan1")) panelSelect.value = "pan1";
})();

// ---- Utils ----
async function fetchText(path) {
  const resp = await fetch(path + "?v=" + Date.now());
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${path}`);
  return resp.text();
}
const trimSplitLines = t => t.replace(/\r/g,"").trim().split("\n");

// parse whitespace-separated numbers (or tokens)
function parseRows(text){
  return trimSplitLines(text).map(line => line.trim().split(/\s+/));
}

// format rows with per-column widths; right-align numbers
function formatColumns(rows, decimals=5){
  const cols = Math.max(...rows.map(r => r.length));
  const widths = Array(cols).fill(0);

  // compute widths using exponential formatting for numbers
  for (const r of rows) {
    for (let j = 0; j < cols; j++) {
      const x = r[j] ?? "";
      const v = isFinite(+x) ? (+x).toExponential(decimals) : x;
      widths[j] = Math.max(widths[j], String(v).length);
    }
  }

  // build formatted lines
  return rows.map(r => {
    return Array.from({length: cols}, (_, j) => {
      const x = r[j] ?? "";
      const v = isFinite(+x) ? (+x).toExponential(decimals) : x;
      return String(v).padStart(widths[j]);
    }).join(" ");
  });
}

function buildAugmentedColumns(Atext, btext){
  const Arows = parseRows(Atext);
  const Brows = parseRows(btext);
  const Bcol = Brows.map(r => r[0] ?? "");   // assume b is a column vector

  const Afmt = formatColumns(Arows, 5);      // compute widths per *panel*
  const maxA = Afmt.reduce((m, s) => Math.max(m, s.length), 0);
  const maxB = Bcol.reduce((m, s) => Math.max(m, (isFinite(+s)?(+s).toExponential(5):s).length), 0);

  // header
  const header = "A".padEnd(maxA) + " | " + "b";
  const sep    = "-".repeat(maxA) + "-+-" + "-".repeat(Math.max(1, maxB));

  // rows
  const lines = [];
  for (let i = 0; i < Math.max(Afmt.length, Bcol.length); i++) {
    const a = Afmt[i] ?? "";
    const b = Bcol[i] ?? "";
    const bfmt = isFinite(+b) ? (+b).toExponential(5) : b;
    lines.push(a.padEnd(maxA) + " | " + bfmt);
  }
  return [header, sep, ...lines].join("\n");
}

// ---- Loaders ----
async function loadAugmented(panel) {
  augEl.textContent = "Loading…";
  const Apath = `${ROOT}${panel}/A.txt`;
  const bpath = `${ROOT}${panel}/b.txt`;
  try {
    const [Atext, btext] = await Promise.all([fetchText(Apath), fetchText(bpath)]);
    augEl.textContent = buildAugmentedColumns(Atext, btext);
  } catch (e) {
    augEl.textContent = "Failed: " + e;
  }
}

async function loadH(panel) {
  HEl.textContent = "Loading…";
  const Hpath = `${ROOT}${panel}/H.txt`;
  try {
    const text = await fetchText(Hpath);
    // Format H as a column as well (nice if it's a vector)
    const rows = parseRows(text);
    const formatted = formatColumns(rows, 5).join("\n");
    HEl.textContent = formatted;
  } catch (e) {
    HEl.textContent = "Failed: " + e;
  }
}

async function loadAll() {
  const panel = panelSelect.value;
  statusEl.textContent = `Loading ${panel}…`;
  await Promise.all([loadAugmented(panel), loadH(panel)]);
  statusEl.textContent = "Loaded";
  setTimeout(() => (statusEl.textContent = ""), 1200);
}

// ---- Events ----
panelSelect.addEventListener("change", loadAll);
document.getElementById("refresh")?.addEventListener("click", loadAll);

// initial
loadAll();

