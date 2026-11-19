/* Help mode shared by the standard and strange views.

   IrgenHelpMode({help, button, docs, status}):
     help   : {params: {key: {t, s, b, d}}} (window.IRGEN_HELP)
     button : the toggle button
     docs   : docs page URL; links go to <docs>#b-<key> (basics) and #d-<key> (in depth)
     status : optional (text) => void for the page's status line
   Keys come from the hovered element: a .row's [data-path] control, else the nearest
   [data-help]. While help mode is on, pointing explains and clicking pins the card;
   nothing on the page is changed. Esc unpins / exits. */
"use strict";
window.IrgenHelpMode = function ({help, button, docs, status = () => {}}) {
  const css = `
body.helpmode, body.helpmode *{cursor:help!important}
.ihm-on{background:var(--accent2)!important;color:#111!important;border-color:var(--accent2)!important}
.ihm-hl{position:fixed;pointer-events:none;border:2px solid var(--accent2);border-radius:8px;background:color-mix(in srgb, var(--accent2) 8%, transparent);z-index:120;transition:all .08s}
.ihm-card{position:fixed;z-index:121;max-width:340px;background:var(--panel2);border:1px solid var(--accent2);border-radius:10px;padding:10px 12px;box-shadow:0 8px 30px #0009;font-size:12.5px;line-height:1.5;color:var(--text)}
.ihm-card h5{margin:0 0 4px;font-size:13.5px;color:var(--accent2)} .ihm-card h5 code{font-weight:400;color:var(--muted);font-size:10.5px;margin-left:6px}
.ihm-card p{margin:0 0 8px}
.ihm-card .links{display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:12px}
.ihm-card a{color:var(--accent);text-decoration:none;border-bottom:1px dotted var(--accent)} .ihm-card a:hover{color:var(--text)}
.ihm-card .pin{margin-left:auto;color:var(--muted);font-size:11px}
.ihm-hide{display:none!important}`;
  document.head.append(Object.assign(document.createElement("style"), {textContent: css}));
  const hl = Object.assign(document.createElement("div"), {className: "ihm-hl ihm-hide"});
  const card = Object.assign(document.createElement("div"), {className: "ihm-card ihm-hide"});
  card.setAttribute("role", "dialog"); card.setAttribute("aria-live", "polite");
  document.body.append(hl, card);
  const label = button.textContent;
  const st = {on: false, pinned: null, cur: null};

  function keyOf(t){
    if (!(t instanceof Element)) return null;
    const r = t.closest(".row");
    if (r){ const p = r.querySelector("[data-path]"); if (p && help.params[p.dataset.path]) return [p.dataset.path, r]; }
    const h = t.closest("[data-help]");
    if (h && help.params[h.dataset.help]) return [h.dataset.help, h];
    const p = t.closest("[data-path]");
    if (p && help.params[p.dataset.path]) return [p.dataset.path, p];
    return null;
  }
  function show(key, target, pinned){
    const e = help.params[key];
    st.cur = [key, target];
    card.innerHTML = `<h5>${e.t}${/^(ui|section)\./.test(key) ? "" : `<code>${key}</code>`}</h5><p>${e.s}</p>`
      + `<div class="links">${e.b ? `<a href="${docs}#b-${key}" target="_blank" rel="noopener">Basics ↗</a>` : ""}`
      + `${e.d ? `<a href="${docs}#d-${key}" target="_blank" rel="noopener">In depth ↗</a>` : ""}`
      + `<span class="pin">${pinned ? "pinned: click elsewhere / Esc" : "click to pin"}</span></div>`;
    const r = target.getBoundingClientRect();
    Object.assign(hl.style, {left: (r.left - 3) + "px", top: (r.top - 3) + "px", width: (r.width + 6) + "px", height: (r.height + 6) + "px"});
    hl.classList.remove("ihm-hide"); card.classList.remove("ihm-hide");
    const cw = card.offsetWidth, ch = card.offsetHeight, W = innerWidth, H = innerHeight;
    let x = r.right + 10, y = r.top;
    if (x + cw > W - 8){ x = Math.max(8, Math.min(r.left, W - cw - 8)); y = r.bottom + 8; }
    if (y + ch > H - 8) y = Math.max(8, r.top - ch - 8);
    card.style.left = x + "px"; card.style.top = y + "px";
  }
  function hide(){ card.classList.add("ihm-hide"); hl.classList.add("ihm-hide"); st.cur = null; }
  function set(on){
    st.on = on; st.pinned = null;
    document.body.classList.toggle("helpmode", on); button.classList.toggle("ihm-on", on);
    button.textContent = on ? "✕ Exit help" : label;
    if (on && help.params["ui.help"]) show("ui.help", button, false); else hide();
    status(on ? "help mode: point at any control" : "ready");
  }
  button.addEventListener("click", e => { e.stopPropagation(); set(!st.on); });
  document.addEventListener("mousemove", e => {
    if (!st.on || st.pinned || card.contains(e.target)) return;
    const k = keyOf(e.target);
    if (k && (!st.cur || st.cur[0] !== k[0] || st.cur[1] !== k[1])) show(k[0], k[1], false);
  });
  const block = e => {
    if (!st.on || card.contains(e.target) || button.contains(e.target)) return;
    e.preventDefault(); e.stopPropagation();
    if (e.type !== "click") return;
    const k = keyOf(e.target);
    if (k && st.pinned !== k[1]){ st.pinned = k[1]; show(k[0], k[1], true); }
    else { st.pinned = null; if (k) show(k[0], k[1], false); else hide(); }
  };
  ["pointerdown", "mousedown", "click", "dblclick", "wheel"].forEach(t => document.addEventListener(t, block, {capture: true, passive: false}));
  document.addEventListener("keydown", e => {
    if (e.key !== "Escape" || !st.on) return;
    e.stopPropagation();
    if (st.pinned){ st.pinned = null; hide(); } else set(false);
  }, true);
  document.addEventListener("focusin", e => { if (st.on && !card.contains(e.target) && e.target !== button) e.target.blur(); }, true);
  const follow = () => { if (st.on && st.cur && document.contains(st.cur[1])) show(st.cur[0], st.cur[1], !!st.pinned); };
  window.addEventListener("resize", follow); document.addEventListener("scroll", follow, true);
  return {set, keyOf, get on(){ return st.on; }};
};
