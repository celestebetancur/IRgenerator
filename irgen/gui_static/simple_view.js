/* Simple / full view, shared by the standard and strange views.

   IrgenSimpleView({toggle, panel, keys}):
     toggle : container with <button data-v="full"> and <button data-v="simple">
     panel  : the parameter side panel
     keys   : parameter paths / data-help keys shown in the simple view
   Simple hides every .row whose control isn't in keys, every [data-help] widget not in
   keys (octave sliders, surfaces, EQ), the hints, and the sections left empty; the sections
   that stay are opened. Hidden parameters keep their values. Full restores everything,
   including which sections were open. Starts in full view. */
"use strict";
window.IrgenSimpleView = function ({toggle, panel, keys}) {
  document.head.append(Object.assign(document.createElement("style"), {textContent: ".isv-hide{display:none!important}"}));
  const want = new Set(keys);
  let simple = false, opened = null;
  const keyOf = e => e.dataset.path || e.dataset.help;

  function apply(){
    panel.querySelectorAll(".isv-hide").forEach(e => e.classList.remove("isv-hide"));
    toggle.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === (simple ? "simple" : "full")));
    const secs = [...panel.querySelectorAll("details")];
    if (!simple){
      if (opened) secs.forEach((d, i) => { d.open = opened[i]; });
      opened = null;
      return;
    }
    if (!opened) opened = secs.map(d => d.open);
    panel.querySelectorAll(".row").forEach(r => {
      const c = r.querySelector("[data-path]");
      if (!c || !want.has(c.dataset.path)) r.classList.add("isv-hide");
    });
    panel.querySelectorAll(".sec [data-help]").forEach(w => { if (!w.closest(".row") && !want.has(keyOf(w))) w.classList.add("isv-hide"); });
    panel.querySelectorAll("div.hint").forEach(h => h.classList.add("isv-hide"));   // explanatory text, not labels / units
    panel.querySelectorAll(".sub").forEach(s => { if (!has(s)) s.classList.add("isv-hide"); });
    secs.forEach(d => { if (has(d)) d.open = true; else d.classList.add("isv-hide"); });
  }
  function has(root){
    return [...root.querySelectorAll("[data-path], .sec [data-help]")].some(e => want.has(keyOf(e)));
  }
  toggle.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { simple = b.dataset.v === "simple"; apply(); }));
  apply();
  return {apply, get simple(){ return simple; }};
};
