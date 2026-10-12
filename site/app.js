"use strict";
const config = window.JEVA_SITE || {};
document.querySelectorAll("[data-link]").forEach(link => {
  const target = config[link.dataset.link];
  if (target && /^https:\/\//.test(target)) {
    link.href = target;
    link.hidden = false;
    link.rel = "noopener noreferrer";
  }
});
document.getElementById("model-access").textContent = config.modelAccess || "";

const phaseText = [
  "Encode each option independently. Store its semantic vector and the projected keys and values used by the decision block.",
  "Encode the current state together with the question and decision type. The option encoder never sees this request.",
  "The state reads the complete option memory. A candidate-weighted message and a residual FFN refine the state; options stay fixed.",
  "Score options with the refined state. Coverage separates answer availability from relative preference; typed readout returns Choice, Noul, or Score."
];
let phase = 0;
let playTimer = null;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
function setPhase(value) {
  phase = value;
  document.querySelectorAll(".phase-button").forEach(button => {
    const active = Number(button.dataset.phase) === phase;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  document.querySelectorAll(".flow-node").forEach(node => node.classList.toggle("active", Number(node.dataset.stage) === phase));
  document.querySelectorAll(".wire").forEach(wire => {
    const active = phase === 0 && wire.classList.contains("option-wire") || phase === 1 && wire.classList.contains("state-wire") || phase === 2 && wire.classList.contains("residual-wire") || phase === 3 && wire.classList.contains("readout-wire");
    wire.classList.toggle("active", active);
  });
  document.querySelector(".typed-output").classList.toggle("active", phase === 3);
  document.getElementById("phase-count").textContent = `0${phase + 1} / 04`;
  document.getElementById("phase-text").textContent = phaseText[phase];
}
function stopPlayback() {
  clearInterval(playTimer);
  playTimer = null;
  const button = document.getElementById("flow-play");
  button.innerHTML = 'Play <span aria-hidden="true">▶</span>';
  button.setAttribute("aria-pressed", "false");
}
document.querySelectorAll(".phase-button").forEach(button => button.addEventListener("click", () => {stopPlayback(); setPhase(Number(button.dataset.phase));}));
document.getElementById("flow-play").addEventListener("click", () => {
  if (playTimer) {stopPlayback(); return;}
  if (phase === 3) setPhase(0);
  const button = document.getElementById("flow-play");
  button.innerHTML = 'Pause <span aria-hidden="true">Ⅱ</span>';
  button.setAttribute("aria-pressed", "true");
  playTimer = setInterval(() => {
    if (phase === 3) {stopPlayback(); return;}
    setPhase(phase + 1);
  }, reducedMotion.matches ? 2600 : 2100);
});
document.addEventListener("visibilitychange", () => {if (document.hidden) stopPlayback();});
setPhase(0);

const optionLabels = {A: "Billing", B: "Technical", C: "Accounts", D: "Security"};
const scenarios = {
  cold: {request: "“My payment was taken twice.”", options: ["A", "B", "C"], cached: [], text: "No candidate representation is cached yet. All modes encode the three options."},
  same: {request: "“I cannot log in after resetting my password.”", options: ["A", "B", "C"], cached: ["A", "B", "C"], text: "The state changes, but the exact candidate tuple is already cached. Set and Hybrid both reuse it."},
  reorder: {request: "“The app crashes when I open my invoice.”", options: ["C", "A", "B"], cached: ["A", "B", "C"], text: "A new ordering misses the exact-tuple cache. Hybrid gathers the same three option memories in the new order."},
  recombine: {request: "“Someone else appears to have used my card.”", options: ["A", "D", "C"], cached: ["A", "B", "C"], text: "The set is new. Hybrid reuses A and C, and encodes only the new Security option D."}
};
function textElement(tag, text, className) {
  const el = document.createElement(tag);
  el.textContent = text;
  if (className) el.className = className;
  return el;
}
function renderScenario(name) {
  const scenario = scenarios[name];
  document.querySelectorAll("[data-scenario]").forEach(button => {
    const active = button.dataset.scenario === name;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  document.getElementById("request-example").textContent = scenario.request;
  document.getElementById("cache-explanation").textContent = scenario.text;
  const candidates = document.getElementById("candidate-tokens");
  candidates.replaceChildren();
  scenario.options.forEach(key => {
    const hit = scenario.cached.includes(key);
    const token = textElement("span", "", `candidate-token ${hit ? "cached" : "fresh"}`);
    token.append(textElement("b", key), document.createTextNode(optionLabels[key]));
    candidates.append(token);
  });
  const memory = document.getElementById("memory-bank");
  memory.replaceChildren();
  Object.keys(optionLabels).forEach(key => {
    const hit = scenario.cached.includes(key);
    const used = scenario.options.includes(key);
    const entry = textElement("div", "", `memory-entry ${hit ? "cached" : "fresh"} ${used ? "" : "not-used"}`);
    entry.append(textElement("b", key), textElement("span", optionLabels[key]), textElement("span", hit ? "cached" : used ? "encode" : "unused", "entry-status"));
    memory.append(entry);
  });
  const hits = scenario.options.filter(key => scenario.cached.includes(key)).length;
  const setHit = scenario.cached.length > 0 && JSON.stringify(scenario.options) === JSON.stringify(["A", "B", "C"]);
  document.getElementById("off-count").textContent = "3";
  document.getElementById("set-count").textContent = setHit ? "0" : "3";
  document.getElementById("hybrid-count").textContent = String(3 - hits);
  document.getElementById("set-hit").textContent = setHit ? "Yes" : "No";
  document.getElementById("option-reuse").textContent = `${hits} / 3`;
}
document.querySelectorAll("[data-scenario]").forEach(button => button.addEventListener("click", () => renderScenario(button.dataset.scenario)));
renderScenario("cold");

let toastTimer;
function notify(text) {
  const toast = document.querySelector(".toast");
  clearTimeout(toastTimer);
  toast.textContent = text;
  toast.hidden = false;
  toastTimer = setTimeout(() => {toast.hidden = true;}, 2200);
}
document.querySelectorAll("[data-copy]").forEach(button => button.addEventListener("click", async () => {
  const text = document.getElementById(button.dataset.copy).innerText;
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.append(area);
      area.select();
      const copied = document.execCommand("copy");
      area.remove();
      if (!copied) throw new Error("Clipboard unavailable");
    }
    notify("Copied to clipboard");
  } catch (_) {notify("Select the code and copy it manually.");}
}));
