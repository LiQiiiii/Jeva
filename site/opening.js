(() => {
  "use strict";
  const opening = document.querySelector(".einstein-opening");
  if (!opening) return;
  const button = document.getElementById("opening-toggle");
  const base = opening.querySelector(".opening-base");
  const reveal = opening.querySelector(".opening-reveal");
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
  let full = false;
  let frame = 0;
  let pointer = null;

  function clearLens() {
    cancelAnimationFrame(frame);
    frame = 0;
    pointer = null;
    opening.style.setProperty("--lens-radius", "0px");
  }

  function drawLens() {
    frame = 0;
    if (!pointer || full) return;
    const bounds = opening.getBoundingClientRect();
    const x = pointer.x - bounds.left;
    const y = pointer.y - bounds.top;
    if (x < 0 || y < 0 || x > bounds.width || y > bounds.height) {
      clearLens();
      return;
    }
    opening.style.setProperty("--lens-x", `${x}px`);
    opening.style.setProperty("--lens-y", `${y}px`);
    opening.style.setProperty("--lens-radius", `${Math.min(225, bounds.width * .25)}px`);
  }

  function setFull(value) {
    clearLens();
    full = value;
    opening.classList.toggle("opening-full", full);
    base.setAttribute("aria-hidden", String(full));
    reveal.setAttribute("aria-hidden", String(!full));
    button.setAttribute("aria-pressed", String(full));
    button.setAttribute("aria-label", full ? "Show English quotation" : "Show Chinese translation");
    button.textContent = full ? "EN / 中文" : "中文 / EN";
  }

  opening.addEventListener("pointermove", event => {
    if (full || !finePointer.matches || event.pointerType === "touch" || event.target.closest(".opening-controls")) {
      clearLens();
      return;
    }
    pointer = {x: event.clientX, y: event.clientY};
    if (!frame) frame = requestAnimationFrame(drawLens);
  });
  opening.addEventListener("pointerleave", clearLens);
  opening.addEventListener("pointercancel", clearLens);
  opening.addEventListener("focusin", clearLens);
  button.addEventListener("click", () => setFull(!full));
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") setFull(false);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearLens();
  });
  window.addEventListener("scroll", clearLens, {passive: true});
  window.addEventListener("resize", clearLens);
  window.addEventListener("blur", clearLens);
  finePointer.addEventListener("change", clearLens);
  button.hidden = false;
  opening.classList.add("opening-interactive");
})();
