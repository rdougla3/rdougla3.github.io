(function () {
  function setupAccordion() {
    document.querySelectorAll("[data-accordion]").forEach(function (accordion) {
      const panels = Array.from(accordion.querySelectorAll(".project-panel"));

      panels.forEach(function (panel) {
        const header = panel.querySelector(".project-panel-header");

        header.addEventListener("click", function () {
          const shouldOpen = !panel.classList.contains("is-open");

          panels.forEach(function (candidate) {
            const candidateHeader = candidate.querySelector(".project-panel-header");
            const candidateBody = candidate.querySelector(".project-panel-body");
            const isTarget = candidate === panel && shouldOpen;

            candidate.classList.toggle("is-open", isTarget);
            candidateHeader.setAttribute("aria-expanded", String(isTarget));
            candidateBody.hidden = !isTarget;
          });
        });
      });
    });
  }

  document.addEventListener("DOMContentLoaded", setupAccordion);
})();
