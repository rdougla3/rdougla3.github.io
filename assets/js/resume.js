const printButton = document.querySelector('[data-print-resume]');

if (printButton) {
  printButton.hidden = false;
  printButton.addEventListener('click', () => window.print());
}
