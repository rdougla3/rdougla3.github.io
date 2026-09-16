(function () {
  document.querySelectorAll('[data-photo-album]').forEach(function (album) {
    const track = album.querySelector('[data-photo-track]');
    const slides = Array.from(track.children);
    if (!slides.length) return;

    // Shuffle once per page load; browsing keeps this order until the next load.
    for (let i = slides.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [slides[i], slides[j]] = [slides[j], slides[i]];
    }
    slides.forEach(function (slide, index) {
      slide.setAttribute('aria-label', (index + 1) + ' of ' + slides.length);
      slide.querySelector('img').loading = index === 0 ? 'eager' : 'lazy';
    });
    track.replaceChildren(...slides);
    track.scrollTo({ left: 0, behavior: 'instant' });

    const position = album.querySelector('[data-photo-position]');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let current = 0;
    let frame;

    function show(index) {
      current = (index + slides.length) % slides.length;
      track.scrollTo({
        left: current * track.clientWidth,
        behavior: reducedMotion.matches ? 'instant' : 'smooth'
      });
    }

    album.querySelector('[data-photo-previous]').addEventListener('click', function () {
      show(current - 1);
    });
    album.querySelector('[data-photo-next]').addEventListener('click', function () {
      show(current + 1);
    });

    track.addEventListener('keydown', function (event) {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const destinations = {
        ArrowLeft: current - 1,
        ArrowRight: current + 1,
        Home: 0,
        End: slides.length - 1
      };
      if (!(event.key in destinations)) return;
      event.preventDefault();
      show(destinations[event.key]);
    });

    // Native scrolling provides touch/swipe support, including without JavaScript.
    track.addEventListener('scroll', function () {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(function () {
        current = Math.round(track.scrollLeft / track.clientWidth);
        position.textContent = (current + 1) + ' / ' + slides.length;
      });
    }, { passive: true });

    album.querySelector('[data-photo-controls]').hidden = slides.length < 2;
  });
})();
