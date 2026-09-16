# Ross Douglas Portfolio

GitHub Pages portfolio scaffold built with Jekyll and the `jekyll-theme-midnight` dark theme.

## Pages

- Home: `index.md`
- About: `about.md`
- Projects: `projects.md`
- Experience: `experience.html`, a printable resume with content in `_data/resume.yml`
- The former `/resume/` URL redirects to `/experience/`.

To preview locally, install the GitHub Pages gem bundle and run:

```bash
bundle install
bundle exec jekyll serve
```

## About photo album

Add an image directly to `photos/` or `assets/images/album/`; the About carousel
automatically includes it on the next Jekyll build (or local preview rebuild).
JPG/JPEG, PNG, WebP, GIF, and AVIF are supported, including uppercase extensions.
No data-file entry or separate preview is required. Refresh the page after rebuilding;
the published site picks up additions when deployed.

The carousel randomly shuffles all photos on each page load, including newly
discovered images. The order stays unchanged while browsing that page.
Optionally add an entry to `_data/photos.yml` to set descriptive alt text or pair
an original (`src`) with a smaller `preview` and its `width`/`height`.
Listed originals and previews are shown once; missing files are skipped, with
the original used if its preview is unavailable. Unlisted images use their filenames
for alt text. Prefer descriptive filenames and web-sized images for new additions.

Browse with the buttons, swipe/scroll, or focus the photos and use the arrow keys
(Home/End jump to the first/last photo). Photos open their originals in a new tab.
Without JavaScript, the album remains horizontally scrollable in data-file order,
followed by unlisted images sorted by path.

## Editing and exporting the resume

Edit `_data/resume.yml` to update the Experience page's resume content. Email and
phone are optional and hidden when blank. The `additional_experience` entries render
as compact blurbs. Dates and descriptions can be left blank when unavailable.
The About and Projects pages are maintained separately from the resume.
Change `assets/css/resume.css` to adjust its typography and layout.

Open `/experience/` and select **Save as PDF** (or use the browser's Print command).
Choose **Save as PDF**, **Letter** paper, **100%** scale, and **default** margins;
disable browser headers and footers. The stylesheet sets an 8.5 × 11 inch page with
0.6 inch side margins and 0.55 inch top/bottom margins. Navigation and export controls
are excluded from the PDF, and text and links remain selectable.

The desktop view uses the same content width and typography as the PDF. On smaller
screens, it reflows for reading without changing the print layout. Content is not
clipped to a fixed height: longer resumes flow onto additional printed pages with
repeated margins. Check print preview after edits to confirm the page count.
