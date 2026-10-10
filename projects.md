---
layout: default
title: Projects
permalink: /projects/
---

# Projects

<section class="project-accordion" data-accordion aria-label="Project list">
  <article class="project-panel is-open">
    <button class="project-panel-header" type="button" aria-expanded="true">
      <span>
        <strong>GAN MNIST Generator</strong>
        <span class="meta">PyTorch, JavaScript, neural networks</span>
      </span>
      <span class="project-panel-icon" aria-hidden="true">+</span>
    </button>
    <div class="project-panel-body">
      <p>This project trains a generative adversarial network on <a href="{{ 'https://en.wikipedia.org/wiki/MNIST_database'}}">MNIST</a>, 
      then exports the trained generator weights so the portfolio can reconstruct the model in the browser.</p>
      <p>The browser demo runs a fully connected generator that maps a 128-dimensional noise vector through linear layers with batch normalization and LeakyReLU activations before producing a 784-value (28x28) MNIST image. The model uses a final Tanh output and contains 2,449,680 <a href="{{ '/assets/models/gan-generator-weights.json' | relative_url }}">trainable parameters</a>.
      </p>

      <div class="gan-demo" data-gan-demo data-weights-url="{{ '/assets/models/gan-generator-weights.json' | relative_url }}">
        <div class="gan-preview">
          <canvas class="gan-canvas" width="28" height="28" aria-label="Generated fake MNIST digit"></canvas>
          <p class="gan-caption" data-gan-caption>Awaiting generation.</p>
        </div>
        <div class="gan-controls">
          <button class="button" type="button" data-generate-mnist>Generate Fake MNIST Image</button>
          <p class="meta" data-gan-status>Weights not loaded yet.</p>
        </div>
      </div>
    </div>
  </article>

  <article class="project-panel">
    <button class="project-panel-header" type="button" aria-expanded="false">
      <span>
        <strong>Mutation-Based Binary Fuzzer</strong>
        <span class="meta">Python, JavaScript, systems testing, crash reproduction</span>
      </span>
      <span class="project-panel-icon" aria-hidden="true">+</span>
    </button>
    <div class="project-panel-body" hidden>
      <p>This project mutates binary inputs for compiled test programs until a segmentation fault is found. The original command-line fuzzer uses a deterministic PRNG seed, extends the payload every 500 iterations, mutates each byte with a 13% probability, and writes both the crashing input and the seed/iteration pair needed to reproduce it.</p>
      <p>The browser version below ports the core mutation loop to JavaScript and runs against lightweight target predicates, while the comparison table shows the native crash artifacts produced by the original fuzzer.</p>

      <div class="fuzzer-demo" data-fuzzer-demo>
        <div class="fuzzer-toolbar">
          <label>
            Target
            <select data-fuzzer-target></select>
          </label>
          <label>
            Seed
            <input type="number" min="1" step="1" value="1337" data-fuzzer-seed>
          </label>
          <label>
            Iterations
            <input type="number" min="100" max="20000" step="100" value="5000" data-fuzzer-iterations>
          </label>
          <button class="button" type="button" data-run-fuzzer>Run Fuzzer</button>
        </div>

        <div class="fuzzer-layout">
          <div class="fuzzer-output">
            <div class="fuzzer-stats" aria-label="Fuzzer run statistics">
              <span>
                <strong data-fuzzer-status>Ready</strong>
                <small>Status</small>
              </span>
            </div>
            <div class="fuzzer-progress" aria-hidden="true">
              <span data-fuzzer-progress></span>
            </div>
            <div class="byte-grid" data-byte-grid aria-label="Current payload bytes"></div>
            <pre class="hex-dump" data-hex-dump>00</pre>
          </div>

          <div class="fuzzer-results">
            <div class="fuzzer-side-stats" aria-label="Current fuzzer metrics">
              <span>
                <strong data-fuzzer-step>0</strong>
                <small>Iteration</small>
              </span>
              <span>
                <strong data-fuzzer-size>1 byte</strong>
                <small>Payload</small>
              </span>
              <span>
                <strong data-fuzzer-mutations>0</strong>
                <small>Mutations</small>
              </span>
            </div>
            <h3>Native Crash Finds</h3>
            <div class="fuzzer-levels" data-fuzzer-levels></div>
          </div>
        </div>
      </div>
    </div>
  </article>

  <article class="project-panel">
    <button class="project-panel-header" type="button" aria-expanded="false">
      <span>
        <strong>Image Steganography</strong>
        <span class="meta">C#, JavaScript, image processing, binary encoding</span>
      </span>
      <span class="project-panel-icon" aria-hidden="true">+</span>
    </button>
    <div class="project-panel-body" hidden>
      <p>This browser demo hides and recovers text messages in uploaded images by storing data in the two least-significant bits of each red, green, and blue channel of fully opaque pixels. Start and end markers, a message length, and a checksum identify the hidden payload, determine how many pixels it occupies, and detect damaged messages.</p>
      <p>Embed a message and download the resulting PNG, or upload a marked image to reveal its message and see how many pixels store it. Images are processed locally in your browser.</p>

      <div class="steg-demo" data-steg-demo>
        <label class="steg-upload" for="steg-upload">
          <strong>Choose an image</strong>
          <input id="steg-upload" data-steg-upload type="file" accept="image/png,image/jpeg,image/webp,image/bmp,.bmp" aria-describedby="steg-upload-help">
          <span id="steg-upload-help">PNG, JPEG, WebP, or BMP · up to 20 MB and 16 million pixels</span>
        </label>

        <div class="steg-layout">
          <div class="steg-preview">
            <div class="steg-image">
              <p data-steg-placeholder>Your image will appear here.</p>
              <canvas data-steg-canvas hidden aria-label="Image preview, showing the embedded result after encoding"></canvas>
            </div>
            <dl class="steg-stats">
              <div><dt>Image dimensions</dt><dd data-steg-dimensions>—</dd></div>
              <div><dt>Message capacity</dt><dd data-steg-capacity>—</dd></div>
              <div><dt>Pixels storing data</dt><dd data-steg-used>—</dd></div>
            </dl>
          </div>

          <div class="steg-actions">
            <fieldset>
              <legend>Hide a message</legend>
              <label for="steg-message">Message to embed</label>
              <textarea id="steg-message" data-steg-message rows="4" placeholder="Write something to hide in the image…" aria-describedby="steg-message-size"></textarea>
              <p id="steg-message-size" class="steg-message-size" data-steg-message-size>Choose an image to see capacity.</p>
              <div class="steg-buttons">
                <button class="button" type="button" data-steg-embed disabled>Embed message</button>
                <a class="button" data-steg-download hidden download>Download PNG</a>
              </div>
            </fieldset>

            <fieldset>
              <legend>Reveal a message</legend>
              <p>Read a message written by this demo, using only the marked image.</p>
              <button class="button" type="button" data-steg-reveal disabled>Reveal message</button>
              <div class="steg-reveal-result" data-steg-reveal-result hidden>
                <label for="steg-revealed">Revealed message</label>
                <textarea id="steg-revealed" data-steg-revealed rows="4" readonly spellcheck="false"></textarea>
              </div>
            </fieldset>
          </div>
        </div>

        <p class="steg-status" data-steg-status role="status" aria-live="polite" aria-atomic="true">Choose an image to get started.</p>
        <p class="steg-note">Keep the downloaded PNG unchanged: resizing, color edits, or JPEG compression can destroy the message. Anyone with this decoder can read it.</p>
        <details class="steg-format">
          <summary>How the watermark works</summary>
          <p>The message is stored as UTF-8 text between start and end markers. A length field identifies the payload boundary, and a checksum detects damaged messages. The pixel count includes this framing as well as the text.</p>
          <p>Each fully opaque pixel stores six bits in the two lowest bits of its red, green, and blue channels. Transparent and partially transparent pixels are skipped, and image dimensions stay unchanged. Detection recognizes this demo’s format.</p>
        </details>
        <noscript><p>Enable JavaScript to embed or reveal messages in your browser.</p></noscript>
      </div>
    </div>
  </article>
</section>

<script src="{{ '/assets/js/accordion.js' | relative_url }}"></script>
<script src="{{ '/assets/js/gan-demo.js' | relative_url }}"></script>
<script src="{{ '/assets/js/fuzzer-demo.js' | relative_url }}"></script>
<script src="{{ '/assets/js/steganography-codec.js' | relative_url }}" defer></script>
<script src="{{ '/assets/js/steganography-demo.js' | relative_url }}" defer></script>
