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
        <strong>Project Three</strong>
        <span class="meta">Coming soon 😎</span>
      </span>
      <span class="project-panel-icon" aria-hidden="true">+</span>
    </button>
    <div class="project-panel-body" hidden>
      <p>Add a short project summary here.</p>
    </div>
  </article>
</section>

<script src="{{ '/assets/js/accordion.js' | relative_url }}"></script>
<script src="{{ '/assets/js/gan-demo.js' | relative_url }}"></script>
<script src="{{ '/assets/js/fuzzer-demo.js' | relative_url }}"></script>
