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
        <strong>Project Two</strong>
        <span class="meta">Add stack here</span>
      </span>
      <span class="project-panel-icon" aria-hidden="true">+</span>
    </button>
    <div class="project-panel-body" hidden>
      <p>Add a short project summary here.</p>
    </div>
  </article>

  <article class="project-panel">
    <button class="project-panel-header" type="button" aria-expanded="false">
      <span>
        <strong>Project Three</strong>
        <span class="meta">Add stack here</span>
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
