(function () {
  function normalRandom() {
    let u = 0;
    let v = 0;

    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();

    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function leakyRelu(value) {
    return value >= 0 ? value : value * 0.2;
  }

  function linear(input, weight, bias) {
    return weight.map(function (row, rowIndex) {
      let sum = bias[rowIndex];

      for (let i = 0; i < input.length; i += 1) {
        sum += row[i] * input[i];
      }

      return sum;
    });
  }

  function addInternalNoise(values, scale) {
    return values.map(function (value, index) {
      return value + normalRandom() * scale[index];
    });
  }

  function runGeneratorStages(weights) {
    const layers = weights.layers;
    let x = layers.constant[0].slice();
    const stages = [];

    x = addInternalNoise(linear(x, layers["layer1.weight"], layers["layer1.bias"]).map(leakyRelu), layers.noise_scale1);
    stages.push({ label: "Layer 1", dimensions: "256", pixels: x.slice(), isFinal: false });

    x = addInternalNoise(linear(x, layers["layer2.weight"], layers["layer2.bias"]).map(leakyRelu), layers.noise_scale2);
    stages.push({ label: "Layer 2", dimensions: "512", pixels: x.slice(), isFinal: false });

    x = addInternalNoise(linear(x, layers["layer3.weight"], layers["layer3.bias"]).map(leakyRelu), layers.noise_scale3);
    stages.push({ label: "Layer 3", dimensions: "1024", pixels: x.slice(), isFinal: false });

    x = linear(x, layers["layer4.weight"], layers["layer4.bias"]).map(Math.tanh);
    stages.push({ label: "Final output", dimensions: "1 x 28 x 28", pixels: x.slice(), isFinal: true });

    return stages;
  }

  function normalizeIntermediate(values) {
    let min = Infinity;
    let max = -Infinity;

    values.forEach(function (value) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    });

    if (max === min) {
      return values.map(function () {
        return 0;
      });
    }

    return values.map(function (value) {
      return ((value - min) / (max - min)) * 2 - 1;
    });
  }

  function previewVector(values, isFinal) {
    if (isFinal) {
      return values;
    }

    const normalized = normalizeIntermediate(values);
    const sourceWidth = Math.ceil(Math.sqrt(normalized.length));
    const sourceHeight = Math.ceil(normalized.length / sourceWidth);
    const preview = [];

    for (let y = 0; y < 28; y += 1) {
      const sourceY = Math.min(sourceHeight - 1, Math.floor((y / 28) * sourceHeight));

      for (let x = 0; x < 28; x += 1) {
        const sourceX = Math.min(sourceWidth - 1, Math.floor((x / 28) * sourceWidth));
        const sourceIndex = sourceY * sourceWidth + sourceX;
        preview.push(normalized[sourceIndex] || -1);
      }
    }

    return preview;
  }

  function drawMnist(canvas, pixels, isFinal) {
    const context = canvas.getContext("2d");
    const image = context.createImageData(28, 28);
    const previewPixels = previewVector(pixels, isFinal);

    previewPixels.forEach(function (value, index) {
      const normalized = Math.max(0, Math.min(255, Math.round(((value + 1) / 2) * 255)));
      const offset = index * 4;

      image.data[offset] = normalized;
      image.data[offset + 1] = normalized;
      image.data[offset + 2] = normalized;
      image.data[offset + 3] = 255;
    });

    context.putImageData(image, 0, 0);
  }

  function sleep(milliseconds) {
    return new Promise(function (resolve) {
      window.setTimeout(resolve, milliseconds);
    });
  }

  function setupAccordion() {
    document.querySelectorAll("[data-accordion]").forEach(function (accordion) {
      const panels = Array.from(accordion.querySelectorAll(".project-panel"));

      panels.forEach(function (panel) {
        const header = panel.querySelector(".project-panel-header");
        const body = panel.querySelector(".project-panel-body");

        header.addEventListener("click", function () {
          panels.forEach(function (candidate) {
            const candidateHeader = candidate.querySelector(".project-panel-header");
            const candidateBody = candidate.querySelector(".project-panel-body");
            const isTarget = candidate === panel;

            candidate.classList.toggle("is-open", isTarget);
            candidateHeader.setAttribute("aria-expanded", String(isTarget));
            candidateBody.hidden = !isTarget;
          });
        });
      });
    });
  }

  function setupGanDemo() {
    document.querySelectorAll("[data-gan-demo]").forEach(function (demo) {
      const canvas = demo.querySelector(".gan-canvas");
      const button = demo.querySelector("[data-generate-mnist]");
      const status = demo.querySelector("[data-gan-status]");
      const caption = demo.querySelector("[data-gan-caption]");
      const weightsUrl = demo.dataset.weightsUrl;
      let weights = null;
      let isAnimating = false;

      button.disabled = true;

      fetch(weightsUrl)
        .then(function (response) {
          if (!response.ok) {
            throw new Error("Weights file not found.");
          }

          return response.json();
        })
        .then(function (loadedWeights) {
          weights = loadedWeights;
          button.disabled = false;
          status.textContent = "Weights loaded. Ready to generate.";
        })
        .catch(function () {
          status.textContent = "Run the GAN notebook export cell to add the weights file.";
        });

      button.addEventListener("click", function () {
        if (!weights || isAnimating) return;

        const stages = runGeneratorStages(weights);
        isAnimating = true;
        button.disabled = true;
        status.textContent = "Generating through network layers.";

        stages.reduce(function (chain, stage) {
          return chain.then(function () {
            drawMnist(canvas, stage.pixels, stage.isFinal);
            caption.textContent = stage.label + " - dimensions: " + stage.dimensions;
            return sleep(250);
          });
        }, Promise.resolve()).then(function () {
          isAnimating = false;
          button.disabled = false;
          status.textContent = "Generated a fake MNIST image.";
        });
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    setupAccordion();
    setupGanDemo();
  });
})();
