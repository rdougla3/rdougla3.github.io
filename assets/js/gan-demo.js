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

  function batchNorm(input, weight, bias, runningMean, runningVar) {
    const epsilon = 1e-5;

    return input.map(function (value, index) {
      const normalized = (value - runningMean[index]) / Math.sqrt(runningVar[index] + epsilon);
      return normalized * weight[index] + bias[index];
    });
  }

  function runGeneratorStages(weights) {
    const layers = weights.layers;
    let x = Array.from({ length: weights.latent_dim }, normalRandom);
    const stages = [];

    if (!layers["net.0.weight"]) {
      throw new Error("Unsupported generator weights architecture.");
    }

    x = linear(x, layers["net.0.weight"], layers["net.0.bias"]);
    x = batchNorm(x, layers["net.1.weight"], layers["net.1.bias"], layers["net.1.running_mean"], layers["net.1.running_var"]).map(leakyRelu);
    stages.push({ label: "Layer 1", dimensions: "512", pixels: x.slice(), isFinal: false });

    x = linear(x, layers["net.3.weight"], layers["net.3.bias"]);
    x = batchNorm(x, layers["net.4.weight"], layers["net.4.bias"], layers["net.4.running_mean"], layers["net.4.running_var"]).map(leakyRelu);
    stages.push({ label: "Layer 2", dimensions: "1024", pixels: x.slice(), isFinal: false });

    x = linear(x, layers["net.6.weight"], layers["net.6.bias"]);
    x = batchNorm(x, layers["net.7.weight"], layers["net.7.bias"], layers["net.7.running_mean"], layers["net.7.running_var"]).map(leakyRelu);
    stages.push({ label: "Layer 3", dimensions: "1024", pixels: x.slice(), isFinal: false });

    x = linear(x, layers["net.9.weight"], layers["net.9.bias"]).map(Math.tanh);
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

        try {
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
        } catch (error) {
          isAnimating = false;
          button.disabled = false;
          status.textContent = "The loaded weights do not match this demo version.";
        }
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    setupGanDemo();
  });
})();
