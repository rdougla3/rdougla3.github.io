(function () {
  const LEVELS = [
    { id: "level-1", label: "Level 1", nativeIterations: 1, nativeBytes: 256, minLength: 1, difficulty: 5, residue: 1, salt: 0x1f },
    { id: "level-2", label: "Level 2", nativeIterations: 1, nativeBytes: 256, minLength: 1, difficulty: 7, residue: 4, salt: 0x2d },
    { id: "level-3", label: "Level 3", nativeIterations: 13, nativeBytes: 256, minLength: 1, difficulty: 12, residue: 8, salt: 0x39 },
    { id: "level-4", label: "Level 4", nativeIterations: 2001, nativeBytes: 296, minLength: 31, difficulty: 11, residue: 3, salt: 0x4b },
    { id: "level-5", label: "Level 5", nativeIterations: 1501, nativeBytes: 286, minLength: 21, difficulty: 13, residue: 6, salt: 0x55 },
    { id: "level-6", label: "Level 6", nativeIterations: 1501, nativeBytes: 286, minLength: 21, difficulty: 14, residue: 9, salt: 0x63 },
    { id: "level-7", label: "Level 7", nativeIterations: 2, nativeBytes: 256, minLength: 1, difficulty: 8, residue: 5, salt: 0x71 },
    { id: "level-8", label: "Level 8", nativeIterations: 2, nativeBytes: 256, minLength: 1, difficulty: 9, residue: 7, salt: 0x84 },
    { id: "level-10", label: "Level 10", nativeIterations: 1503, nativeBytes: 286, minLength: 21, difficulty: 16, residue: 2, salt: 0x9a }
  ];

  const DEFAULT_MAX_ITERATIONS = 5000;
  const EPOCH_SIZE = 500;
  const MUTATION_RATE = 0.13;
  const CHUNK_SIZE = 45;

  function XorShift32(seed) {
    this.state = seed >>> 0 || 1;
  }

  XorShift32.prototype.next = function () {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  };

  XorShift32.prototype.float = function () {
    return this.next() / 4294967296;
  };

  XorShift32.prototype.byte = function () {
    return this.next() & 255;
  };

  function hashPayload(payload, salt) {
    let hash = (2166136261 ^ salt) >>> 0;

    payload.forEach(function (byte, index) {
      hash ^= (byte + index + salt) & 255;
      hash = Math.imul(hash, 16777619) >>> 0;
    });

    return hash >>> 0;
  }

  function crashFound(level, payload) {
    if (payload.length < level.minLength) return false;
    return hashPayload(payload, level.salt) % level.difficulty === level.residue;
  }

  function mutatePayload(payload, rng) {
    let mutations = 0;

    for (let i = 0; i < payload.length; i += 1) {
      if (rng.float() < MUTATION_RATE) {
        payload[i] = rng.byte();
        mutations += 1;
      }
    }

    return mutations;
  }

  function extendPayload(payload, rng) {
    for (let i = 0; i < 10; i += 1) {
      payload.push(rng.byte());
    }
  }

  function formatBytes(length) {
    return length === 1 ? "1 byte" : length + " bytes";
  }

  function byteToHex(byte) {
    return byte.toString(16).padStart(2, "0").toUpperCase();
  }

  function renderBytes(container, payload) {
    container.replaceChildren();

    payload.slice(0, 96).forEach(function (byte) {
      const cell = document.createElement("span");
      const hue = Math.round(145 + (byte / 255) * 70);
      const light = Math.round(34 + (byte / 255) * 18);
      cell.style.background = "linear-gradient(135deg, hsl(" + hue + " 58% " + (light + 8) + "%), hsl(" + (hue - 38) + " 62% " + light + "%))";
      cell.title = "0x" + byteToHex(byte);
      container.appendChild(cell);
    });
  }

  function renderHex(container, payload) {
    const rows = [];

    for (let rowStart = 0; rowStart < payload.length; rowStart += 16) {
      const row = payload.slice(rowStart, rowStart + 16);
      const address = rowStart.toString(16).padStart(4, "0");
      const hex = row.map(byteToHex).join(" ").padEnd(47, " ");
      const ascii = row.map(function (byte) {
        return byte >= 32 && byte <= 126 ? String.fromCharCode(byte) : ".";
      }).join("");

      rows.push(address + "  " + hex + "  " + ascii);
    }

    container.textContent = rows.join("\n") || "00";
  }

  function renderLevelBars(container) {
    const maxIterations = LEVELS.reduce(function (max, level) {
      return Math.max(max, level.nativeIterations);
    }, 1);

    container.replaceChildren();

    LEVELS.forEach(function (level) {
      const row = document.createElement("button");
      const label = document.createElement("span");
      const bar = document.createElement("span");
      const fill = document.createElement("span");
      const meta = document.createElement("span");

      row.type = "button";
      row.className = "fuzzer-level";
      row.dataset.levelId = level.id;
      label.textContent = level.label;
      bar.className = "fuzzer-level-bar";
      fill.style.width = Math.max(4, (level.nativeIterations / maxIterations) * 100) + "%";
      meta.textContent = level.nativeIterations + " iter";

      bar.appendChild(fill);
      row.append(label, bar, meta);
      container.appendChild(row);
    });
  }

  function setActiveLevel(container, levelId) {
    container.querySelectorAll(".fuzzer-level").forEach(function (row) {
      row.classList.toggle("is-active", row.dataset.levelId === levelId);
    });
  }

  function setupFuzzerDemo() {
    document.querySelectorAll("[data-fuzzer-demo]").forEach(function (demo) {
      const targetSelect = demo.querySelector("[data-fuzzer-target]");
      const seedInput = demo.querySelector("[data-fuzzer-seed]");
      const iterationsInput = demo.querySelector("[data-fuzzer-iterations]");
      const runButton = demo.querySelector("[data-run-fuzzer]");
      const status = demo.querySelector("[data-fuzzer-status]");
      const step = demo.querySelector("[data-fuzzer-step]");
      const size = demo.querySelector("[data-fuzzer-size]");
      const mutations = demo.querySelector("[data-fuzzer-mutations]");
      const progress = demo.querySelector("[data-fuzzer-progress]");
      const byteGrid = demo.querySelector("[data-byte-grid]");
      const hexDump = demo.querySelector("[data-hex-dump]");
      const levelBars = demo.querySelector("[data-fuzzer-levels]");
      let running = false;

      LEVELS.forEach(function (level) {
        const option = document.createElement("option");
        option.value = level.id;
        option.textContent = level.label;
        targetSelect.appendChild(option);
      });

      targetSelect.value = "level-10";
      renderLevelBars(levelBars);
      setActiveLevel(levelBars, targetSelect.value);
      renderBytes(byteGrid, [0]);
      renderHex(hexDump, [0]);

      function currentLevel() {
        return LEVELS.find(function (level) {
          return level.id === targetSelect.value;
        }) || LEVELS[0];
      }

      function paint(state) {
        step.textContent = state.iteration.toLocaleString();
        size.textContent = formatBytes(state.payload.length);
        mutations.textContent = state.mutations.toLocaleString();
        progress.style.width = Math.min(100, (state.iteration / state.maxIterations) * 100) + "%";
        renderBytes(byteGrid, state.payload);
        renderHex(hexDump, state.payload);
      }

      function finish(message) {
        running = false;
        runButton.disabled = false;
        targetSelect.disabled = false;
        seedInput.disabled = false;
        iterationsInput.disabled = false;
        status.textContent = message;
      }

      function run() {
        if (running) return;

        const level = currentLevel();
        const seed = Math.max(1, parseInt(seedInput.value, 10) || 1337);
        const maxIterations = Math.max(100, Math.min(20000, parseInt(iterationsInput.value, 10) || DEFAULT_MAX_ITERATIONS));
        const rng = new XorShift32(seed);
        const state = {
          iteration: 0,
          maxIterations: maxIterations,
          mutations: 0,
          payload: [0]
        };

        running = true;
        runButton.disabled = true;
        targetSelect.disabled = true;
        seedInput.disabled = true;
        iterationsInput.disabled = true;
        status.textContent = "Fuzzing";
        progress.style.width = "0%";

        function tick() {
          let steps = 0;

          while (steps < CHUNK_SIZE && state.iteration < maxIterations) {
            if (state.iteration > 0 && state.iteration % EPOCH_SIZE === 0) {
              extendPayload(state.payload, rng);
            }

            state.mutations += mutatePayload(state.payload, rng);
            state.iteration += 1;

            if (crashFound(level, state.payload)) {
              paint(state);
              finish("Crash found");
              return;
            }

            steps += 1;
          }

          paint(state);

          if (state.iteration >= maxIterations) {
            finish("No crash");
            return;
          }

          window.requestAnimationFrame(tick);
        }

        tick();
      }

      targetSelect.addEventListener("change", function () {
        setActiveLevel(levelBars, targetSelect.value);
      });

      levelBars.addEventListener("click", function (event) {
        const row = event.target.closest(".fuzzer-level");
        if (!row || running) return;
        targetSelect.value = row.dataset.levelId;
        setActiveLevel(levelBars, targetSelect.value);
      });

      runButton.addEventListener("click", run);
    });
  }

  document.addEventListener("DOMContentLoaded", setupFuzzerDemo);
})();
