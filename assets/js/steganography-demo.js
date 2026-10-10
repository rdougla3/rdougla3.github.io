(function () {
  'use strict';
  const codec = window.SteganographyCodec;
  const demo = document.querySelector('[data-steg-demo]');
  if (!demo || !codec) return;
  const find = (name) => demo.querySelector('[data-steg-' + name + ']');
  const upload = find('upload');
  const canvas = find('canvas');
  const context = canvas.getContext('2d', { willReadFrequently: true });
  const message = find('message');
  const embedButton = find('embed');
  const revealButton = find('reveal');
  const download = find('download');
  const status = find('status');
  const revealed = find('revealed');
  const MAX_FILE_BYTES = 20 * 1024 * 1024;
  const MAX_PIXELS = 16000000;
  let pixels = null;
  let capacity = null;
  let outputURL = null;
  let revision = 0;
  let busy = false;
  let filename = 'hidden-message';

  function say(text, error = false) {
    status.textContent = text;
    status.classList.toggle('is-error', error);
  }

  function clearDownload() {
    if (outputURL) URL.revokeObjectURL(outputURL);
    outputURL = null;
    download.hidden = true;
    download.removeAttribute('href');
  }

  function clearRevealed() {
    revealed.value = '';
    find('reveal-result').hidden = true;
  }

  function updateControls() {
    const bytes = codec.byteLength(message.value);
    embedButton.disabled = busy || !pixels || !capacity.canStoreFrame || !bytes || bytes > capacity.capacity;
    revealButton.disabled = busy || !pixels;
    message.disabled = busy;
    find('message-size').textContent = capacity
      ? bytes.toLocaleString() + ' UTF-8 bytes · ' + codec.pixelsForBytes(bytes).toLocaleString() +
        ' storage pixels needed / ' + capacity.usablePixels.toLocaleString() + ' available'
      : bytes.toLocaleString() + ' UTF-8 bytes · choose an image to see capacity';
    find('message-size').classList.toggle('is-error', !!capacity && (!capacity.canStoreFrame || bytes > capacity.capacity));
    message.setAttribute('aria-invalid', String(!!capacity && (!capacity.canStoreFrame || bytes > capacity.capacity)));
  }

  function showDetection(result) {
    find('used').textContent = result.status === 'found' ? result.usedPixels.toLocaleString() : '—';
    if (result.status === 'found') {
      say('Watermark detected: ' + result.messageBytes.toLocaleString() + ' message bytes in ' +
        result.usedPixels.toLocaleString() + ' pixels, including markers. Select Reveal message to read it.');
    } else if (result.status === 'absent') {
      say(capacity.canStoreFrame
        ? 'No compatible watermark found. You can embed a new message in this image.'
        : 'This image has too few fully opaque pixels to store a watermark. Choose a larger or opaque image.');
    } else {
      say(result.reason, true);
    }
  }

  upload.addEventListener('change', async function () {
    const request = ++revision;
    const file = upload.files[0];
    clearDownload();
    clearRevealed();
    pixels = null;
    capacity = null;
    canvas.hidden = true;
    find('placeholder').hidden = false;
    find('dimensions').textContent = '—';
    find('capacity').textContent = '—';
    find('used').textContent = '—';
    busy = false;
    updateControls();
    if (!file) { say('Choose an image to get started.'); return; }
    if (file.size > MAX_FILE_BYTES) { say('Choose an image smaller than 20 MB.', true); return; }
    if (!/\.(png|jpe?g|webp|bmp)$/i.test(file.name) && !/^image\/(png|jpeg|webp|bmp|x-ms-bmp)$/.test(file.type)) {
      say('Choose a PNG, JPEG, WebP, or BMP image.', true);
      return;
    }

    busy = true;
    updateControls();
    say('Reading image…');
    const url = URL.createObjectURL(file);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      if (request !== revision) return;
      if (image.naturalWidth * image.naturalHeight > MAX_PIXELS) {
        throw new Error('Choose an image with at most 16 million pixels. Images are not resized because that would change hidden data.');
      }
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      context.drawImage(image, 0, 0);
      pixels = context.getImageData(0, 0, canvas.width, canvas.height);
      capacity = codec.measure(pixels.data);
      filename = file.name.replace(/\.[^.]+$/, '') || 'image';
      canvas.hidden = false;
      find('placeholder').hidden = true;
      find('dimensions').textContent = canvas.width.toLocaleString() + ' × ' + canvas.height.toLocaleString();
      find('capacity').textContent = capacity.capacity.toLocaleString() + ' bytes';
      showDetection(codec.inspect(pixels.data));
    } catch (error) {
      if (request !== revision) return;
      pixels = null;
      capacity = null;
      say(error.name === 'EncodingError' ? 'This file could not be read as an image. Choose another file.' : error.message, true);
    } finally {
      URL.revokeObjectURL(url);
      if (request === revision) {
        busy = false;
        updateControls();
      }
    }
  });

  message.addEventListener('input', function () {
    const hadDownload = !!outputURL;
    clearDownload();
    updateControls();
    if (hadDownload) say('Message changed. Select Embed message to update the image and download.');
  });

  embedButton.addEventListener('click', async function () {
    if (embedButton.disabled) return;
    const request = revision;
    clearDownload();
    clearRevealed();
    busy = true;
    updateControls();
    say('Embedding message and preparing PNG…');
    try {
      const result = codec.embed(pixels.data, message.value);
      pixels = new ImageData(result.data, pixels.width, pixels.height);
      context.putImageData(pixels, 0, 0);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
      if (request !== revision) return;
      if (!blob) throw new Error('The PNG could not be created. Try a smaller image.');
      outputURL = URL.createObjectURL(blob);
      download.href = outputURL;
      download.download = filename + '-hidden.png';
      download.hidden = false;
      find('used').textContent = result.usedPixels.toLocaleString();
      say('Message embedded in ' + result.usedPixels.toLocaleString() + ' pixels, including markers. Download the PNG to keep it.');
    } catch (error) {
      if (request === revision) say(error.message, true);
    } finally {
      if (request === revision) { busy = false; updateControls(); }
    }
  });

  revealButton.addEventListener('click', function () {
    if (!pixels || busy) return;
    clearRevealed();
    const result = codec.inspect(pixels.data);
    showDetection(result);
    if (result.status === 'found') {
      revealed.value = result.message;
      find('reveal-result').hidden = false;
      say('Message revealed. Both markers and the integrity checksum are valid.');
    }
  });

  window.addEventListener('pagehide', clearDownload);
  updateControls();
})();
