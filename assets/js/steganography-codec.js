(function (root) {
  'use strict';

  // Version 1: start marker, uint32 length, uint32 CRC-32, UTF-8, end marker.
  // Integer fields are big-endian; each byte is stored most-significant bits first.
  const encoder = new TextEncoder();
  const START = encoder.encode('RDSTEG01');
  const END = encoder.encode('ENDSTEG!');
  const HEADER_BYTES = 16;
  const OVERHEAD_BYTES = HEADER_BYTES + END.length;

  function measure(rgba) {
    let usablePixels = 0;
    for (let i = 3; i < rgba.length; i += 4) {
      if (rgba[i] === 255) usablePixels += 1;
    }
    const totalBytes = Math.floor(usablePixels * 6 / 8);
    return {
      usablePixels,
      totalBytes,
      capacity: Math.max(0, totalBytes - OVERHEAD_BYTES),
      canStoreFrame: totalBytes >= OVERHEAD_BYTES
    };
  }

  function pixelsForBytes(messageBytes) {
    return Math.ceil((messageBytes + OVERHEAD_BYTES) * 8 / 6);
  }

  function byteLength(message) {
    return encoder.encode(message).length;
  }

  function checksum(bytes) {
    let crc = 0xffffffff;
    for (const byte of bytes) {
      crc ^= byte;
      for (let bit = 0; bit < 8; bit += 1) {
        crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
      }
    }
    return (crc ^ 0xffffffff) >>> 0;
  }

  function channels(rgba) {
    let offset = 0;
    return function next() {
      while (offset < rgba.length) {
        const pixel = offset - offset % 4;
        // Transparent RGB values can change during canvas/PNG conversion.
        if (rgba[pixel + 3] !== 255) {
          offset = pixel + 4;
        } else if (offset % 4 === 3) {
          offset += 1;
        } else {
          return offset++;
        }
      }
      throw new Error('The image does not have enough usable pixels.');
    };
  }

  function embed(rgba, message) {
    const info = measure(rgba);
    const payload = encoder.encode(message);
    if (!info.canStoreFrame || payload.length > info.capacity) {
      throw new Error('The message exceeds this image’s capacity.');
    }
    const frame = new Uint8Array(payload.length + OVERHEAD_BYTES);
    const header = new DataView(frame.buffer);
    frame.set(START);
    header.setUint32(8, payload.length);
    header.setUint32(12, checksum(payload));
    frame.set(payload, HEADER_BYTES);
    frame.set(END, HEADER_BYTES + payload.length);

    const data = new Uint8ClampedArray(rgba);
    const next = channels(data);
    for (const byte of frame) {
      for (let shift = 6; shift >= 0; shift -= 2) {
        const index = next();
        data[index] = (data[index] & 0xfc) | ((byte >>> shift) & 3);
      }
    }
    return { data, messageBytes: payload.length, usedPixels: pixelsForBytes(payload.length) };
  }

  function inspect(rgba) {
    const info = measure(rgba);
    const absent = { status: 'absent', reason: 'No compatible watermark found.' };
    const damaged = { status: 'damaged', reason: 'Watermark found, but the message is incomplete or damaged.' };
    if (info.totalBytes < START.length) return absent;

    const next = channels(rgba);
    function read(count) {
      const bytes = new Uint8Array(count);
      for (let i = 0; i < count; i += 1) {
        for (let pair = 0; pair < 4; pair += 1) {
          bytes[i] = (bytes[i] << 2) | (rgba[next()] & 3);
        }
      }
      return bytes;
    }

    const marker = read(START.length);
    if (!marker.subarray(0, 6).every((byte, i) => byte === START[i])) return absent;
    if (!marker.every((byte, i) => byte === START[i])) {
      return { status: 'unsupported', reason: 'This watermark version is not supported.' };
    }
    if (!info.canStoreFrame) return damaged;
    const fields = read(8);
    const header = new DataView(fields.buffer);
    const length = header.getUint32(0);
    // Validate the declared size before allocating or reading any payload.
    if (length > info.capacity) return damaged;
    const expectedChecksum = header.getUint32(4);
    const payload = read(length);
    const end = read(END.length);
    if (!end.every((byte, i) => byte === END[i]) || checksum(payload) !== expectedChecksum) return damaged;
    try {
      // Preserve a leading U+FEFF character as part of the user's message.
      const message = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(payload);
      return { status: 'found', message, messageBytes: length, usedPixels: pixelsForBytes(length) };
    } catch (_) {
      return damaged;
    }
  }

  const codec = { measure, pixelsForBytes, byteLength, embed, inspect };
  if (typeof module !== 'undefined' && module.exports) module.exports = codec;
  else root.SteganographyCodec = codec;
})(globalThis);
