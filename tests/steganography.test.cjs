const test = require('node:test');
const assert = require('node:assert/strict');
const codec = require('../assets/js/steganography-codec.js');

function image(count) {
  const data = new Uint8ClampedArray(count * 4);
  for (let i = 0; i < count; i += 1) {
    data.set([(i * 37) % 256, (i * 73) % 256, (i * 113) % 256, 255], i * 4);
  }
  return data;
}

// Independent byte access for fully opaque fixtures, to exercise malformed frames.
function writeByte(data, offset, value) {
  for (let pair = 0; pair < 4; pair += 1) {
    const channel = offset * 4 + pair;
    const index = Math.floor(channel / 3) * 4 + channel % 3;
    data[index] = (data[index] & 252) | ((value >>> (6 - pair * 2)) & 3);
  }
}

function readByte(data, offset) {
  let value = 0;
  for (let pair = 0; pair < 4; pair += 1) {
    const channel = offset * 4 + pair;
    value = value * 4 + (data[Math.floor(channel / 3) * 4 + channel % 3] & 3);
  }
  return value;
}

test('arbitrary colors round trip, preserving input, alpha, and upper six bits', () => {
  const source = image(512);
  const before = source.slice();
  const text = 'Hello, 世界! 🐉\nLine two: café <script>alert(1)</script>\u0000\ufeff RDSTEG01 ENDSTEG!';
  const result = codec.embed(source, text);
  const decoded = codec.inspect(result.data);
  assert.equal(decoded.status, 'found');
  assert.equal(decoded.message, text);
  assert.equal(decoded.messageBytes, Buffer.byteLength(text));
  assert.equal(decoded.usedPixels, result.usedPixels);
  assert.deepEqual(source, before);
  for (let i = 0; i < source.length; i += 1) {
    if (i % 4 === 3) assert.equal(source[i], result.data[i]);
    else {
      assert.equal(source[i] & 252, result.data[i] & 252);
      assert(Math.abs(source[i] - result.data[i]) <= 3);
    }
  }
  assert.deepEqual(result.data.slice(result.usedPixels * 4), source.slice(result.usedPixels * 4));
});

test('known framing, CRC-32 check vector, bit order, and exact storage count', () => {
  const encoded = codec.embed(image(44), '123456789');
  assert.equal(encoded.usedPixels, 44);
  const bytes = Array.from({ length: 33 }, (_, i) => readByte(encoded.data, i));
  assert.deepEqual(bytes.slice(0, 8), Array.from(Buffer.from('RDSTEG01')));
  assert.deepEqual(bytes.slice(8, 16), [0, 0, 0, 9, 0xcb, 0xf4, 0x39, 0x26]);
  assert.deepEqual(bytes.slice(25), Array.from(Buffer.from('ENDSTEG!')));
});

test('capacity boundaries and UTF-8 byte limits', () => {
  assert.equal(codec.measure(image(44)).capacity, 9);
  assert.equal(codec.measure(image(31)).canStoreFrame, false);
  assert.throws(() => codec.embed(image(31), ''), /capacity/);
  assert.equal(codec.inspect(codec.embed(image(32), '').data).message, '');
  assert.equal(codec.inspect(codec.embed(image(44), '123456789').data).message, '123456789');
  assert.throws(() => codec.embed(image(44), '1234567890'), /capacity/);
  assert.equal(codec.byteLength('🐉🐉🐉'), 12);
  assert.throws(() => codec.embed(image(44), '🐉🐉🐉'), /capacity/);
  assert.equal(codec.inspect(codec.embed(image(64), '\ufeffmessage').data).message, '\ufeffmessage');
});

test('skips transparent and partially transparent pixels without changing them', () => {
  const source = image(200);
  for (let i = 0; i < 100; i += 1) source[i * 4 + 3] = i % 2 ? 128 : 0;
  assert.equal(codec.measure(source).usablePixels, 100);
  const result = codec.embed(source, 'Opaque pixels only');
  assert.equal(codec.inspect(result.data).message, 'Opaque pixels only');
  assert.deepEqual(result.data.slice(0, 400), source.slice(0, 400));
  const transparent = new Uint8ClampedArray(400);
  assert.equal(codec.inspect(transparent).status, 'absent');
  assert.throws(() => codec.embed(transparent, 'hello'), /capacity/);
});

test('unmarked and tiny images report no compatible watermark', () => {
  for (const count of [0, 1, 10, 31, 1000]) {
    assert.equal(codec.inspect(image(count)).status, 'absent');
  }
});

test('detects corrupt lengths, payloads, end markers, and truncated data', () => {
  const original = codec.embed(image(100), 'hello').data;
  const length = original.slice();
  for (let i = 8; i < 12; i += 1) writeByte(length, i, 255);
  assert.equal(codec.inspect(length).status, 'damaged');
  const payload = original.slice();
  writeByte(payload, 16, 0);
  assert.equal(codec.inspect(payload).status, 'damaged');
  const end = original.slice();
  writeByte(end, 21, 0);
  assert.equal(codec.inspect(end).status, 'damaged');
  assert.equal(codec.inspect(original.slice(0, 30 * 4)).status, 'damaged');
});

test('unsupported versions and malformed UTF-8 are never returned as messages', () => {
  const version = codec.embed(image(100), 'x').data;
  writeByte(version, 7, 0x32);
  assert.equal(codec.inspect(version).status, 'unsupported');
  const invalidUTF8 = codec.embed(image(100), 'x').data;
  writeByte(invalidUTF8, 16, 0xff);
  // CRC-32 of the single byte 0xff is 0xff000000.
  [0xff, 0, 0, 0].forEach((byte, i) => writeByte(invalidUTF8, 12 + i, byte));
  assert.equal(codec.inspect(invalidUTF8).status, 'damaged');
});

test('replacing a longer message uses the new length and pixel count', () => {
  const first = codec.embed(image(500), 'A long message that used more of the image.');
  const second = codec.embed(first.data, 'Hi');
  assert.equal(codec.inspect(second.data).message, 'Hi');
  assert.equal(codec.inspect(second.data).usedPixels, 35);
});
