// Optional proxy: serves the page on :3000 and forwards images to the Python service.
// The Python service (python -m vision_assist.api) already serves everything on :8001 by itself.
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;
const VISION_API = process.env.VISION_API || 'http://127.0.0.1:8001';
const MAX_BODY = 20 * 1024 * 1024;

// Forward the image to the Python vision service and return its description.
async function process_input(buffer, contentType) {
  const res = await fetch(`${VISION_API}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': contentType || 'application/octet-stream' },
    body: buffer,
    signal: AbortSignal.timeout(30000),
  });
  let data;
  try { data = await res.json(); } catch { throw new Error(`Vision service returned ${res.status}`); }
  if (!res.ok) throw new Error(data.error || `Vision service returned ${res.status}`);
  return data;
}

function sendJson(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/api/process') {
    const chunks = [];
    let size = 0;
    req.on('error', () => {});
    req.on('data', (c) => {
      size += c.length;
      if (size > MAX_BODY) {
        sendJson(res, 413, { output: 'Error: image too large' });
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', async () => {
      if (size > MAX_BODY) return;
      try {
        sendJson(res, 200, await process_input(Buffer.concat(chunks), req.headers['content-type']));
      } catch (err) {
        const msg = err.cause && err.cause.code === 'ECONNREFUSED'
          ? 'Vision service is not running. Start it with: python -m vision_assist.api'
          : err.message;
        sendJson(res, 502, { output: `Error: ${msg}` });
      }
    });
    return;
  }
  if (req.method === 'GET' && (req.url === '/' || req.url === '/index.html')) {
    const stream = fs.createReadStream(path.join(__dirname, 'public', 'index.html'));
    stream.on('error', () => { res.writeHead(500); res.end('index.html missing'); });
    stream.on('open', () => res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }));
    return stream.pipe(res);
  }
  res.writeHead(404);
  res.end('Not found');
}).listen(PORT, () => console.log(`http://localhost:${PORT} -> ${VISION_API}`));
