// Renderiza src/promo.html fotograma a fotograma.
//
//   node tools/render_promo.cjs --stills 0.5,3,8 --out build/stills   → PNG sueltos para revisar
//   node tools/render_promo.cjs --frames 0:100                         → flujo binario por stdout
//   node tools/render_promo.cjs --list 0,4,8                           → igual, con fotogramas sueltos
//
// En modo --frames cada fotograma sale como varias submuestras (desenfoque de movimiento) que
// tools/build.py promedia: [u32 índice][u32 n] y n × ([u32 bytes][PNG]).
"use strict";

const fs = require("fs");
const http = require("http");
const path = require("path");
const { chromium } = require("playwright");

const ROOT = path.resolve(__dirname, "..");
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".json": "application/json", ".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".woff2": "font/woff2" };

function arg(name, def) {
  const i = process.argv.indexOf(name);
  return i === -1 ? def : process.argv[i + 1];
}

function serve() {
  // servidor local mínimo: solo archivos del repositorio
  const srv = http.createServer((req, res) => {
    const p = path.join(ROOT, decodeURIComponent(new URL(req.url, "http://x").pathname));
    if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { "content-type": TYPES[path.extname(p)] || "application/octet-stream" });
    fs.createReadStream(p).pipe(res);
  });
  return new Promise((r) => srv.listen(0, "127.0.0.1", () => r({ srv, base: `http://127.0.0.1:${srv.address().port}` })));
}

(async () => {
  const { srv, base } = await serve();
  const browser = await chromium.launch({ args: ["--force-color-profile=srgb", "--disable-lcd-text", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  page.on("pageerror", (e) => process.stderr.write("pageerror: " + e.message + "\n"));
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") process.stderr.write("console: " + m.text() + "\n"); });
  await page.goto(base + "/src/promo.html", { waitUntil: "load" });
  await page.evaluate(() => window.PROMO.init());
  const cdp = await page.context().newCDPSession(page);
  const grab = async () => Buffer.from((await cdp.send("Page.captureScreenshot", { format: "png", optimizeForSpeed: true })).data, "base64");
  const render = (t) => page.evaluate((x) => window.PROMO.renderAt(x), t);

  const stills = arg("--stills");
  if (stills) {
    const out = path.resolve(arg("--out", "build/stills"));
    fs.mkdirSync(out, { recursive: true });
    for (const s of stills.split(",")) {
      const t = Number(s);
      await render(t);
      fs.writeFileSync(path.join(out, "t" + t.toFixed(2).padStart(5, "0") + ".png"), await grab());
      process.stderr.write("still " + t + "\n");
    }
  } else {
    let list;
    if (arg("--list")) list = arg("--list").split(",").map(Number);
    else { const [a, b] = arg("--frames").split(":").map(Number); list = Array.from({ length: b - a }, (_, k) => a + k); }
    const fps = await page.evaluate(() => window.PROMO.fps);
    const u32 = (n) => { const x = Buffer.alloc(4); x.writeUInt32LE(n); return x; };
    for (const i of list) {
      const t = i / fps;
      const [n, shutter] = await page.evaluate((x) => [window.PROMO.samplesAt(x), window.PROMO.shutterAt(x)], t);
      const parts = [u32(i), u32(n)];
      for (let k = 0; k < n; k++) {
        const off = n === 1 ? 0 : ((k + 0.5) / n - 0.5) * shutter / fps;
        await render(Math.max(0, t + off));
        const png = await grab();
        parts.push(u32(png.length), png);
      }
      const buf = Buffer.concat(parts);
      if (!process.stdout.write(buf)) await new Promise((r) => process.stdout.once("drain", r));
    }
  }
  await browser.close();
  srv.close();
})().catch((e) => { process.stderr.write(String(e && e.stack || e) + "\n"); process.exit(1); });
