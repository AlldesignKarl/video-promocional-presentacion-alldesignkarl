// Graba la tienda online (alldesignkarl.com) tal y como se ve en un móvil.
//
//   NODE_PATH=$(npm root -g) node tools/tienda.cjs
//
// Abre la copia del código de la web (tienda/index.html) en Chromium, en local, y responde a sus
// llamadas a Wix con el catálogo real (tools/catalogo.json) y las fotos reales de las piezas
// (assets/fotos). No se conecta a ningún servicio externo.
//
// Salida en build/tienda: pagina.jpg (la página entera, sin la cabecera), cabecera.png y
// tienda.json (dónde empieza cada sección, para que el vídeo sepa hasta dónde bajar).
"use strict";

const fs = require("fs");
const http = require("http");
const path = require("path");
const { chromium } = require("playwright");

const ROOT = path.resolve(__dirname, "..");
const FOTOS = path.join(ROOT, "assets", "fotos");
const OUT = path.join(ROOT, "build", "tienda");
const ANCHO = 414;          // ancho CSS de un móvil grande
const ESCALA = 2;           // píxeles por punto: 828 px de ancho
const TYPES = { ".html": "text/html; charset=utf-8", ".woff2": "font/woff2", ".jpg": "image/jpeg", ".png": "image/png" };

function serve() {
  const srv = http.createServer((req, res) => {
    const p = path.join(ROOT, decodeURIComponent(new URL(req.url, "http://x").pathname));
    if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { "content-type": TYPES[path.extname(p)] || "application/octet-stream" });
    fs.createReadStream(p).pipe(res);
  });
  return new Promise((r) => srv.listen(0, "127.0.0.1", () => r({ srv, base: `http://127.0.0.1:${srv.address().port}` })));
}

function fontCss(base) {
  const f = (file) => `url(${base}/assets/fonts/${file}) format('woff2')`;
  return [
    ["Cormorant Garamond", "normal", 300, "cormorant-garamond-latin-300-normal.woff2"],
    ["Cormorant Garamond", "normal", 400, "cormorant-garamond-latin-400-normal.woff2"],
    ["Cormorant Garamond", "normal", 500, "cormorant-garamond-latin-500-normal.woff2"],
    ["Cormorant Garamond", "normal", 600, "cormorant-garamond-latin-600-normal.woff2"],
    ["Cormorant Garamond", "italic", 400, "cormorant-garamond-latin-400-italic.woff2"],
    ["Cormorant Garamond", "italic", 500, "cormorant-garamond-latin-500-italic.woff2"],
    ["Pinyon Script", "normal", 400, "pinyon-script-latin-400-normal.woff2"],
  ].map(([fam, st, w, file]) => `@font-face{font-family:'${fam}';font-style:${st};font-weight:${w};font-display:block;src:${f(file)};}`).join("\n");
}

(async () => {
  const cat = JSON.parse(fs.readFileSync(path.join(__dirname, "catalogo.json"), "utf8"));
  const { srv, base } = await serve();
  const browser = await chromium.launch({ args: ["--force-color-profile=srgb"] });
  const ctx = await browser.newContext({ viewport: { width: ANCHO, height: 844 }, deviceScaleFactor: ESCALA, isMobile: true, hasTouch: true, reducedMotion: "reduce", locale: "es-ES" });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => process.stderr.write("pageerror: " + e.message + "\n"));

  const json = (route, body) => route.fulfill({ status: 200, contentType: "application/json", headers: { "Access-Control-Allow-Origin": "*" }, body: JSON.stringify(body) });
  await page.route("https://www.wixapis.com/**", (route) => {
    const req = route.request();
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*", "Access-Control-Allow-Methods": "*" } });
    const p = new URL(req.url()).pathname;
    if (p === "/oauth2/token") return json(route, { access_token: "local", refresh_token: "local", expires_in: 14400 });
    if (p === "/stores/v3/products/query") return json(route, { products: cat.products });
    if (p.startsWith("/ecom/v2/carts/current")) return json(route, { cart: { id: "local", lineItems: [], currency: "EUR" } });
    return json(route, {});
  });
  await page.route("https://static.wixstatic.com/**", (route) => {
    const name = new URL(route.request().url()).pathname.split("/media/")[1].split("/")[0];
    const file = cat.fotos[name] && path.join(FOTOS, cat.fotos[name]);
    if (!file || !fs.existsSync(file)) return route.fulfill({ status: 404, body: "" });
    return route.fulfill({ status: 200, contentType: "image/jpeg", body: fs.readFileSync(file) });
  });
  await page.route("https://fonts.googleapis.com/**", (route) => route.fulfill({ status: 200, contentType: "text/css", body: fontCss(base) }));
  await page.route(/^https:\/\/(fonts\.gstatic\.com|manage\.wix\.com)\//, (route) => route.abort());

  await page.goto(base + "/tienda/index.html", { waitUntil: "load" });
  await page.waitForSelector("#grid .card:not(.skeleton) img");
  // los precios cambian con el tiempo: el vídeo no los enseña
  await page.addStyleTag({ content: ".card-price{visibility:hidden!important}.toast,#toast,#b2b-bar,#b2b-shopbar{display:none!important}" });
  await page.evaluate(async () => {
    // baja por toda la página para que carguen las fotos diferidas
    for (let y = 0; y < document.body.scrollHeight; y += 400) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 40)); }
    window.scrollTo(0, 0);
    await Promise.all([...document.images].map((im) => (im.complete ? 0 : new Promise((r) => { im.onload = im.onerror = r; }))));
    await document.fonts.ready;
  });
  await page.waitForTimeout(400);

  fs.mkdirSync(OUT, { recursive: true });
  const header = await page.$("header");
  await header.screenshot({ path: path.join(OUT, "cabecera.png") });
  const info = await page.evaluate(() => {
    const top = (s) => { const e = document.querySelector(s); return e ? Math.round(e.getBoundingClientRect().top + window.scrollY) : null; };
    const h = document.querySelector("header");
    return {
      cabecera: Math.round(h.getBoundingClientRect().height),
      secciones: { inicio: top("#inicio"), colecciones: top("#colecciones"), rejilla: top("#grid"), nosotros: top("#nosotros"), mayoristas: top("#mayoristas"), contacto: top("#contacto") },
      alto: document.documentElement.scrollHeight,
      // recuadros que el vídeo resalta (en puntos CSS de la página)
      puntos: Object.fromEntries([
        ["minimo", "#mayoristas", "Pedido mínimo de 100"], ["envio", "#mayoristas", "Envío a toda Europa"],
        ["cuenta", "#mayoristas", "Crea tu cuenta de mayorista"], ["boton", "#mayoristas", "Crear cuenta de mayorista"],
      ].map(([k, sec, txt]) => {
        // el elemento más pequeño que contiene el texto, y de ahí su tarjeta (el primer padre con borde o fondo)
        const all = [...document.querySelectorAll(sec + " *")].filter((e) => e.offsetParent && e.textContent.trim().startsWith(txt));
        let e = all[all.length - 1];
        for (let p = e; p && p !== document.body; p = p.parentElement) {
          const cs = getComputedStyle(p);
          if (cs.borderTopWidth !== "0px" || cs.backgroundColor !== "rgba(0, 0, 0, 0)") { e = p; break; }
        }
        const r = e.getBoundingClientRect();
        return [k, { x: Math.round(r.left), y: Math.round(r.top + window.scrollY), w: Math.round(r.width), h: Math.round(r.height) }];
      })),
    };
  });
  await page.addStyleTag({ content: "header{visibility:hidden!important}" });
  const fin = info.secciones.contacto || info.alto;
  await page.screenshot({ path: path.join(OUT, "pagina.png"), fullPage: true, clip: { x: 0, y: 0, width: ANCHO, height: fin } });
  info.ancho = ANCHO;
  info.escala = ESCALA;
  info.fin = fin;
  fs.writeFileSync(path.join(OUT, "tienda.json"), JSON.stringify(info, null, 1));
  console.log(JSON.stringify(info));
  await browser.close();
  srv.close();
})().catch((e) => { process.stderr.write(String(e && e.stack || e) + "\n"); process.exit(1); });
