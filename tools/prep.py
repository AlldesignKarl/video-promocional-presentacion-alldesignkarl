"""Prepara el material del vídeo en build/prep.

    python3 tools/prep.py

- nosotros.jpg: la foto de los dos, recortada por encima del rótulo que lleva la imagen original.
- Fotos de las piezas a un tamaño cómodo para el navegador (máx. 1400 px).
- tienda.jpg: la captura de la tienda online (build/tienda/pagina.png, de tools/tienda.cjs).
"""
import json
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
FOTOS = ROOT / "assets" / "fotos"
OUT = ROOT / "build" / "prep"


def guardar(im, nombre, maximo=1400, q=90):
    im = im.convert("RGB")
    if max(im.size) > maximo:
        k = maximo / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    dest = OUT / nombre
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, quality=q, optimize=True)
    return im.size


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    # La foto original lleva un rótulo («Burnt out but make it fashion») a la altura de los brazos:
    # se usa solo la parte de arriba (caras y hombros), sin el rótulo.
    nos = Image.open(FOTOS / "nosotros.webp").convert("RGB")
    w, h = nos.size
    caja = (0, round(h * 0.16), w, round(h * 0.565))
    nos = nos.crop(caja)
    nos = ImageEnhance.Color(nos).enhance(0.92)
    meta["nosotros"] = guardar(nos, "nosotros.jpg", 1600, 92)

    for f in sorted(FOTOS.rglob("*.jpg")):
        rel = f.relative_to(FOTOS)
        meta[str(rel)] = guardar(Image.open(f), str(rel), 1400 if "colores" not in str(rel) else 900)

    t = ROOT / "build" / "tienda"
    if not (t / "pagina.png").exists():
        raise SystemExit("falta build/tienda/pagina.png: ejecuta antes  node tools/tienda.cjs")
    pag = Image.open(t / "pagina.png").convert("RGB")
    pag.save(OUT / "tienda.jpg", quality=90, optimize=True)
    shutil.copy(t / "tienda.json", OUT / "tienda.json")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print("material listo en", OUT)


if __name__ == "__main__":
    main()
