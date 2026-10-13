"""Locución del vídeo con voz neuronal (Kokoro v1.0, voz española «ef_dora»).

    python3 tools/voz.py            # genera audio/locucion.flac y src/tiempos.js

Lee el guion de tools/guion.json, sintetiza cada frase por separado con sherpa-onnx (todo en
local, sin servicios externos), las coloca en la línea de tiempo del vídeo y calcula cuándo
empieza y acaba cada trozo de subtítulo buscando las pausas reales de la voz.

Requisitos: pip install sherpa-onnx soundfile numpy. El modelo se descarga solo la primera vez
(GitHub, k2-fsa/sherpa-onnx, «kokoro-multi-lang-v1_0», ~330 MB) en build/modelos.
"""
import json
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
MODELOS = ROOT / "build" / "modelos"
KOKORO = MODELOS / "kokoro-multi-lang-v1_0"
URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2"
SR = 48000

INICIO = 0.8        # la voz entra a los 0,8 s
FIN_VOZ = 47.4      # y termina hacia los 47,4 s (el vídeo dura 50 s)
# pausa después de cada frase (se escalan un poco para cuadrar con FIN_VOZ)
PAUSAS = {"hola": 0.7, "historia": 0.7, "como": 0.6, "productos": 0.5, "personaliza": 0.5,
          "calidad": 0.8, "mayoristas": 0.5, "tienda": 0.8}


def modelo():
    if (KOKORO / "model.onnx").exists():
        return
    MODELOS.mkdir(parents=True, exist_ok=True)
    tar = MODELOS / "kokoro.tar.bz2"
    print("descargando el modelo de voz…", flush=True)
    urllib.request.urlretrieve(URL, tar)
    with tarfile.open(tar) as t:
        t.extractall(MODELOS, filter="data")
    tar.unlink()


def sintetizar(guion):
    import sherpa_onnx

    d = str(KOKORO)
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=d + "/model.onnx", voices=d + "/voices.bin", tokens=d + "/tokens.txt",
            data_dir=d + "/espeak-ng-data", dict_dir=d + "/dict", lexicon=d + "/lexicon-us-en.txt", lang="es"),
        num_threads=4))
    tts = sherpa_onnx.OfflineTts(cfg)
    sid = {"ef_dora": 28, "em_alex": 29, "em_santa": 53}[guion["voz"]]
    clips = []
    for f in guion["frases"]:
        texto = f["texto"]
        for k, v in guion["pronunciacion"].items():
            texto = texto.replace(k, v)
        a = tts.generate(texto, sid=sid, speed=guion["velocidad"])
        x = np.asarray(a.samples, dtype=np.float32)
        x = remuestrear(x, a.sample_rate)
        # fuera el silencio de los bordes (se deja un respiro de 30 ms)
        env = envolvente(x)
        on = np.where(env > 0.008)[0]
        a0, a1 = max(0, on[0] - int(0.03 * SR)), min(len(x), on[-1] + int(0.06 * SR))
        clips.append(x[a0:a1])
        print(f"  {f['id']:<12} {len(clips[-1]) / SR:5.2f} s", flush=True)
    return clips


def remuestrear(x, sr):
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                        "-af", f"aresample={SR}:resampler=soxr:precision=28", "-f", "f32le", "-"],
                       input=x.tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def envolvente(x, win=0.01):
    n = int(win * SR)
    return np.sqrt(np.convolve(x * x, np.ones(n) / n, mode="same"))


FUNCION = {"a", "al", "con", "de", "del", "e", "el", "en", "la", "las", "los", "por", "que", "se", "tu", "un", "una", "y"}


def trozos(texto):
    """Divide una frase en trozos de subtítulo: por la puntuación, uniendo los muy cortos y partiendo
    los largos por la mitad (sin dejar una preposición o un artículo colgando al final)."""
    partes = [p.strip() for p in re.split(r"(?<=[,:.])\s+", texto) if p.strip()]
    unidas = []
    for p in partes:
        if unidas and (len(unidas[-1]) < 12 or len(p) < 12) and len(unidas[-1]) + len(p) < 40:
            unidas[-1] += " " + p
        else:
            unidas.append(p)
    out = []
    for p in unidas:
        if len(p) <= 34:
            out.append((p, True))
            continue
        pal = p.split()
        mejor, corte = 1e9, 1
        for i in range(1, len(pal)):
            izq = len(" ".join(pal[:i]))
            coste = abs(izq - len(p) / 2) + (100 if pal[i - 1].lower() in FUNCION else 0)
            if coste < mejor:
                mejor, corte = coste, i
        out.append((" ".join(pal[:corte]), False))
        out.append((" ".join(pal[corte:]), True))
    return out  # (texto, termina_en_pausa)


def pausas(x, minimo=0.05):
    """Silencios internos (centro, duración) de un clip de voz."""
    env = envolvente(x, 0.015)
    sil = env < 0.3 * np.median(env)
    res, i = [], 0
    while i < len(sil):
        if sil[i]:
            j = i
            while j < len(sil) and sil[j]:
                j += 1
            if i > 0 and j < len(sil) and (j - i) / SR >= minimo:
                res.append(((i + j) / 2 / SR, (j - i) / SR))
            i = j
        else:
            i += 1
    return res


def silabas(t):
    return max(1, len(re.findall(r"[aeiouáéíóúü]+", t.lower())))


def tiempos_trozos(texto, x, t0):
    """Reparte la frase por sílabas (las comas cuentan como una pausa corta) y ajusta cada corte
    con puntuación a la pausa real más cercana de la voz."""
    tr = trozos(texto)
    dur = len(x) / SR
    peso = [silabas(t) + (0.8 if p and i < len(tr) - 1 else 0) for i, (t, p) in enumerate(tr)]
    sil = pausas(x)
    bordes, acc = [0.0], 0.0
    for i in range(len(tr) - 1):
        acc += peso[i]
        b = dur * acc / sum(peso)
        if tr[i][1] and sil:
            c, d = min(sil, key=lambda s: abs(s[0] - b))
            if abs(c - b) < 0.45:
                b = c
        bordes.append(max(b, bordes[-1] + 0.2))
    bordes.append(dur)
    return [{"t": txt, "a": round(t0 + bordes[i], 3), "b": round(t0 + bordes[i + 1], 3)} for i, (txt, _) in enumerate(tr)]


def procesar(x):
    """Voz más cálida y presente: paso alto, ecualización suave, compresión y una sala muy discreta."""
    flt = ("highpass=f=75,equalizer=f=200:t=q:w=1:g=1.5,equalizer=f=3200:t=q:w=1.2:g=2,"
           "equalizer=f=7500:t=q:w=1.5:g=-2.5,acompressor=threshold=0.12:ratio=2.5:attack=8:release=120:makeup=1.6,"
           "aecho=0.9:0.9:23|37|53:0.06|0.045|0.03")
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-af", flt,
                        "-f", "f32le", "-"], input=x.tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def main():
    guion = json.loads((ROOT / "tools" / "guion.json").read_text(encoding="utf-8"))
    modelo()
    print("sintetizando la locución…", flush=True)
    clips = sintetizar(guion)
    ids = [f["id"] for f in guion["frases"]]
    habla = sum(len(c) for c in clips) / SR
    base = sum(PAUSAS.get(i, 0) for i in ids[:-1])
    escala = (FIN_VOZ - INICIO - habla) / base
    if escala < 0.6:
        sys.exit(f"la locución dura demasiado ({habla:.1f} s): sube la velocidad en tools/guion.json")
    total = 50.0
    pista = np.zeros(int(total * SR), dtype=np.float32)
    t = INICIO
    frases = []
    for f, c in zip(guion["frases"], clips):
        i0 = int(round(t * SR))
        pista[i0:i0 + len(c)] += c
        frases.append({"id": f["id"], "texto": f["texto"], "a": round(t, 3), "b": round(t + len(c) / SR, 3),
                       "trozos": tiempos_trozos(f["texto"], c, t)})
        t += len(c) / SR + PAUSAS.get(f["id"], 0) * escala
    pista = procesar(pista)[: len(pista)]
    (ROOT / "audio").mkdir(exist_ok=True)
    sf.write(ROOT / "audio" / "locucion.flac", pista, SR, subtype="PCM_24")
    (ROOT / "src" / "tiempos.js").write_text(
        "// Generado por tools/voz.py: cuándo suena cada frase de la locución y cada trozo de subtítulo.\n"
        "window.TIEMPOS = " + json.dumps(frases, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
    for fr in frases:
        print(f"  {fr['a']:6.2f}–{fr['b']:6.2f}  {fr['id']}")
    print(f"voz: {habla:.1f} s · pausas ×{escala:.2f} · audio/locucion.flac · src/tiempos.js")


if __name__ == "__main__":
    main()
