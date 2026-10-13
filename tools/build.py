"""Construye el vídeo completo.

    python3 tools/build.py            # todo (reutiliza los fotogramas ya renderizados)
    python3 tools/build.py --fresh    # vuelve a renderizar todos los fotogramas

Pasos: material (tools/prep.py) → render del vídeo en paralelo con desenfoque de movimiento
(tools/render_promo.cjs) → música (tools/musica.py) → mezcla de la locución (audio/locucion.flac,
de tools/voz.py) con la música, que baja sola cuando habla la voz → vídeo final a -14 LUFS,
versión ligera para WhatsApp y portada.
"""
import json
import os
import struct
import subprocess
import sys
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
OUT = ROOT / "video"
FPS = 30
DUR = 50.0
SR = 48000
NAME = "AlldesignKarl_Presentacion_1080x1920"
NODE_ENV = dict(os.environ, NODE_PATH=subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip())


def run(cmd, **kw):
    print("·", " ".join(str(c) for c in cmd)[:160], flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, **kw)


def prep():
    if not (BUILD / "tienda" / "tienda.json").exists():
        run(["node", "tools/tienda.cjs"], env=NODE_ENV)
    if not (BUILD / "prep" / "meta.json").exists():
        run([sys.executable, "tools/prep.py"])


def read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError
    return b


def render_frames(workers):
    frames = BUILD / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    total = int(round(DUR * FPS))
    todo = [i for i in range(total) if not (frames / f"f{i:05d}.png").exists()]
    if not todo:
        return total
    chunks = [todo[k::workers] for k in range(workers)]
    errors = []

    def worker(idx):
        mine = chunks[idx]
        p = subprocess.Popen(["node", "tools/render_promo.cjs", "--list", ",".join(map(str, mine))],
                             cwd=ROOT, env=NODE_ENV, stdout=subprocess.PIPE, stderr=sys.stderr)
        rng = np.random.default_rng(idx + 1)
        try:
            while True:
                try:
                    fi, n = struct.unpack("<II", read_exact(p.stdout, 8))
                except EOFError:
                    break
                acc = None
                for _ in range(n):
                    (ln,) = struct.unpack("<I", read_exact(p.stdout, 4))
                    im = np.asarray(Image.open(BytesIO(read_exact(p.stdout, ln))).convert("RGB"), dtype=np.float32)
                    acc = im if acc is None else acc + im
                acc /= n
                # grano muy fino + tramado contra el banding de los fondos lisos
                g = np.random.default_rng(fi * 7919).standard_normal(acc.shape[:2]).astype(np.float32)
                lum = acc.mean(2, keepdims=True) / 255
                acc += g[..., None] * (1.6 * (1 - lum) + 0.9)
                acc += rng.random(acc.shape, dtype=np.float32) - rng.random(acc.shape, dtype=np.float32)
                Image.fromarray(np.clip(acc + 0.5, 0, 255).astype(np.uint8)).save(frames / f"f{fi:05d}.png", compress_level=1)
                if fi % 30 < workers:
                    print(f"  fotograma {fi}/{total}", flush=True)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        if p.wait() != 0:
            errors.append(RuntimeError(f"render {idx} terminó con error"))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(workers) if chunks[k]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    missing = [i for i in range(total) if not (frames / f"f{i:05d}.png").exists()]
    if missing:
        raise RuntimeError(f"faltan fotogramas: {missing[:10]}")
    return total


def lufs(x):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                        "-af", "ebur128", "-f", "null", "-"], input=x.astype(np.float32).tobytes(), capture_output=True)
    txt = p.stderr.decode()
    return float(txt[txt.rindex("I:"):].split()[1])


def mezcla():
    (BUILD / "audio").mkdir(parents=True, exist_ok=True)
    run([sys.executable, "tools/musica.py", "build/audio/musica.f32"])
    n = int(DUR * SR)
    musica = np.fromfile(BUILD / "audio" / "musica.f32", dtype=np.float32).reshape(-1, 2)[:n]
    voz, sr = sf.read(ROOT / "audio" / "locucion.flac", dtype="float32")
    assert sr == SR
    voz = np.pad(voz, (0, max(0, n - len(voz))))[:n]
    voz2 = np.stack([voz, voz], 1)
    # niveles de partida: voz a -16 LUFS y música a -22 LUFS
    voz2 *= 10 ** ((-16 - lufs(voz2)) / 20)
    musica *= 10 ** ((-22 - lufs(musica)) / 20)
    # la música baja ~6 dB mientras habla la voz (ataque rápido, vuelta suave)
    env = np.abs(voz)
    att, rel = np.exp(-1 / (0.03 * SR)), np.exp(-1 / (0.3 * SR))
    e = lfilter([1 - rel], [1, -rel], env)            # seguidor lento
    e = np.maximum(e, lfilter([1 - att], [1, -att], env))
    hablando = np.clip(e / 0.02, 0, 1)
    hablando = lfilter([1 - rel], [1, -rel], hablando)
    gan = 10 ** (-6 * hablando / 20)
    mix = voz2 + musica * gan[:, None]
    raw = BUILD / "audio" / "mezcla.f32"
    mix.astype(np.float32).tofile(raw)
    # dos pasadas: -14 LUFS con el pico real por debajo de -1,5 dB (Instagram, TikTok, YouTube, WhatsApp)
    inp = ["-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", str(raw)]
    meas = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *inp, "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                          cwd=ROOT, capture_output=True, text=True).stderr
    m = json.loads(meas[meas.rindex("{"):meas.rindex("}") + 1])
    flt = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
           f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,aresample={SR}")
    run(["ffmpeg", "-v", "error", "-y", *inp, "-af", flt, "-c:a", "flac", "-sample_fmt", "s32", "build/audio/mezcla.flac"])
    # la música sola, por si se quiere reutilizar
    run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "build/audio/musica.f32",
         "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", str(SR), "-c:a", "flac", "-sample_fmt", "s32", "audio/musica.flac"])


def encode(total):
    OUT.mkdir(exist_ok=True)
    color = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
    vf = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
    final = OUT / f"{NAME}.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", "build/frames/f%05d.png", "-i", "build/audio/mezcla.flac",
         "-frames:v", str(total), "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-maxrate", "16M", "-bufsize", "24M",
         "-profile:v", "high", "-level", "4.2", "-tune", "film", *color, "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
         "-shortest", "-movflags", "+faststart", str(final)])
    run(["ffmpeg", "-v", "error", "-y", "-i", str(final), "-vf", "scale=720:1280:flags=lanczos,format=yuv420p", "-c:v", "libx264",
         "-preset", "slow", "-crf", "23", "-profile:v", "high", *color, "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
         str(OUT / f"{NAME.replace('_1080x1920', '')}_WhatsApp_720p.mp4")])
    Image.open(BUILD / "frames" / f"f{int(48.5 * FPS):05d}.png").convert("RGB").save(OUT / "portada.jpg", quality=92)


if __name__ == "__main__":
    prep()
    if "--fresh" in sys.argv:
        for f in (BUILD / "frames").glob("*.png"):
            f.unlink()
    n = render_frames(int(os.environ.get("WORKERS", "4")))
    mezcla()
    encode(n)
    print("listo:", OUT)
