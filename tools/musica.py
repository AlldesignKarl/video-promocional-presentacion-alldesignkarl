"""Música original del vídeo (sintetizada desde cero, sin derechos de terceros).

Sol mayor, 92 BPM (compás de 2,61 s). Guitarra punteada (Karplus-Strong) y piano suaves al empezar,
después bajo, shaker, chasquidos y un pad cálido; en la parte de mayoristas y la tienda online la base
se llena un poco más, y en el cierre todo resuelve en Sol con una campanita. Una campanita y un
soplo suave marcan cada cambio de escena (los tiempos salen de src/tiempos.js).

Uso: python3 tools/musica.py build/audio/musica.f32   (float32 estéreo intercalado, 48 kHz)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

ROOT = Path(__file__).resolve().parent.parent
SR = 48000
DUR = 50.0
BEAT = 60 / 92
BAR = 4 * BEAT
N = int(SR * DUR)
rng = np.random.default_rng(2026)

# acordes por compás: (bajo, voces del piano, notas del arpegio)
G = (43, [55, 59, 62, 67], [55, 62, 67, 71, 74, 71, 67, 62])
D = (42, [54, 57, 62, 66], [54, 62, 66, 69, 74, 69, 66, 62])   # Re/Fa#
EM = (40, [52, 55, 59, 64], [52, 59, 64, 67, 71, 67, 64, 59])
C = (36, [52, 55, 60, 64], [48, 55, 60, 64, 67, 64, 60, 55])
D7 = (38, [54, 57, 60, 62], [50, 57, 62, 66, 69, 66, 62, 57])
COMPASES = [G, D, EM, C] * 4 + [D7, G]   # el compás 17 (44,35 s) resuelve en Sol justo con el cierre


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def place(buf, sig, t0, gain=1.0, pan=0.0):
    i = int(round(t0 * SR))
    if sig.ndim == 1:
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        sig = np.stack([sig * l, sig * r], 1) * 1.4142
    if i < 0:
        sig, i = sig[-i:], 0
    j = min(len(buf), i + len(sig))
    if j > i:
        buf[i:j] += sig[: j - i] * gain


def lp(x, fc, order=2):
    return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x, axis=0)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x, axis=0)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x, axis=0)


def pluck(freq, dur=2.4, bright=0.5, decay=0.996):
    """Cuerda punteada (Karplus-Strong), calculada periodo a periodo."""
    n = int(dur * SR)
    p = max(2, int(round(SR / freq)))
    exc = rng.uniform(-1, 1, p)
    exc = lp(exc, 1200 + 5000 * bright, 1)
    out = np.zeros(n + p)
    out[:p] = exc
    k = p
    while k < n:
        prev = out[k - p:k]
        blk = decay * 0.5 * (prev + np.concatenate([[out[k - p - 1] if k - p - 1 >= 0 else 0], prev[:-1]]))
        m = min(p, n - k)
        out[k:k + m] = blk[:m]
        k += p
    y = out[:n]
    y = y - lp(y, 90, 1)                                 # fuera el retumbe
    y += 0.25 * bp(y, 180, 420, 1)                      # algo de caja de guitarra
    fade = np.minimum(1, (n - np.arange(n)) / (0.08 * SR))
    return y * fade


def piano(freq, dur=3.2, vel=0.6):
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    for h in range(1, 9):
        fh = freq * h * np.sqrt(1 + 0.0004 * h * h)       # ligera inarmonicidad
        if fh > 9000:
            break
        y += (vel ** (h * 0.5)) / h ** 1.3 * np.sin(2 * np.pi * fh * t + rng.uniform(0, 6.28)) * np.exp(-t * (0.9 + 0.55 * h))
    y *= np.minimum(1, t / 0.004)
    ham = lp(rng.standard_normal(len(t)), 2500) * np.exp(-t * 90) * 0.04
    rel = np.minimum(1, (len(t) - np.arange(len(t))) / (0.3 * SR))
    return (y + ham) * rel


def pad(notes, dur):
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros((len(t), 2))
    for m in notes:
        for ch, det in ((0, -0.07), (1, 0.07)):
            f = mtof(m + det)
            for h in range(1, 7):
                y[:, ch] += np.sin(2 * np.pi * f * h * t + rng.uniform(0, 6.28)) / h ** 1.6
    y = lp(y, 1400, 2)
    env = np.minimum(1, t / 0.9) * np.minimum(1, (dur - t) / 0.9)
    return y * env[:, None] / len(notes)


def bass(freq, dur):
    t = np.arange(int(dur * SR)) / SR
    y = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(6 * np.pi * freq * t)
    y = np.tanh(1.4 * y) / 1.4
    return y * np.minimum(1, t / 0.01) * np.exp(-t * 0.8) * np.minimum(1, (dur - t) / 0.08)


def kick():
    t = np.arange(int(0.35 * SR)) / SR
    f = 46 + 70 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 11)


def shaker(vel):
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    return bp(rng.standard_normal(n), 5000, 11000) * (t / 0.012 * np.exp(1 - t / 0.012)) * 0.35 * vel


def snap():
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    return bp(rng.standard_normal(n), 1200, 3600) * np.exp(-t * 38) * 0.5


def bell(freq, dur=2.6):
    t = np.arange(int(dur * SR)) / SR
    mod = 2.2 * np.exp(-t * 3) * np.sin(2 * np.pi * freq * 3.5 * t)
    return np.sin(2 * np.pi * freq * t + mod) * np.exp(-t * 2.2) * np.minimum(1, t / 0.002)


def soplo(dur=0.9):
    """Soplo de aire que crece hasta el cambio de escena."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal((n, 2))
    env = (t / dur) ** 2.2 * np.minimum(1, (dur - t) / 0.06)
    return bp(x, 500, 5000) * env[:, None] * 0.12


def reverb_ir(seg=2.4):
    n = int(seg * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-t * 3.0)[:, None]
    ir = lp(ir, 5200)
    ir[: int(0.012 * SR)] = 0
    return ir / np.sqrt((ir ** 2).sum(0))


def escenas():
    s = (ROOT / "src" / "tiempos.js").read_text(encoding="utf-8")
    return [f["a"] for f in json.loads(s[s.index("["): s.rindex("]") + 1])][1:]


def main(out):
    dry = np.zeros((N + SR, 2))
    wet = np.zeros_like(dry)
    for b, (bn, voces, arp) in enumerate(COMPASES):
        t0 = b * BAR
        final = b == len(COMPASES) - 1
        seccion = 0 if b < 2 else 1 if b < 8 else 2 if b < 12 else 3
        # piano: acorde al principio del compás (y un toque suave a mitad en las partes llenas)
        for k, m in enumerate(voces):
            pv = piano(mtof(m), 5.5 if final else 3.4, 0.55)
            place(dry, pv, t0 + k * 0.012, 0.075, -0.25 + 0.15 * k)
            place(wet, pv, t0 + k * 0.012, 0.07)
            if seccion >= 2 and not final:
                place(dry, piano(mtof(m + 12), 1.6, 0.4), t0 + 2 * BEAT + 0.01 * k, 0.035, 0.3)
        # guitarra: arpegio en corcheas con un poco de vida en el tiempo y la fuerza
        if not final:
            for i, m in enumerate(arp):
                tt = t0 + i * BEAT / 2 + rng.normal(0, 0.006) + (0.012 if i % 2 else 0)
                v = (0.8 if i % 2 == 0 else 0.55) * rng.uniform(0.85, 1.05)
                pl = pluck(mtof(m), 2.2, 0.35 + 0.25 * (i % 2 == 0))
                place(dry, pl, tt, 0.22 * v, 0.35 if i % 2 else -0.15)
                place(wet, pl, tt, 0.08 * v)
        else:
            for i, m in enumerate([55, 59, 62, 67, 71, 74]):
                pl = pluck(mtof(m), 4.5, 0.4, 0.9975)
                place(dry, pl, t0 + i * 0.045, 0.16, -0.3 + 0.12 * i)
                place(wet, pl, t0 + i * 0.045, 0.1)
            # y el arpegio sigue, cada vez más suave, mientras se ve la ficha de contacto
            for i, m in enumerate(G[2] * 2):
                if i < 2:
                    continue
                tt = t0 + i * BEAT / 2 + rng.normal(0, 0.006)
                v = 0.75 * (1 - i / 18) * rng.uniform(0.85, 1.05)
                pl = pluck(mtof(m + (12 if i >= 8 else 0)), 2.4, 0.35)
                place(dry, pl, tt, 0.34 * v, 0.35 if i % 2 else -0.15)
                place(wet, pl, tt, 0.16 * v)
        if seccion >= 1 and not final:
            place(dry, bass(mtof(bn), BAR * 0.95), t0, 0.12)
            place(dry, bass(mtof(bn), BEAT * 1.4), t0 + 2.5 * BEAT, 0.07)
            pd = pad([m + 12 for m in voces[1:]], BAR + 0.9)
            place(dry, pd, t0 - 0.2, 0.05 if seccion == 1 else 0.075)
            place(wet, pd, t0 - 0.2, 0.05)
            for s in range(16):
                vel = (1.0 if s % 4 == 2 else 0.55 if s % 2 else 0.35) * rng.uniform(0.8, 1.1)
                place(dry, shaker(vel), t0 + s * BEAT / 4 + (0.018 if s % 2 else 0), 0.55 if seccion == 1 else 0.7, 0.4)
            if seccion >= 2:
                for k in (1, 3):
                    place(dry, snap(), t0 + k * BEAT, 0.16, -0.1)
                    place(wet, snap(), t0 + k * BEAT, 0.12)
            if seccion >= 1:
                for k in ((0, 2) if seccion < 3 else (0, 1.5, 2, 3.5)):
                    place(dry, kick(), t0 + k * BEAT, 0.2 if seccion == 1 else 0.26)
        if final:
            place(dry, bass(mtof(43), 5.0), t0, 0.14)
            pd = pad([67, 71, 74, 79], 5.6)
            place(dry, pd, t0, 0.11)
            place(wet, pd, t0, 0.06)
            for i, m in enumerate([79, 83, 86, 91]):
                bl = bell(mtof(m), 3.0)
                place(dry, bl, t0 + 0.12 + i * 0.09, 0.05, 0.2 - 0.1 * i)
                place(wet, bl, t0 + 0.12 + i * 0.09, 0.08)
    # campanita y soplo en cada cambio de escena (la imagen cambia 0,35 s antes de la frase)
    notas = [74, 79, 76, 83, 78, 81, 86]
    for i, ta in enumerate(escenas()[:-1]):
        tc = ta - 0.35
        place(dry, soplo(), tc - 0.9, 1.0)
        bl = bell(mtof(notas[i % len(notas)] + 12), 2.0)
        place(dry, bl, tc, 0.035, 0.45 if i % 2 else -0.45)
        place(wet, bl, tc, 0.06)
    mix = dry + fftconvolve(wet, reverb_ir(), axes=0)[: len(wet)] * 0.9
    mix = hp(mix, 35)
    mix = mix[:N]
    # entrada y final suaves
    t = np.arange(N) / SR
    mix *= np.minimum(1, t / 0.05)[:, None] * np.clip((DUR - t) / 1.3, 0, 1)[:, None] ** 1.5
    mix /= np.abs(mix).max() / 0.8
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    mix.astype(np.float32).tofile(out)
    print("música:", out, f"{len(mix) / SR:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "build" / "audio" / "musica.f32"))
