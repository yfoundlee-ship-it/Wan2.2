#!/usr/bin/env python3
"""
10-second music bed for the flyer - synthesised from scratch with numpy.

96 BPM, 16 beats, Am - F - C - G.  The hits are placed on the same beats as the
on-screen events (title at 1.25 s, date card at 3.75 s, CTA at 5.2 s).

    python make_audio.py                 # -> out/event-flyer.wav  (44.1 kHz stereo)
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SR = 44100
BPM = 96.0
SPB = 60.0 / BPM                      # 0.625 s per beat
DUR = 10.0
N = int(SR * DUR)

rng = np.random.default_rng(11)


# ------------------------------------------------------------------ helpers
def t_axis(n):
    return np.arange(n) / SR


def place(buf, x, at):
    """Add *x* into *buf* starting at time *at* (seconds)."""
    i = int(at * SR)
    if i >= buf.size:
        return
    x = x[: buf.size - i]
    buf[i:i + x.size] += x
    return buf


def env(n, attack, decay, power=2.0):
    t = t_axis(n)
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    d = np.exp(-t / max(decay, 1e-4) * power)
    return a * d


def lp(x, cutoff):
    """one-pole low-pass (cutoff in Hz)"""
    a = np.exp(-2 * np.pi * cutoff / SR)
    out = np.empty_like(x)
    acc = 0.0
    for i in range(x.size):          # loop is fine: only used on short buffers
        acc = (1 - a) * x[i] + a * acc
        out[i] = acc
    return out


def hp(x, cutoff):
    return x - lp(x, cutoff)


def note(freq, dur, detune=0.0):
    n = int(dur * SR)
    t = t_axis(n)
    f = freq * (1 + detune)
    return np.sin(2 * np.pi * f * t), n


# ------------------------------------------------------------------ instruments
def kick(dur=0.42):
    n = int(dur * SR)
    t = t_axis(n)
    f = 118 * np.exp(-t * 26) + 46
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.16, 1.6)
    click = hp(rng.normal(0, 1, n), 1800) * env(n, 0.0005, 0.012)
    return (body * 0.95 + click * 0.22).astype(np.float32)


def clap(dur=0.30):
    n = int(dur * SR)
    t = t_axis(n)
    noise = hp(rng.normal(0, 1, n), 900)
    sq = np.zeros(n, np.float32)
    for k, off in enumerate((0.0, 0.011, 0.023, 0.036)):
        i = int(off * SR)
        seg = noise[i:] * env(n - i, 0.0006, 0.030 if k < 3 else 0.12)
        sq[i:] += seg * (0.8 if k < 3 else 1.0)
    return (sq * 0.55).astype(np.float32)


def hat(dur=0.085, open_=False):
    n = int(dur * SR)
    x = hp(rng.normal(0, 1, n), 6500)
    return (x * env(n, 0.0004, 0.020 if not open_ else 0.075)).astype(np.float32)


def shaker(dur=0.07):
    n = int(dur * SR)
    x = hp(rng.normal(0, 1, n), 4200)
    return (x * env(n, 0.004, 0.022)).astype(np.float32)


def bass(freq, dur, cutoff=340):
    n = int(dur * SR)
    t = t_axis(n)
    saw = 2 * (t * freq - np.floor(0.5 + t * freq))
    x = lp(saw, cutoff) * env(n, 0.004, 0.16, 1.2)
    sub = np.sin(2 * np.pi * freq / 2 * t) * env(n, 0.004, 0.20, 1.4) * 0.7
    return (x * 0.5 + sub).astype(np.float32)


def pluck(freq, dur=0.36):
    n = int(dur * SR)
    t = t_axis(n)
    x = np.zeros(n, np.float32)
    for h, a in ((1, 1.0), (2, 0.34), (3, 0.16), (4, 0.07)):
        x += a * np.sin(2 * np.pi * freq * h * t)
    x = lp(x, 5200) * env(n, 0.002, 0.085, 1.1)
    return (x * 0.3).astype(np.float32)


def pad(freqs, dur, gain=0.16):
    n = int(dur * SR)
    t = t_axis(n)
    x = np.zeros(n, np.float32)
    for f in freqs:
        for det in (-0.0015, 0.0, 0.0015):
            x += np.sin(2 * np.pi * f * (1 + det) * t + rng.uniform(0, 6.28))
    x = lp(x / (len(freqs) * 3), 1500) * env(n, 0.35, 1.6, 1.0)
    return (x * gain).astype(np.float32)


def riser(dur, lo=300, hi=7000):
    """noise whose high end opens up - used as a transition sweep"""
    n = int(dur * SR)
    t = t_axis(n)
    noise = rng.normal(0, 1, n)
    out = np.zeros(n, np.float32)
    blocks = 24
    for b in range(blocks):
        i0, i1 = int(n * b / blocks), int(n * (b + 1) / blocks)
        cut = lo + (hi - lo) * (b / (blocks - 1)) ** 2
        out[i0:i1] = hp(noise[i0:i1], cut)
    return (out * (t / DUR * 0 + (t / max(dur, 1e-4)) ** 1.8) * 0.22).astype(np.float32)


def crash(dur=1.6):
    n = int(dur * SR)
    x = hp(rng.normal(0, 1, n), 3000)
    return (x * env(n, 0.002, 0.42, 1.1) * 0.28).astype(np.float32)


def whoosh(dur=0.45):
    n = int(dur * SR)
    t = t_axis(n)
    x = lp(rng.normal(0, 1, n), 2200)
    return (x * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5 * 0.35).astype(np.float32)


def reverb(x, decay=0.35, mix=0.22, bright=0.45):
    """cheap FFT convolution with a noisy exponential tail"""
    ir_n = int(SR * 0.9)
    ir = rng.normal(0, 1, ir_n) * np.exp(-np.arange(ir_n) / (SR * decay))
    ir = lp(ir, 6000 * bright + 1500)
    ir /= np.abs(ir).sum() + 1e-9
    n = int(2 ** np.ceil(np.log2(x.size + ir_n)))
    y = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n))[: x.size]
    return (x * (1 - mix) + y * mix).astype(np.float32)


# ------------------------------------------------------------------ the track
def make_track() -> np.ndarray:
    dry = np.zeros(N, np.float32)
    wet_src = np.zeros(N, np.float32)

    def beat(b):
        return b * SPB

    # --- chords ------------------------------------------------------------
    A3, C4, E4, F3, G3, A4, C5, E5 = 220.0, 261.6, 329.6, 174.6, 196.0, 440.0, 523.3, 659.3
    prog = [
        ([A3, C4, E4], A3),
        ([F3 * 2, A3 * 1.5, C4], F3),
        ([C4, E4, G3 * 2], C4 / 2),
        ([G3 * 2, A3 * 1.5, E4 / 2], G3),
    ]
    for i, (chord, _) in enumerate(prog):
        place(dry, pad(chord, SPB * 4, gain=0.13), beat(i * 4))

    # --- riser + whooshes ---------------------------------------------------
    place(dry, riser(1.15), 0.10)
    place(wet_src, whoosh(0.5), 1.10)
    place(wet_src, whoosh(0.45), 4.95)
    place(wet_src, whoosh(0.6), 1.12)

    # --- groove -------------------------------------------------------------
    KICKS = [1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 8.75 + 0.3125]
    for at in KICKS:
        place(dry, kick() * 1.0, at)
        place(dry, kick() * 0.85, at + 0.3125) if at > 3.0 else None
    place(dry, kick(0.5) * 1.05, 5.20)          # lands with the CTA button

    for at in (1.875, 3.125, 4.375, 5.625, 6.875, 8.125):
        place(dry, clap(), at)
    place(dry, crash(), 3.72)                   # date card
    place(dry, crash(1.3) * 0.8, 5.20)          # CTA

    # hats on 8ths, shakers on 16ths once the ribbon appears
    b = beat(2)
    while b < DUR - 0.1:
        place(dry, hat(open_=(abs((b / SPB) % 2 - 1) < 0.05)) * 0.5, b)
        b += SPB / 2
    b = 4.55
    while b < DUR - 0.05:
        place(dry, shaker() * 0.42, b)
        b += SPB / 4

    # --- bass line ----------------------------------------------------------
    roots = [(1.25, A3 / 2), (2.5, A3 / 2), (3.0, 174.6), (3.75, F3), (4.375, F3),
             (5.0, C4 / 2), (5.625, C4 / 2), (6.25, G3), (6.875, G3),
             (7.5, A3 / 2), (8.125, A3 / 2), (8.75, A3 / 2)]
    for at, f in roots:
        place(dry, bass(f, 0.34), at)

    # --- pluck arp (starts with the title) ---------------------------------
    arp = [A4, C5, E5, C5, A4, E5, C5, A4]
    for i in range(32):
        at = 1.25 + i * SPB / 2
        if at > DUR - 0.3:
            break
        f = arp[i % len(arp)] * (1.0 if i % 16 < 8 else 1.0)
        gain = 0.55 if i % 4 == 0 else 0.34
        v = pluck(f) * gain
        place(dry, v * (1.0 if i % 2 == 0 else 0.78), at)

    # --- tail ---------------------------------------------------------------
    out = reverb(wet_src + dry, decay=0.30, mix=0.20)
    # fade the very end so the loop is clean
    fade = int(SR * 0.25)
    out[-fade:] *= np.linspace(1, 0, fade) ** 1.5
    out[: int(SR * 0.02)] *= np.linspace(0, 1, int(SR * 0.02))

    # soft-knee limiter + normalise to -1.2 dBFS
    out = np.tanh(out * 1.35) * 0.92
    peak = np.abs(out).max()
    out = out / peak * 0.87
    return out.astype(np.float32)


def write_wav(path: Path, mono: np.ndarray):
    """tiny haas-style widening: a 12 ms delayed, slightly quieter right channel"""
    d = int(SR * 0.012)
    left = mono
    right = np.concatenate([np.zeros(d, np.float32), mono[:-d]]) * 0.88 + mono * 0.22
    stereo = np.stack([left, right], axis=1)
    pcm = np.clip(stereo, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser(description="Synthesise the 10s flyer music bed.")
    ap.add_argument("--out", default=str(ROOT / "event-flyer.wav"))
    args = ap.parse_args()
    track = make_track()
    p = Path(args.out)
    write_wav(p, track)
    print(f"audio -> {p}  ({DUR:g}s, {p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
