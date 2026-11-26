#!/usr/bin/env python3
"""Tablio meme-reel audio: original phonk-style bed + meme sound effects, all synthesized
from scratch (no samples, no ElevenLabs, nothing copyrighted).
Used by tablio_meme.py. Standalone: python3 tablio_meme_audio.py spec.json out.wav
"""
import json, math, sys
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.io import wavfile

SR = 48000
rng = np.random.default_rng(7)


def hz(m): return 440 * 2 ** ((m - 69) / 12)
def env(n, a=0.002, d=0.2):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(a, 1e-5)) * np.exp(-t / d)
def bp(x, lo, hi, order=2): return sosfilt(butter(order, [lo, hi], 'band', fs=SR, output='sos'), x)
def hp(x, f, order=2): return sosfilt(butter(order, f, 'high', fs=SR, output='sos'), x)
def lp(x, f, order=2): return sosfilt(butter(order, f, 'low', fs=SR, output='sos'), x)
def norm(x, peak=1.0): return x / (np.abs(x).max() + 1e-9) * peak
def fade(x, a=0.004, r=0.03):
    x = x.copy(); na, nr = int(a * SR), int(r * SR)
    if na: x[:na] *= np.linspace(0, 1, na)
    if nr: x[-nr:] *= np.linspace(1, 0, nr)
    return x
def keyframes(n, pts):
    """pts = [(t_seconds, value), ...] -> per-sample array (linear)."""
    t = np.arange(n) / SR
    return np.interp(t, [p[0] for p in pts], [p[1] for p in pts])


# ------------------------------------------------------------------ voice-like synthesis
# vowel formants (F, bandwidth, gain)
V = {
    'a': [(800, 130, 1.0), (1200, 140, 0.55), (2800, 220, 0.22), (3600, 300, 0.10)],
    'i': [(320, 90, 1.0), (2300, 160, 0.45), (3000, 220, 0.30), (3700, 300, 0.12)],
    'u': [(420, 100, 1.0), (850, 120, 0.45), (2600, 220, 0.12), (3400, 300, 0.06)],
    'e': [(550, 110, 1.0), (1800, 150, 0.5), (2600, 220, 0.25), (3500, 300, 0.10)],
}

def fenv(freq, vowel):
    return sum(g / (1 + ((freq - F) / (bw / 2)) ** 2) for F, bw, g in V[vowel])

def vocal(f0, vowels, weights, nh=48, tilt=0.8, breath=0.08):
    """Additive harmonic voice: f0 per sample, vowels list, weights list of per-sample arrays."""
    n = len(f0)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros(n)
    for k in range(1, nh + 1):
        fk = k * f0
        a = sum(w * fenv(fk, v) for v, w in zip(vowels, weights))
        a = a * (fk < SR / 2 - 2000) / k ** tilt
        out += a * np.sin(k * ph + rng.uniform(0, 6.28))
    if breath:
        nz = rng.uniform(-1, 1, n)
        br = sum(w * (bp(nz, max(60, V[v][0][0] - 200), V[v][0][0] + 400) + 0.6 * bp(nz, V[v][1][0] - 300, V[v][1][0] + 500)) for v, w in zip(vowels, weights))
        out = norm(out) + breath * norm(br)
    return out


# ------------------------------------------------------------------ sound effects
def sfx_boom():
    n = int(SR * 1.7); t = np.arange(n) / SR
    f = 32 + 95 * np.exp(-t * 9)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.0)
    s += 0.35 * np.sin(2 * 2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5)
    s = np.tanh(s * 2.2)
    click = lp(rng.uniform(-1, 1, n), 1800) * env(n, 0.0005, 0.012) * 1.5
    rumble = lp(rng.uniform(-1, 1, n), 160) * np.exp(-t * 2.5) * 2.0
    return fade(norm(s + click + rumble), 0.0005, 0.08)

def sfx_scratch():
    # source: a short buzzy chord, read back and forth like a record under a hand
    src_n = SR * 2; t = np.arange(src_n) / SR
    src = sum(np.sign(np.sin(2 * np.pi * f * t)) * a for f, a in ((196, 1), (233, .7), (294, .6), (392, .4)))
    src = bp(src, 300, 4200) + 0.5 * bp(rng.uniform(-1, 1, src_n), 1500, 6000)
    dur = 0.62; n = int(SR * dur); tt = np.arange(n) / SR
    rate = 2.8 * np.sin(2 * np.pi * 3.4 * tt - 0.4) + 0.6 * np.sin(2 * np.pi * 9 * tt)
    pos = np.clip(SR * 0.6 + np.cumsum(rate), 0, src_n - 1)
    out = np.interp(pos, np.arange(src_n), src)
    out *= np.clip(np.abs(rate) / 1.2, 0, 1) ** 0.7          # quiet when the record stops
    gate = (np.sin(2 * np.pi * 6.8 * tt) > -0.6).astype(float)  # fader cuts
    gate = lp(gate, 200)
    return fade(norm(hp(out * gate, 180)), 0.002, 0.05)

def sfx_ding():
    n = int(SR * 1.6); t = np.arange(n) / SR
    out = np.zeros(n)
    for f0, t0, g in ((1318.5, 0.0, 1.0), (1975.5, 0.11, 0.8)):
        i = int(t0 * SR); tt = t[: n - i]
        for mul, a, d in ((1, 1, 0.9), (2.0, .35, .5), (3.01, .18, .3), (4.2, .1, .2), (5.43, .06, .12)):
            out[i:] += g * a * np.sin(2 * np.pi * f0 * mul * tt) * np.exp(-tt / d) * np.minimum(1, tt / 0.002)
    return fade(norm(out), 0.0005, 0.1)

def sfx_gasp():
    n = int(SR * 0.55); t = np.arange(n) / SR
    nz = rng.uniform(-1, 1, n)
    s = bp(nz, 700, 1500) * 1.0 + bp(nz, 1900, 3200) * 0.7 + bp(nz, 3500, 6000) * 0.25
    f0 = keyframes(n, [(0, 230), (0.2, 300), (0.55, 260)])
    voiced = vocal(f0, ['a'], [np.ones(n)], nh=30, breath=0) * 0.18
    e = keyframes(n, [(0, 0), (0.035, 1.0), (0.14, 0.85), (0.42, 0.25), (0.55, 0)])
    return fade(norm((norm(s) + voiced) * e), 0.002, 0.04)

def sfx_scream():
    # cartoon cat yowl: "mrrEEEAAOWW" (fits Tabby)
    dur = 1.15; n = int(SR * dur); t = np.arange(n) / SR
    f0 = keyframes(n, [(0, 480), (0.16, 1020), (0.75, 960), (1.15, 640)])
    f0 = f0 * (1 + 0.035 * np.sin(2 * np.pi * 6.3 * t) * np.clip((t - 0.2) / 0.2, 0, 1))
    f0 = f0 * (1 + 0.012 * lp(rng.normal(0, 1, n), 30) * 8)
    wi = keyframes(n, [(0, 1), (0.25, 0.1), (0.4, 0), (1.15, 0)])
    wu = keyframes(n, [(0, 0), (0.7, 0), (1.15, 0.9)])
    wa = np.clip(1 - wi - wu, 0, 1)
    v = vocal(f0, ['i', 'a', 'u'], [wi, wa, wu], nh=24, tilt=0.55, breath=0.22)
    v = v * (1 + 0.25 * np.sin(2 * np.pi * 38 * t))           # rough growl
    v = np.tanh(norm(v) * 2.6)
    e = keyframes(n, [(0, 0), (0.05, 0.7), (0.18, 1), (0.9, 0.9), (1.15, 0)])
    return fade(norm(hp(v * e, 250)), 0.003, 0.08)

def sfx_laugh():
    out = np.zeros(int(SR * 1.35))
    for voice_i, (base, off, g) in enumerate(((290, 0.0, 1.0), (215, 0.035, 0.7), (360, 0.07, 0.45))):
        for s in range(6):
            t0 = off + s * 0.2; sd = 0.15
            n = int(SR * sd); t = np.arange(n) / SR
            f0 = np.full(n, base * (1 - 0.035 * s)) * (1 - 0.12 * t / sd) * (1 + 0.02 * voice_i)
            v = vocal(f0, ['a'], [np.ones(n)], nh=30, breath=0.12)
            h = bp(rng.uniform(-1, 1, n), 900, 3500) * np.exp(-t * 60) * 0.6     # the "h"
            e = np.minimum(1, t / 0.012) * np.exp(-t * 9)
            seg = (norm(v) * e + h) * g * (1 - 0.08 * s)
            i = int(t0 * SR); out[i:i + n] += seg[: len(out) - i]
    return fade(norm(out), 0.002, 0.08)

def sfx_sad():
    # "wah wah wah waaah", additive brass with a wah on every note
    notes = [(62, 0.36), (61, 0.36), (60, 0.36), (59, 1.25)]
    total = sum(d for _, d in notes); out = np.zeros(int(SR * total)); t0 = 0
    for i, (m, d) in enumerate(notes):
        n = int(SR * d); t = np.arange(n) / SR
        f0 = np.full(n, hz(m - 12))
        if i == len(notes) - 1: f0 *= 1 + 0.018 * np.sin(2 * np.pi * 5.2 * t) * np.clip((t - 0.25) / 0.3, 0, 1)
        bright = keyframes(n, [(0, 0.25), (0.09, 0.82), (d * 0.6, 0.62), (d, 0.3)])
        ph = 2 * np.pi * np.cumsum(f0) / SR; s = np.zeros(n)
        for k in range(1, 18):
            s += bright ** (k - 1) / k * np.sin(k * ph)
        e = keyframes(n, [(0, 0), (0.03, 1), (d - 0.06, 0.85), (d, 0)])
        j = int(t0 * SR); out[j:j + n] += s * e; t0 += d
    return fade(norm(out), 0.003, 0.08)

def sfx_whoosh():
    n = int(SR * 0.45); t = np.arange(n) / SR
    nz = rng.uniform(-1, 1, n)
    lo = bp(nz, 400, 1400); hi = bp(nz, 1600, 5000)
    w = keyframes(n, [(0, 0), (0.45, 1)])
    e = np.sin(np.pi * np.clip(t / 0.45, 0, 1)) ** 2
    return fade(norm((lo * (1 - w) + hi * w) * e), 0.002, 0.03)

SFX = {'boom': sfx_boom, 'scratch': sfx_scratch, 'ding': sfx_ding, 'gasp': sfx_gasp,
       'scream': sfx_scream, 'laugh': sfx_laugh, 'sad': sfx_sad, 'whoosh': sfx_whoosh}
SFX_GAIN = {'boom': 0.95, 'scratch': 0.65, 'ding': 0.55, 'gasp': 0.6, 'scream': 0.6, 'laugh': 0.55, 'sad': 0.6, 'whoosh': 0.35}


# ------------------------------------------------------------------ music bed
BPM = 150; BEAT = 60 / BPM; STEP = BEAT / 4
RIFF = [77, None, 80, 77, None, 75, None, 72, 77, None, 80, 82, None, 80, 77, None]
RIFF2 = [77, None, 80, 77, None, 75, None, 72, 84, None, 82, 80, None, 77, 75, None]

def kick():
    k = int(SR * 0.35); t = np.arange(k) / SR
    f = 48 + 110 * np.exp(-t * 30)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(k, 0.001, 0.16)
    s[:200] += rng.uniform(-1, 1, 200) * np.linspace(0.6, 0, 200)
    return np.tanh(s * 1.6)
def clap():
    k = int(SR * 0.25); nz = rng.uniform(-1, 1, k); e = np.zeros(k)
    for o in (0, 0.011, 0.022):
        i = int(o * SR); e[i:] += env(k - i, 0.0005, 0.012 if o < 0.02 else 0.09)
    return bp(nz, 900, 5200) * e * 1.6
def hat(open_=False):
    k = int(SR * (0.18 if open_ else 0.05))
    return hp(rng.uniform(-1, 1, k), 7500) * env(k, 0.0005, 0.06 if open_ else 0.014)
def cowbell(m, d=0.22):
    k = int(SR * d); t = np.arange(k) / SR; f = hz(m)
    sq = np.sign(np.sin(2 * np.pi * f * t)) + 0.8 * np.sign(np.sin(2 * np.pi * f * 1.483 * t))
    return bp(sq, f * 0.8, f * 4.5) * env(k, 0.001, 0.075) * 0.55
def bass808(m, d, glide_from=None):
    k = int(SR * d); t = np.arange(k) / SR; f = np.full(k, hz(m))
    if glide_from is not None:
        g = int(SR * 0.06); f[:g] = np.linspace(hz(glide_from), hz(m), g)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, t / 0.003) * np.exp(-t / (d * 0.9))
    rel = int(SR * 0.02); s[-rel:] *= np.linspace(1, 0, rel)
    return np.tanh(s * 2.2) * 0.8
def pad(notes, d):
    k = int(SR * d); t = np.arange(k) / SR; s = np.zeros(k)
    for m in notes:
        for det in (-0.12, 0.0, 0.12):
            s += 2 * ((t * hz(m + det)) % 1) - 1
    return lp(s / (len(notes) * 3), 1800) * np.minimum(1, t / 0.04) * np.exp(-t / (d * 0.6))
def bell(m, d=1.6):
    k = int(SR * d); t = np.arange(k) / SR; f = hz(m)
    s = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 6) + 0.2 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t * 12)
    return s * env(k, 0.001, 0.5) * 0.35

class Bus:
    def __init__(self, total):
        self.N = int(SR * (total + 2.0)); self.L = np.zeros(self.N); self.R = np.zeros(self.N)
    def add(self, sig, t, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i >= self.N or i + len(sig) <= 0: return
        if i < 0: sig = sig[-i:]; i = 0
        j = min(self.N, i + len(sig))
        self.L[i:j] += sig[:j - i] * gain * (1 - max(0, pan)); self.R[i:j] += sig[:j - i] * gain * (1 + min(0, pan))

def music_bed(total, drop_t, end_t):
    """150 BPM F-minor phonk loop. Grid anchored so the drop lands exactly on drop_t
    (the first reaction shot); end card at end_t gets a final hit, held chord and bells."""
    main = Bus(total)
    kmin = -int(math.floor(drop_t / STEP + 1e-6)); kmax = int(math.ceil((end_t - drop_t) / STEP))
    for k in range(kmin, kmax):
        t = drop_t + k * STEP
        if t < -1e-6 or t >= end_t - 1e-3: continue
        s, bar, intro = k % 16, k // 16, k < 0
        if intro and k >= -4: continue                    # one-beat stop before the drop
        m = (RIFF if bar % 2 == 0 else RIFF2)[s]
        if m is not None:
            cb = cowbell(m)
            if intro: cb = lp(cb, 2200) * 0.85
            main.add(cb, t, 0.9, pan=0.25 if s % 2 else -0.25)
        if s % 2 == 0 or (not intro and s in (7, 15)):
            main.add(hat(), t, 0.35 if s % 4 == 0 else 0.22, pan=0.3)
        if s in (4, 12): main.add(clap(), t, 0.55)
        if not intro:
            if s in (0, 6, 10): main.add(kick(), t, 0.95)
            line = {0: (41, 0.75, None), 6: (41, 0.5, None), 10: ((44 if bar % 2 == 0 else 46), 0.45, 41), 13: (39, 0.45, 44)}
            if s in line:
                mm, d, g = line[s]; main.add(bass808(mm, d, g), t, 0.75)
            if s == 14: main.add(hat(True), t, 0.25, pan=-0.3)
        elif s == 0 and t > 0.05:
            main.add(kick(), t, 0.6)
    # cut the loop at the end card
    e = int(end_t * SR); f = int(0.03 * SR)
    for ch in (main.L, main.R):
        ch[e:e + f] *= np.linspace(1, 0, f); ch[e + f:] = 0
    cr = int(SR * 1.2)
    main.add(hp(rng.uniform(-1, 1, cr), 4000) * env(cr, 0.001, 0.35), drop_t, 0.35)
    main.add(kick(), end_t, 1.0); main.add(bass808(41, 2.4), end_t, 0.8)
    main.add(pad([65, 68, 72, 77], 2.8), end_t, 0.45)
    main.add(hp(rng.uniform(-1, 1, cr), 4000) * env(cr, 0.001, 0.5), end_t, 0.3)
    for i, m in enumerate([84, 80, 89]):
        main.add(bell(m), end_t + 0.35 + i * 0.3, 0.8, pan=(-0.2, 0.2, 0)[i])
    return main


def timeline(spec):
    t, starts = 0.0, []
    for sh in spec['shots']:
        starts.append(t); t += sh['dur']
    return starts, t, t + spec.get('end_dur', 2.6)

def build_mix(spec, out_wav):
    starts, end_t, total = timeline(spec)
    react = [s for s, sh in zip(starts, spec['shots']) if sh['type'] == 'reaction']
    drop_t = react[0] if react else min(end_t * 0.35, 3.2)
    bed = spec.get('bed', 'beat')
    if bed == 'none':
        bus = Bus(total)
    else:
        bus = music_bed(total, drop_t, end_t)
        bus.L *= 0.55; bus.R *= 0.55
        if bed == 'laugh':
            bus.add(sfx_laugh(), 0.2, 0.25)
    cache = {}
    def put(name, t, g=1.0):
        if name not in SFX: return
        if name not in cache: cache[name] = SFX[name]()
        bus.add(cache[name], t, SFX_GAIN[name] * g)
    for s, sh in zip(starts, spec['shots']):
        if s > 0: put('whoosh', s - 0.12, 0.8)
        if sh.get('sfx'): put(sh['sfx'], s + 0.02)
    put('ding', end_t + 0.25)
    n = int(SR * total)
    mix = np.stack([bus.L, bus.R], 1)[:n]
    fd = int(SR * 0.3); mix[-fd:] *= np.linspace(1, 0, fd)[:, None]
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.89
    wavfile.write(out_wav, SR, (mix * 32767).astype(np.int16))
    return total

if __name__ == '__main__':
    if sys.argv[1] == '--sfx':           # write every sound effect for inspection
        import os
        os.makedirs(sys.argv[2], exist_ok=True)
        for k, f in SFX.items():
            wavfile.write(f'{sys.argv[2]}/{k}.wav', SR, (f() * 0.9 * 32767).astype(np.int16))
    else:
        print('total', build_mix(json.load(open(sys.argv[1])), sys.argv[2]))
