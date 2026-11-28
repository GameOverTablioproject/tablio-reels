"""Compose the needle reel: rendered frames + frame counter + freeze + text lines + Tablio end card + sound.
python3 compose.py FRAMES_DIR OUT.mp4 [--vo vo.mp3] [--timing timing.json] [--stills DIR]
timing.json: {"vo_start": 7.4, "lines": [[start, end], ...4 lines, relative to video start], "end": total}
"""
import sys, os, json, math, subprocess, tempfile, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
sys.path.insert(0, '/home/claude/tablio-reels/scripts')
import tablio_meme as tm
import tablio_meme_audio as au

W, H, FPS = 1080, 1920, 30
N = 210
EMERALD = (16, 185, 129); CORAL = (255, 77, 77)
LINES = [
    ("You just watched a needle fall for seven seconds.", "seven seconds."),
    ("Give us a few more, and stop wasting yours on tables.", "yours"),
    ("Tablio turns tables in PDFs, screenshots and photos into clean Excel sheets.", "clean Excel sheets."),
]
# default timing (estimated read) until the real VO arrives
T = {"freeze_hold": 0.45, "lines": [[7.45, 10.0], [10.0, 12.6], [12.6, 15.6]], "end_card": 15.6, "end": 17.8, "vo_start": 7.45}

args = sys.argv[1:]
fdir, out = args[0], args[1]
vo = args[args.index('--vo') + 1] if '--vo' in args else None
if '--timing' in args: T.update(json.load(open(args[args.index('--timing') + 1])))
stills = args[args.index('--stills') + 1] if '--stills' in args else None
if stills: os.makedirs(stills, exist_ok=True)

mont, inter = tm.mont, tm.inter
DIG = mont(200, 800)
DW = max(DIG.getlength(str(d)) for d in range(10))
LAB = inter(30, 'Bold')

def grain(im, amt, seed):
    a = np.asarray(im).astype(np.int16)
    r = np.random.default_rng(seed).normal(0, amt, a.shape[:2])[..., None]
    return Image.fromarray(np.clip(a + r, 0, 255).astype(np.uint8))

VIG = None
def vignette(im):
    global VIG
    if VIG is None:
        y, x = np.mgrid[0:H, 0:W]
        d = np.sqrt(((x - W / 2) / (W * 0.62)) ** 2 + ((y - H / 2) / (H * 0.62)) ** 2)
        VIG = np.clip(1 - 0.42 * np.clip(d - 0.35, 0, 1) ** 1.6, 0, 1)[..., None]
    return Image.fromarray((np.asarray(im) * VIG).astype(np.uint8))

def counter_layer(n, k):
    """n = frames left; k = frame index (0-based)."""
    L = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    s = str(n); tension = max(0, (40 - n) / 40)
    col = tuple(int(255 + (c - 255) * min(1, tension * 1.4)) for c in CORAL)
    x = 72; y = 330
    for i, ch in enumerate(s):
        cx = x + DW * i + DW / 2
        # soft shadow for legibility over bright latex
        d.text((cx + 4, y + 6), ch, font=DIG, fill=(0, 0, 0, 120), anchor='mm')
        d.text((cx, y), ch, font=DIG, fill=col + (255,), anchor='mm')
    for row, lab in enumerate(('FRAMES UNTIL', 'CONTACT')):
        sp = 6; lx = 78; ly = 466 + row * 40
        for c in lab:
            d.text((lx + 2, ly + 2), c, font=LAB, fill=(0, 0, 0, 110), anchor='lm')
            d.text((lx, ly), c, font=LAB, fill=(235, 235, 235, 215), anchor='lm'); lx += LAB.getlength(c) + sp
    if tension > 0:   # pulse: scale the layer slightly each frame near the end
        sc = 1 + 0.035 * tension * (1 if k % 2 else 0)
        if sc > 1:
            L2 = L.resize((int(W * sc), int(H * sc)), Image.BICUBIC)
            ox, oy = int(72 * sc - 72), int(330 * sc - 330)
            L = L2.crop((ox, oy, ox + W, oy + H))
    return L

def base_frame(i):
    im = Image.open(os.path.join(fdir, f'{i:04d}.png')).convert('RGB').resize((W, H), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
    return vignette(im)

def wrap(text, font, maxw):
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if font.getlength(t) <= maxw: cur = t
        else: lines.append(cur); cur = w
    lines.append(cur); return lines

def text_layer(text, accent, a):
    f = mont(84, 800); L = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    lines = wrap(text, f, 900); lh = 104; y0 = H / 2 - lh * len(lines) / 2 + 20 * (1 - a)
    acc_words = set(accent.split())
    in_acc = False
    full = text.split(); wi = 0
    acc_start = text.find(accent); acc_end = acc_start + len(accent)
    pos = 0
    for li, line in enumerate(lines):
        x = W / 2 - f.getlength(line) / 2; y = y0 + li * lh
        for w in line.split():
            idx = text.find(w, pos); pos = idx + len(w)
            col = EMERALD if acc_start <= idx < acc_end else (255, 255, 255)
            d.text((x + 3, y + 5), w, font=f, fill=(0, 0, 0, int(140 * a)))
            d.text((x, y), w, font=f, fill=col + (int(255 * a),))
            x += f.getlength(w + ' ')
    return L

frames_out = []
def emit(im, tag=None):
    frames_out.append(im)
    if stills and tag: im.save(f'{stills}/{tag}.png')

# ---------------------------------------------------------------- video
tmp = tempfile.mkdtemp(prefix='needle_')
vid = f'{tmp}/v.mp4'
ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-profile:v', 'high', vid], stdin=subprocess.PIPE)
nf = 0
def put(im, tag=None):
    global nf
    if stills and tag: im.save(f'{stills}/{tag}.png')
    ff.stdin.write(im.convert('RGB').tobytes()); nf += 1

last = None
for k in range(N):
    fr = grain(base_frame(k + 1), 5, k).convert('RGBA')
    fr.alpha_composite(counter_layer(N - k, k))
    last = fr
    put(fr, {0: 'a_first', 105: 'a_mid', 180: 'a_late', N - 1: 'a_last'}.get(k))
# freeze on "1": hold, then dim + text lines, then end card
frozen = base_frame(N)
end_card_t = T['end_card']; total = T['end']
k = 0
while True:
    t = N / FPS + k / FPS
    if t >= end_card_t - 1e-6: break
    th = t - N / FPS
    if th < T['freeze_hold']:
        fr = grain(frozen, 5, 1000 + k).convert('RGBA'); fr.alpha_composite(counter_layer(1, 0))
        put(fr, 'b_hold' if k == 3 else None); k += 1; continue
    dim = min(1, (th - T['freeze_hold']) / 0.35)
    bg = ImageEnhance.Brightness(frozen.filter(ImageFilter.GaussianBlur(10 * dim))).enhance(1 - 0.68 * dim)
    fr = grain(bg, 5, 1000 + k).convert('RGBA')
    cl = counter_layer(1, 0); cl.putalpha(cl.getchannel('A').point(lambda v: int(v * (1 - dim)))); fr.alpha_composite(cl)
    for li, (s, e) in enumerate(T['lines']):
        if s <= t < e:
            a = min(1, (t - s) / 0.18) * min(1, (e - t) / 0.12 + 0.0001)
            fr.alpha_composite(text_layer(*LINES[li], a))
            if stills and abs(t - (s + 0.6)) < 0.5 / FPS: fr.save(f'{stills}/c_line{li + 1}.png')
    put(fr); k += 1
ne = int(round((total - end_card_t) * FPS))
for j in range(ne):
    put(tm.end_card('PDF', j / FPS), 'd_end' if j == ne - 1 else None)
ff.stdin.close(); ff.wait()
dur = nf / FPS

# ---------------------------------------------------------------- sound: tension bed -> hard silence -> soft bed under VO + ding
SR = au.SR; n = int(dur * SR); mix = np.zeros(n)
tf = N / FPS; nT = int(tf * SR); t = np.arange(nT) / SR
rng = np.random.default_rng(3)
# low drone rising in pitch and level
f0 = 42 + 14 * (t / tf) ** 2
ph = 2 * np.pi * np.cumsum(f0) / SR
drone = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3.01 * ph)) * (0.25 + 0.75 * (t / tf) ** 1.5)
air = au.bp(rng.uniform(-1, 1, nT), 300, 2500) * (t / tf) ** 3 * 0.6
# high whine creeping in (tension)
whine = np.sin(2 * np.pi * np.cumsum(2600 + 900 * (t / tf) ** 2) / SR) * 0.05 * (t / tf) ** 4
bed = au.norm(drone, 0.55) + au.norm(air, 0.18) + whine
mix[:nT] += bed
# ticks: every 10 frames, then 6, 4, 2 as the counter nears 0
def tick(f_hz=3200, d=0.012):
    m = int(0.03 * SR); tt = np.arange(m) / SR
    return (np.sin(2 * np.pi * f_hz * tt) + 0.5 * au.bp(rng.uniform(-1, 1, m), 2000, 8000)) * np.exp(-tt / d)
fr_i = 0
while fr_i < N:
    left = N - fr_i
    step = 10 if left > 90 else 6 if left > 45 else 4 if left > 20 else 2
    s = int(fr_i / FPS * SR); tk = tick(3000 + 600 * (1 - left / N)) * (0.35 + 0.4 * (1 - left / N))
    mix[s:s + len(tk)] += tk[:max(0, n - s)]
    fr_i += step
# heartbeat in the last 3 s
for hb in np.arange(4.2, tf, 0.62 - 0.0):
    for off, g in ((0, 1.0), (0.16, 0.7)):
        s = int((hb + off) * SR); m = int(0.18 * SR); tt = np.arange(m) / SR
        thump = np.sin(2 * np.pi * (55 - 20 * tt / 0.18) * tt) * np.exp(-tt / 0.06) * g * 0.9
        mix[s:s + m] += thump[:max(0, n - s)]
mix[nT - int(0.01 * SR):nT] *= np.linspace(1, 0, int(0.01 * SR))   # hard cut on the freeze
# soft pad under the VO section (starts after the freeze hold)
ps = tf + T['freeze_hold'] + 0.3; pe = dur
m = int((pe - ps) * SR); tt = np.arange(m) / SR
pad = sum(np.sin(2 * np.pi * au.hz(mm) * tt + i) for i, mm in enumerate((45, 52, 57, 64))) * 0.05
pad = au.lp(pad, 1800) * np.minimum(1, tt / 1.2)
mix[int(ps * SR):int(ps * SR) + m] += pad
ding = au.sfx_ding(); s = int(end_card_t * SR) + int(0.15 * SR); mix[s:s + len(ding)] += 0.5 * ding[:max(0, n - s)]
wav = f'{tmp}/fx.wav'
au.wavfile.write(wav, SR, (au.norm(mix, 0.9) * 32767).astype(np.int16))

final_wav = f'{tmp}/mix.wav'
if vo:
    # VO placed at vo_start, effects ducked to -9 dB under it
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', wav, '-i', vo, '-filter_complex',
                    f"[1:a]aresample=48000,adelay={int(T['vo_start'] * 1000)}|{int(T['vo_start'] * 1000)},apad[v];"
                    f"[0:a]volume=0.55[f];[f][v]amix=inputs=2:duration=first:normalize=0[m]", '-map', '[m]', '-ar', '48000', final_wav], check=True)
else:
    final_wav = wav
meas = subprocess.run(['ffmpeg', '-hide_banner', '-i', final_wav, '-af', 'loudnorm=I=-14:TP=-2:LRA=11:print_format=json', '-f', 'null', '-'],
                      capture_output=True, text=True).stderr
js = json.loads(meas[meas.rindex('{'):meas.rindex('}') + 1])
ln = (f"loudnorm=I=-14:TP=-2:LRA=11:measured_I={js['input_i']}:measured_TP={js['input_tp']}:"
      f"measured_LRA={js['input_lra']}:measured_thresh={js['input_thresh']}:offset={js['target_offset']}:linear=true")
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', vid, '-i', final_wav, '-af', ln, '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-t', f'{dur:.3f}', '-movflags', '+faststart', out], check=True)
subprocess.run(['rm', '-rf', tmp])
print('duration', dur)
