#!/usr/bin/env python3
"""Tablio meme-reel renderer. Fully local: PIL frames -> ffmpeg, synthesized audio
(tablio_meme_audio.py next to this file). No ElevenLabs, no voice, no external assets.
Usage: python3 tablio_meme.py spec.json out.mp4 [--stills DIR]
Spec: {"format":"meme","id","kw","style":"stroke|pill","top","bed":"beat|laugh|none",
       "shots":[{"type","dur","sfx","top","p"}],"end_dur"}
Shot types: chat, monitor, slide, paper, column_a, locked, retype, reaction.
"""
import json, math, os, random, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tablio_meme_audio as audio

W, H, FPS = 1080, 1920, 30
EMERALD = (16, 185, 129)
MONT = '/usr/share/fonts/truetype/google-fonts/ofl/montserrat/Montserrat[wght].ttf'
INTER = '/usr/share/fonts/opentype/inter/Inter-%s.otf'
EMOJI = '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf'
SS = 2

_fc = {}
def mont(size, weight=800):
    k = ('m', size, weight)
    if k not in _fc:
        f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([weight]); _fc[k] = f
    return _fc[k]
def inter(size, style='Bold'):
    k = ('i', size, style)
    if k not in _fc: _fc[k] = ImageFont.truetype(INTER % style, size)
    return _fc[k]

_emo = {}
def emoji_img(ch, size):
    k = (ch, size)
    if k not in _emo:
        f = ImageFont.truetype(EMOJI, 109)
        im = Image.new('RGBA', (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
        im = im.crop(im.getbbox())
        s = size / max(im.size)
        _emo[k] = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    return _emo[k]

def is_emoji(c):
    o = ord(c)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF

def runs(text):
    out, cur = [], ''
    for c in text:
        if ord(c) in (0xFE0F, 0x200D): continue
        if is_emoji(c):
            if cur: out.append(('t', cur)); cur = ''
            out.append(('e', c))
        else:
            cur += c
    if cur: out.append(('t', cur))
    return out

def rich_width(text, font, esize):
    return sum(font.getlength(s) if k == 't' else esize * 1.08 for k, s in runs(text))

def wrap(text, font, esize, maxw):
    lines, cur = [], ''
    for wd in text.split(' '):
        t = (cur + ' ' + wd).strip()
        if rich_width(t, font, esize) <= maxw or not cur: cur = t
        else: lines.append(cur); cur = wd
    if cur: lines.append(cur)
    return lines

def text_layer(text, style):
    """Top POV text as a transparent RGBA layer (rendered once per text)."""
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    pill = style == 'pill'
    font = mont(58, 700) if pill else mont(70, 800)
    esize = 62 if pill else 74
    lines = wrap(text, font, esize, 900 if pill else 960)
    asc, desc = font.getmetrics(); lh = int((asc + desc) * 1.18)
    y = 250
    for ln in lines:
        lw = rich_width(ln, font, esize); x = W / 2 - lw / 2
        if pill:
            d.rounded_rectangle([x - 22, y - 6, x + lw + 22, y + asc + desc + 8], radius=22, fill=(255, 255, 255, 255))
        for kind, s in runs(ln):
            if kind == 't':
                if pill: d.text((x, y), s, font=font, fill=(0, 0, 0, 255))
                else: d.text((x, y), s, font=font, fill=(255, 255, 255, 255), stroke_width=7, stroke_fill=(0, 0, 0, 255))
                x += font.getlength(s)
            else:
                em = emoji_img(s, esize)
                lay.alpha_composite(em, (int(x + esize * 0.04), int(y + (asc + desc) / 2 - em.height / 2 + 2)))
                x += esize * 1.08
        y += lh + (14 if pill else 0)
    return lay

# ---------------------------------------------------------------- helpers
def vgrad(w, h, top, bot):
    a = np.linspace(0, 1, h)[:, None, None]
    arr = np.array(top)[None, None, :] * (1 - a) + np.array(bot)[None, None, :] * a
    return Image.fromarray(np.repeat(arr, w, axis=1).astype(np.uint8), 'RGB')

def vignette(img, strength=0.55):
    w, h = img.size
    y, x = np.ogrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    m = 1 - strength * np.clip((x ** 2 + y ** 2) / 1.6, 0, 1) ** 1.3
    return Image.fromarray(np.clip(np.asarray(img).astype(np.float32) * m[..., None], 0, 255).astype(np.uint8))

def noise(img, amt=10, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(img).astype(np.int16) + rng.integers(-amt, amt + 1, (img.height, img.width, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def find_coeffs(dst, src):
    m = []
    for (x, y), (X, Y) in zip(dst, src):
        m.append([x, y, 1, 0, 0, 0, -X * x, -X * y]); m.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    return np.linalg.solve(np.array(m, float), np.array(src, float).reshape(8)).tolist()

def warp_onto(bg, img, quad):
    w, h = img.size
    layer = img.convert('RGBA').transform(bg.size, Image.PERSPECTIVE, find_coeffs(quad, [(0, 0), (w, 0), (w, h), (0, h)]), Image.BICUBIC)
    bg.paste(layer, (0, 0), layer)
    return bg

TABLE = [["Region", "Q1", "Q2", "Q3"],
         ["North", "41,200", "43,900", "47,350"],
         ["South", "38,760", "36,410", "39,980"],
         ["East", "52,130", "55,020", "58,640"],
         ["West", "29,870", "31,550", "30,920"],
         ["Central", "44,300", "46,780", "49,010"],
         ["Online", "61,450", "64,200", "71,830"],
         ["Total", "267,710", "277,860", "297,730"]]

def sheet_image(w, h, title=None, dark=False, grid=True):
    bg = (250, 250, 250) if not dark else (24, 24, 27)
    fg = (20, 20, 20) if not dark else (230, 230, 230)
    im = Image.new('RGB', (w, h), bg); d = ImageDraw.Draw(im)
    y0 = 0
    if title:
        fs = int(h * 0.075)
        while fs > 12 and inter(fs, 'Bold').getlength(title) > w * 0.88: fs -= 2
        d.text((w * 0.06, h * 0.05), title, font=inter(fs, 'Bold'), fill=fg)
        y0 = int(h * 0.19)
    rows = len(TABLE); rh = (h - y0 - h * 0.04) / rows
    cols = [0.06, 0.33, 0.545, 0.76]
    for r, row in enumerate(TABLE):
        y = y0 + r * rh
        if r == 0:
            d.rectangle([w * 0.04, y, w * 0.96, y + rh], fill=(34, 120, 90) if not dark else (16, 120, 90))
        elif grid and r % 2 == 0:
            d.rectangle([w * 0.04, y, w * 0.96, y + rh], fill=(236, 240, 238) if not dark else (36, 36, 40))
        for c, cell in enumerate(row):
            f = inter(int(rh * 0.42), 'Bold' if (r == 0 or c == 0 or r == rows - 1) else 'Medium')
            d.text((w * cols[c], y + rh * 0.24), cell, font=f, fill=(255, 255, 255) if r == 0 else fg)
        if grid: d.line([w * 0.04, y + rh, w * 0.96, y + rh], fill=(200, 205, 203) if not dark else (60, 60, 66), width=2)
    return im

def moire(im, seed=1, amt=0.22):
    w, h = im.size
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(0.18, 0.26), rng.uniform(0.17, 0.25)
    m = np.sin((x * math.cos(0.32) + y * math.sin(0.32)) * a + (x * x + y * y) * 0.00002) * np.sin((x * math.cos(-0.2) + y * math.sin(-0.2)) * b)
    arr = np.asarray(im).astype(np.float32) * np.stack([1 + amt * m, 1 - amt * 0.6 * m, 1 + amt * 0.9 * np.roll(m, 7, 1)], -1)
    arr *= (0.9 + 0.1 * np.sin(x * 2.1))[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def cursor(d, x, y, s=1.0):
    pts = [(0, 0), (0, 46), (12, 35), (21, 55), (29, 51), (20, 32), (36, 32)]
    d.polygon([(x + px * s, y + py * s) for px, py in pts], fill=(255, 255, 255), outline=(0, 0, 0))

# ---------------------------------------------------------------- shots (each returns a BW x BH image)
BW, BH = int(W * 1.12), int(H * 1.12)
OX, OY = (BW - W) // 2, (BH - H) // 2

def shot_chat(p, t_frac=1.0, seed=0):
    im = Image.new('RGB', (BW, BH), (11, 11, 13)); d = ImageDraw.Draw(im)
    ox, oy = OX, OY
    d.rectangle([0, oy + 580, BW, oy + 720], fill=(22, 22, 25))
    d.ellipse([ox + 60, oy + 605, ox + 150, oy + 695], fill=(70, 70, 78))
    d.text((ox + 105, oy + 650), (p.get('name', 'bro')[:1]).upper(), font=inter(44, 'Bold'), fill=(230, 230, 230), anchor='mm')
    d.text((ox + 175, oy + 612), p.get('name', 'bro'), font=inter(46, 'Bold'), fill=(240, 240, 240))
    d.text((ox + 175, oy + 668), 'online', font=inter(28, 'Medium'), fill=(140, 140, 150))
    y = oy + 780
    msgs = p.get('msgs', [])
    shown = msgs[:max(1, min(len(msgs), int(t_frac * (len(msgs) + 0.4)) + 1))]
    for m in shown:
        me = m.get('from') == 'me'
        if 'img' in m:
            gw, gh = 330, 240; x0 = ox + 70
            for i in range(4):
                gx = x0 + (i % 2) * (gw + 10); gy = y + (i // 2) * (gh + 10)
                th = moire(sheet_image(520, 380, None), seed=i + 3, amt=0.35).rotate(random.Random(i).uniform(-7, 7), resample=Image.BICUBIC, fillcolor=(30, 30, 34))
                th = th.resize((gw, gh)).filter(ImageFilter.GaussianBlur(1.2))
                th = Image.blend(th, Image.new('RGB', th.size, (60, 70, 90)), 0.25)
                mask = Image.new('L', (gw, gh), 0); ImageDraw.Draw(mask).rounded_rectangle([0, 0, gw, gh], 18, fill=255)
                im.paste(th, (gx, gy), mask)
            n = m.get('n', 14)
            lx, ly = x0 + gw + 10, y + gh + 10
            mask = Image.new('L', (gw, gh), 0); ImageDraw.Draw(mask).rounded_rectangle([0, 0, gw, gh], 18, fill=150)
            im.paste((0, 0, 0), (lx, ly, lx + gw, ly + gh), mask)
            d.text((lx + gw / 2, ly + gh / 2), f'+{n - 3}', font=inter(84, 'Bold'), fill=(255, 255, 255), anchor='mm')
            d.text((x0, y + 2 * gh + 30), f'{n} photos', font=inter(30, 'Medium'), fill=(140, 140, 150))
            y += 2 * gh + 90
        else:
            f = inter(52, 'Medium'); tw = f.getlength(m['text'])
            bw, bh = tw + 72, 116
            x = ox + W - 70 - bw if me else ox + 70
            d.rounded_rectangle([x, y, x + bw, y + bh], radius=44, fill=EMERALD if me else (44, 44, 50))
            d.text((x + 36, y + 28), m['text'], font=f, fill=(255, 255, 255))
            y += bh + 28
    d.rounded_rectangle([ox + 50, oy + H - 300, ox + W - 50, oy + H - 200], radius=50, fill=(28, 28, 32))
    d.text((ox + 100, oy + H - 272), 'Message', font=inter(38, 'Medium'), fill=(110, 110, 120))
    return im

def shot_monitor(p, seed=0):
    bg = vgrad(BW, BH, (38, 34, 30), (14, 12, 11))
    scr = moire(sheet_image(1500, 1000, p.get('title', 'Sales_FINAL.xlsx')), seed=seed + 11, amt=0.30)
    bez = Image.new('RGB', (1580, 1080), (12, 12, 12)); bez.paste(scr, (40, 40))
    cx, cy = BW / 2, BH / 2 + 40
    quad = [(cx - 520, cy - 340), (cx + 480, cy - 300), (cx + 530, cy + 320), (cx - 500, cy + 360)]
    warp_onto(bg, bez, quad)
    gl = Image.new('L', (BW, BH), 0); ImageDraw.Draw(gl).polygon([(cx - 200, cy - 360), (cx - 40, cy - 360), (cx - 300, cy + 360), (cx - 460, cy + 360)], fill=55)
    bg.paste((255, 255, 255), (0, 0), gl.filter(ImageFilter.GaussianBlur(40)))
    bg = bg.filter(ImageFilter.GaussianBlur(1.6))
    return noise(vignette(bg, 0.6), 12, seed)

def shot_slide(p, seed=0):
    bg = vgrad(BW, BH, (18, 18, 24), (6, 6, 8))
    sl = sheet_image(1600, 1000, p.get('title', 'Q3 results by region'))
    sl = Image.blend(sl, Image.new('RGB', sl.size, (255, 244, 220)), 0.15)
    cx, cy = BW / 2, BH / 2 + 20
    quad = [(cx - 560, cy - 300), (cx + 600, cy - 340), (cx + 560, cy + 380), (cx - 520, cy + 330)]
    halo = Image.new('L', (BW, BH), 0); ImageDraw.Draw(halo).polygon(quad, fill=120)
    bg.paste((120, 120, 140), (0, 0), halo.filter(ImageFilter.GaussianBlur(70)))
    warp_onto(bg, sl, quad)
    bg = bg.filter(ImageFilter.GaussianBlur(3.2))
    d = ImageDraw.Draw(bg)
    for i, x in enumerate([90, 330, 560, 820, 1060]):
        r = 150 + (i % 2) * 30; yb = BH - 240 + (i % 3) * 40
        d.ellipse([x - r * 0.75, yb - r, x + r * 0.75, yb + r * 0.9], fill=(4, 4, 6))
        d.rectangle([x - r * 1.4, yb + r * 0.6, x + r * 1.4, BH], fill=(4, 4, 6))
    return noise(vignette(bg, 0.5), 16, seed)

def shot_paper(p, seed=0):
    bg = vgrad(BW, BH, (120, 92, 64), (70, 52, 36))
    arr = np.asarray(bg).astype(np.float32)
    yy = np.arange(BH)[:, None]; xx = np.arange(BW)[None, :]
    grain = 10 * np.sin(yy * 0.05 + 6 * np.sin(xx * 0.004)) + 5 * np.sin(yy * 0.31)
    bg = Image.fromarray(np.clip(arr + grain[..., None], 0, 255).astype(np.uint8))
    pap = sheet_image(1150, 1400, p.get('title', 'printed_v2 (scan)'), grid=False)
    pa = np.asarray(pap).astype(np.float32)
    shade = np.linspace(1.0, 0.78, pap.width)[None, :, None] * (0.92 + 0.08 * np.sin(np.arange(pap.height) * 0.012))[:, None, None]
    pap = Image.fromarray(np.clip(pa * shade * np.array([1.0, 0.97, 0.9]), 0, 255).astype(np.uint8))
    cx, cy = BW / 2, BH / 2 + 150
    quad = [(cx - 380, cy - 520), (cx + 420, cy - 480), (cx + 460, cy + 560), (cx - 440, cy + 530)]
    sh = Image.new('L', (BW, BH), 0); ImageDraw.Draw(sh).polygon([(x + 30, y + 40) for x, y in quad], fill=150)
    bg.paste((0, 0, 0), (0, 0), sh.filter(ImageFilter.GaussianBlur(30)))
    warp_onto(bg, pap, quad)
    bg = bg.filter(ImageFilter.GaussianBlur(1.3))
    hs = Image.new('L', (BW, BH), 0); ImageDraw.Draw(hs).ellipse([cx - 200, cy - 360, cx + 120, cy - 60], fill=90)
    bg.paste((255, 255, 250), (0, 0), hs.filter(ImageFilter.GaussianBlur(80)))
    return noise(vignette(bg, 0.5), 10, seed)

def shot_column_a(p, seed=0):
    """Spreadsheet where the whole pasted table landed in column A."""
    im = vgrad(BW, BH, (16, 16, 19), (8, 8, 10))
    ww, wh = 1000, 1020
    win = Image.new('RGB', (ww, wh), (255, 255, 255)); d = ImageDraw.Draw(win)
    d.rectangle([0, 0, ww, 74], fill=(33, 115, 70))
    d.text((26, 20), p.get('file', 'Book1.xlsx'), font=inter(32, 'Bold'), fill=(255, 255, 255))
    d.rectangle([0, 74, ww, 138], fill=(243, 243, 243)); d.line([0, 138, ww, 138], fill=(210, 210, 210), width=2)
    d.text((22, 92), 'A2', font=inter(28, 'Bold'), fill=(60, 60, 60))
    d.text((110, 92), 'fx', font=inter(28, 'Bold'), fill=(150, 150, 150))
    d.text((160, 92), 'North 41,200 43,900 47,350', font=inter(28, 'Medium'), fill=(40, 40, 40))
    rh, hx, cw, top = 58, 64, 156, 138
    d.rectangle([0, top, ww, top + 46], fill=(236, 236, 236))
    for i, L in enumerate('ABCDEF'):
        x = hx + i * cw
        if i == 0: d.rectangle([x, top, x + cw, top + 46], fill=(198, 230, 210))
        d.text((x + cw / 2, top + 23), L, font=inter(26, 'Bold' if i == 0 else 'Medium'), fill=(30, 100, 60) if i == 0 else (90, 90, 90), anchor='mm')
        d.line([x, top, x, wh], fill=(220, 220, 220), width=2)
    lines = [' '.join(r) for r in TABLE]
    lines = [p.get('title', 'Q3 results by region   Page 1 of 3')] + lines[:2] + ['South38,76036,41039,980'] + lines[3:6] + ['Online 61,450 64,200 71,830 Total 267,710', '277,860 297,730', '', '']
    for r in range(15):
        y = top + 46 + r * rh
        d.rectangle([0, y, hx, y + rh], fill=(236, 236, 236))
        d.text((hx / 2, y + rh / 2), str(r + 1), font=inter(24, 'Medium'), fill=(110, 110, 110), anchor='mm')
        d.line([0, y + rh, ww, y + rh], fill=(222, 222, 222), width=2)
        if r < len(lines) and lines[r]:
            d.text((hx + 10, y + 13), lines[r], font=inter(29, 'Medium'), fill=(25, 25, 25))
    y2 = top + 46 + rh
    d.rectangle([hx, y2, hx + cw, y2 + rh], outline=(33, 115, 70), width=5)
    cursor(d, hx + cw * 3.4, top + 46 + rh * 6.3, 1.3)
    im.paste(win, (OX + (W - ww) // 2, OY + 640))
    return noise(im, 4, seed)

def lock_icon(d, cx, cy, s, col):
    d.rounded_rectangle([cx - 30 * s, cy - 4 * s, cx + 30 * s, cy + 40 * s], radius=int(8 * s), fill=col)
    d.arc([cx - 20 * s, cy - 40 * s, cx + 20 * s, cy + 6 * s], 180, 360, fill=col, width=int(9 * s))
    d.line([cx - 20 * s, cy - 17 * s, cx - 20 * s, cy], fill=col, width=int(9 * s))
    d.line([cx + 20 * s, cy - 17 * s, cx + 20 * s, cy], fill=col, width=int(9 * s))

def shot_locked(p, seed=0):
    """PDF viewer: selection, then 'Copying is not allowed'."""
    im = Image.new('RGB', (BW, BH), (82, 86, 89)); d = ImageDraw.Draw(im)
    ox, oy = OX, OY
    d.rectangle([0, oy + 560, BW, oy + 640], fill=(50, 54, 57))
    d.text((ox + 40, oy + 582), p.get('file', 'report_final.pdf'), font=inter(32, 'Medium'), fill=(230, 230, 230))
    d.text((ox + W - 40, oy + 582), '1 / 14', font=inter(30, 'Medium'), fill=(200, 200, 200), anchor='ra')
    px0, py0, pw = ox + 90, oy + 700, W - 180
    d.rectangle([px0, py0, px0 + pw, oy + 1720], fill=(255, 255, 255))
    d.text((px0 + 60, py0 + 60), p.get('title', 'Q3 Market Report'), font=inter(46, 'Bold'), fill=(20, 20, 20))
    rr = random.Random(seed)
    y = py0 + 150
    for blk in range(3):
        for ln in range(4):
            wln = pw - 120 - (rr.randint(40, 260) if ln == 3 else rr.randint(0, 30))
            if blk == 1: d.rectangle([px0 + 56, y - 6, px0 + 60 + wln + 4, y + 24], fill=(150, 195, 255))
            d.rounded_rectangle([px0 + 60, y, px0 + 60 + wln, y + 18], radius=6, fill=(170, 170, 175) if blk != 1 else (90, 110, 150))
            y += 40
        y += 30
    for r in range(4):
        d.line([px0 + 60, y + r * 50, px0 + pw - 60, y + r * 50], fill=(200, 200, 200), width=2)
    # modal
    mw, mh = 820, 420; mx, my = ox + (W - mw) // 2, oy + 980
    sh = Image.new('L', (BW, BH), 0); ImageDraw.Draw(sh).rounded_rectangle([mx + 10, my + 24, mx + mw + 10, my + mh + 24], 36, fill=150)
    im.paste((0, 0, 0), (0, 0), sh.filter(ImageFilter.GaussianBlur(24)))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([mx, my, mx + mw, my + mh], radius=36, fill=(250, 250, 250))
    lock_icon(d, mx + mw / 2, my + 95, 1.3, (220, 53, 69))
    d.text((mx + mw / 2, my + 205), p.get('msg', 'Copying is not allowed'), font=inter(50, 'Bold'), fill=(20, 20, 20), anchor='mm')
    d.text((mx + mw / 2, my + 268), 'This document is protected.', font=inter(32, 'Medium'), fill=(110, 110, 115), anchor='mm')
    d.rounded_rectangle([mx + mw / 2 - 110, my + 318, mx + mw / 2 + 110, my + 390], radius=22, fill=(40, 40, 44))
    d.text((mx + mw / 2, my + 354), 'OK', font=inter(34, 'Bold'), fill=(255, 255, 255), anchor='mm')
    cursor(d, mx + mw / 2 + 60, my + 372, 1.3)
    return im

def shot_retype(p, seed=0):
    """3:47 AM, pixelated screenshot on the left, hand-typed sheet on the right."""
    bg = vgrad(BW, BH, (12, 16, 30), (4, 5, 9)); d = ImageDraw.Draw(bg)
    clock = p.get('clock', '3:47 AM')
    d.text((BW / 2, OY + 640), clock, font=mont(120, 800), fill=(255, 255, 255), anchor='mm')
    scr = Image.new('RGB', (1600, 1000), (30, 30, 34)); sd = ImageDraw.Draw(scr)
    pix = sheet_image(760, 900, None).resize((76, 90), Image.BILINEAR).resize((760, 900), Image.NEAREST)
    scr.paste(pix, (20, 50))
    typed = sheet_image(780, 900, None)
    td = ImageDraw.Draw(typed)
    rows = len(TABLE); rh = (900 - 900 * 0.04) / rows
    td.rectangle([0, rh * 5, 780, 900], fill=(250, 250, 250))
    for r in range(5, rows):
        td.line([780 * 0.04, rh * (r + 1), 780 * 0.96, rh * (r + 1)], fill=(200, 205, 203), width=2)
    td.text((780 * 0.06, rh * 5 + rh * 0.24), 'Centr', font=inter(int(rh * 0.42), 'Bold'), fill=(20, 20, 20))
    td.line([780 * 0.06 + inter(int(rh * 0.42), 'Bold').getlength('Centr') + 4, rh * 5 + rh * 0.2, 780 * 0.06 + inter(int(rh * 0.42), 'Bold').getlength('Centr') + 4, rh * 5 + rh * 0.8], fill=(16, 120, 90), width=4)
    td.rectangle([780 * 0.545 - 8, rh * 2 + 6, 780 * 0.545 + 150, rh * 3 - 6], outline=(220, 53, 69), width=5)
    td.text((780 * 0.545, rh * 2 + rh * 0.24), '36,140', font=inter(int(rh * 0.42), 'Medium'), fill=(220, 53, 69))
    scr.paste(typed, (800, 50))
    sd.rectangle([0, 0, 1600, 44], fill=(45, 45, 50))
    cx, cy = BW / 2, BH / 2 + 160
    quad = [(cx - 560, cy - 330), (cx + 560, cy - 330), (cx + 590, cy + 350), (cx - 590, cy + 350)]
    glow = Image.new('L', (BW, BH), 0); ImageDraw.Draw(glow).polygon(quad, fill=110)
    bg.paste((90, 120, 200), (0, 0), glow.filter(ImageFilter.GaussianBlur(90)))
    warp_onto(bg, scr, quad)
    return noise(vignette(bg, 0.5), 8, seed)

def draw_tabby(size, mood='shock', bg_top=(70, 86, 110), bg_bot=(26, 30, 40)):
    S = size * SS
    im = vgrad(S, int(S * 16 / 9), bg_top, bg_bot); d = ImageDraw.Draw(im)
    w, h = im.size; cx, cy = w / 2, h * 0.55; R = w * 0.46
    fur, rim = (14, 14, 16), (60, 64, 74)
    d.ellipse([cx - R * 1.35, cy + R * 0.55, cx + R * 1.35, cy + R * 3.2], fill=fur)
    for sgn in (-1, 1):
        ear = [(cx + sgn * R * 0.88, cy - R * 0.35), (cx + sgn * R * 0.82, cy - R * 1.32), (cx + sgn * R * 0.22, cy - R * 0.80)]
        d.polygon([(x + sgn * 6 * SS, y - 6 * SS) for x, y in ear], fill=rim)
        d.polygon(ear, fill=fur)
        d.polygon([(cx + sgn * R * 0.78, cy - R * 0.48), (cx + sgn * R * 0.76, cy - R * 1.08), (cx + sgn * R * 0.38, cy - R * 0.78)], fill=(58, 34, 42))
    d.ellipse([cx - R - 6 * SS, cy - R * 0.95 - 6 * SS, cx + R + 6 * SS, cy + R * 0.85 + 6 * SS], fill=rim)
    d.ellipse([cx - R, cy - R * 0.95, cx + R, cy + R * 0.85], fill=fur)
    ex, ey, er = R * 0.42, cy - R * 0.12, R * 0.25
    for sgn in (-1, 1):
        x = cx + sgn * ex
        if mood == 'shock':
            d.ellipse([x - er * 1.15, ey - er * 1.25, x + er * 1.15, ey + er * 1.15], fill=EMERALD)
            d.ellipse([x - er * 0.16, ey - er * 0.25, x + er * 0.16, ey + er * 0.15], fill=(5, 5, 5))
            d.ellipse([x - er * 0.75, ey - er * 0.95, x - er * 0.35, ey - er * 0.55], fill=(255, 255, 255))
        elif mood == 'squint':
            d.ellipse([x - er * 1.05, ey - er * 0.32, x + er * 1.05, ey + er * 0.32], fill=EMERALD)
            d.rectangle([x - er * 0.08, ey - er * 0.3, x + er * 0.08, ey + er * 0.3], fill=(5, 5, 5))
            d.polygon([(x - er * 1.3, ey - er * 0.95), (x + er * 1.3, ey - er * (0.55 if sgn > 0 else 0.35)), (x + er * 1.3, ey - er * 0.15), (x - er * 1.3, ey - er * 0.15)], fill=fur)
            d.line([(x - er * 1.1, ey - er * (0.6 if sgn < 0 else 0.9)), (x + er * 1.1, ey - er * (0.3 if sgn < 0 else 0.6))], fill=rim, width=7 * SS)
        elif mood == 'cry':
            d.ellipse([x - er, ey - er, x + er, ey + er], fill=EMERALD)
            d.ellipse([x - er * 0.45, ey - er * 0.45, x + er * 0.45, ey + er * 0.45], fill=(5, 5, 5))
            d.ellipse([x - er * 0.6, ey - er * 0.7, x - er * 0.2, ey - er * 0.3], fill=(255, 255, 255))
            d.rounded_rectangle([x - er * 0.25, ey + er * 0.9, x + er * 0.25, ey + R * 1.1], radius=int(er * 0.25), fill=(120, 190, 255))
        else:
            d.ellipse([x - er, ey - er * 0.8, x + er, ey + er * 0.8], fill=EMERALD)
            d.ellipse([x - er * 0.12, ey - er * 0.7, x + er * 0.12, ey + er * 0.7], fill=(5, 5, 5))
    ny = cy + R * 0.25
    d.polygon([(cx - R * 0.07, ny), (cx + R * 0.07, ny), (cx, ny + R * 0.07)], fill=(95, 60, 70))
    if mood == 'shock':
        d.ellipse([cx - R * 0.09, ny + R * 0.13, cx + R * 0.09, ny + R * 0.36], fill=(60, 22, 30))
    else:
        d.arc([cx - R * 0.16, ny + R * 0.02, cx, ny + R * 0.16], 0, 180, fill=rim, width=5 * SS)
        d.arc([cx, ny + R * 0.02, cx + R * 0.16, ny + R * 0.16], 0, 180, fill=rim, width=5 * SS)
    for sgn in (-1, 1):
        for k, dy in enumerate((-0.05, 0.05, 0.15)):
            d.line([(cx + sgn * R * 0.25, ny + R * 0.08 + R * dy * 0.3), (cx + sgn * R * 1.05, ny + R * dy - R * 0.02 * (1 - k))], fill=(150, 150, 158), width=3 * SS)
    return im.resize((size, int(size * 16 / 9)), Image.LANCZOS)

def shot_reaction(p, seed=0):
    mood = p.get('mood', 'shock')
    cols = {'shock': ((96, 64, 160), (30, 18, 52)), 'squint': ((150, 110, 60), (48, 34, 18)),
            'cry': ((60, 100, 160), (16, 28, 52)), 'stare': ((90, 96, 104), (28, 30, 34))}.get(mood, ((90, 96, 104), (28, 30, 34)))
    im = draw_tabby(BW, mood, *cols)
    im = im.crop((0, 0, BW, BH)) if im.height >= BH else im.resize((BW, BH))
    if mood == 'shock':
        d = ImageDraw.Draw(im); rng = random.Random(seed); cx, cy = BW / 2, BH * 0.5
        for i in range(48):
            a = i / 48 * 2 * math.pi + rng.uniform(-0.04, 0.04)
            r0, r1 = BW * 0.62, BW * 1.3
            d.line([(cx + math.cos(a) * r0, cy + math.sin(a) * r0), (cx + math.cos(a) * r1, cy + math.sin(a) * r1)], fill=(255, 255, 255), width=rng.randint(2, 6))
    if p.get('label'):
        ImageDraw.Draw(im).text((BW * 0.72, BH * 0.30), p['label'], font=mont(170, 900), fill=(255, 255, 255), stroke_width=10, stroke_fill=(0, 0, 0), anchor='mm')
    return noise(vignette(im, 0.45), 6, seed)

def wordmark(img, cx, cy, cap, color=(255, 255, 255)):
    """TABLIO + round emerald dot on the baseline (~0.28x cap height)."""
    f = mont(int(cap / 0.70), 800); d = ImageDraw.Draw(img)
    tw = f.getlength('TABLIO'); dot = cap * 0.28; gap = cap * 0.06
    x0 = cx - (tw + gap + dot) / 2
    bb = f.getbbox('TABLIO')
    ytop = cy - (bb[3] - bb[1]) / 2 - bb[1]
    d.text((x0, ytop), 'TABLIO', font=f, fill=color)
    base = ytop + bb[3]; dx = x0 + tw + gap
    d.ellipse([dx, base - dot, dx + dot, base], fill=EMERALD)

_end = {}
def end_card(kw, t):
    im = Image.new('RGB', (W, H), (10, 10, 12))
    ease = lambda v: 1 - (1 - min(1, max(0, v))) ** 3
    if 'tab' not in _end:
        tb = draw_tabby(560, 'stare', (10, 10, 12), (10, 10, 12))
        _end['tab'] = tb.crop((0, int(tb.height * 0.16), tb.width, int(tb.height * 0.62)))
        gl = Image.new('L', (W, H), 0); ImageDraw.Draw(gl).ellipse([140, 520, 940, 1200], fill=70)
        _end['glow'] = gl.filter(ImageFilter.GaussianBlur(160))
        L = Image.new('RGBA', (W * 2, 600), (0, 0, 0, 0)); wordmark(L, W, 300, 300)
        _end['wm'] = L.resize((W, 300), Image.LANCZOS)
    tab = _end['tab']
    rise = ease((t - 0.3) / 0.45)
    if rise > 0: im.paste(tab, (W // 2 - tab.width // 2, int(H - 170 - tab.height * rise)))
    im.paste((16, 120, 90), (0, 0), _end['glow'])
    a1, a2, a3 = ease(t / 0.25), ease((t - 0.15) / 0.3), ease((t - 0.45) / 0.3)
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(lay).text((W / 2, 690), 'Just use', font=mont(86, 700), fill=(240, 240, 240, int(255 * a1)), anchor='mm')
    im.paste(lay, (0, 0), lay)
    if a2 > 0:
        s = 0.85 + 0.15 * a2
        L2 = _end['wm'].resize((int(W * s), int(300 * s)), Image.LANCZOS)
        L2.putalpha(L2.getchannel('A').point(lambda v: int(v * a2)))
        im.paste(L2, (int(W / 2 - L2.width / 2), int(860 - L2.height / 2)), L2)
    if a3 > 0:
        lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); dl = ImageDraw.Draw(lay)
        dl.text((W / 2, 1055), 'tablioproject.com', font=mont(58, 600), fill=(220, 220, 220, int(255 * a3)), anchor='mm')
        pill_txt = f'comment "{kw}" for the free guide'
        f = mont(44, 700); tw = f.getlength(pill_txt)
        dl.rounded_rectangle([W / 2 - tw / 2 - 34, 1150, W / 2 + tw / 2 + 34, 1240], radius=45, fill=(16, 185, 129, int(255 * a3)))
        dl.text((W / 2, 1195), pill_txt, font=f, fill=(255, 255, 255, int(255 * a3)), anchor='mm')
        dl.text((W / 2, 1310), '@tablioproject', font=inter(36, 'Medium'), fill=(150, 150, 160, int(255 * a3)), anchor='mm')
        im.paste(lay, (0, 0), lay)
    return im

BUILDERS = {'chat': shot_chat, 'monitor': shot_monitor, 'slide': shot_slide, 'paper': shot_paper,
            'column_a': shot_column_a, 'locked': shot_locked, 'retype': shot_retype, 'reaction': shot_reaction}

# ---------------------------------------------------------------- render
def frames(spec, stills=None):
    shots = spec['shots']; style = spec.get('style', 'stroke')
    top = spec['top']; layers = {}
    for si, sh in enumerate(shots):
        if sh['type'] not in BUILDERS: raise SystemExit(f'unknown shot type {sh["type"]}')
        n = int(round(sh['dur'] * FPS))
        if sh.get('top'): top = sh['top']
        if top not in layers: layers[top] = text_layer(top, style)
        p = sh.get('p', {})
        base = None if sh['type'] == 'chat' else BUILDERS[sh['type']](p, seed=si)
        rng = random.Random(si)
        for k in range(n):
            tt = k / FPS
            if sh['type'] == 'chat' and (k % 3 == 0 or base is None):
                base = shot_chat(p, t_frac=k / max(1, n - 1))
            zoom = (1 + 0.10 * math.exp(-tt * 14)) * (1 + 0.04 * k / n)   # punch-in + slow push
            cw, ch = W / zoom, H / zoom
            jx = jy = 0
            if sh['type'] == 'reaction' and tt < 0.45:
                amp = 26 * (1 - tt / 0.45); jx, jy = rng.uniform(-amp, amp), rng.uniform(-amp, amp)
            x0 = (BW - cw) / 2 + jx; y0 = (BH - ch) / 2 + jy
            fr = base.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
            if tt < 0.05: fr = Image.blend(fr, Image.new('RGB', (W, H), (255, 255, 255)), 0.35)
            fr = fr.convert('RGBA'); fr.alpha_composite(layers[top]); fr = fr.convert('RGB')
            if stills and (k == n // 2 or (sh['type'] == 'chat' and k == n - 1)): fr.save(f'{stills}/shot{si}_{sh["type"]}_{k}.png')
            yield fr
    n = int(round(spec.get('end_dur', 2.6) * FPS))
    for k in range(n):
        fr = end_card(spec['kw'], k / FPS)
        if stills and k == n - 1: fr.save(f'{stills}/end.png')
        yield fr

def render(spec, out, stills=None):
    if stills: os.makedirs(stills, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='meme_')
    wav = f'{tmp}/mix.wav'; total = audio.build_mix(spec, wav)
    # two-pass loudness normalisation to -14 LUFS, -2 dBTP
    meas = subprocess.run(['ffmpeg', '-hide_banner', '-i', wav, '-af', 'loudnorm=I=-14:TP=-2:LRA=11:print_format=json', '-f', 'null', '-'],
                          capture_output=True, text=True).stderr
    js = json.loads(meas[meas.rindex('{'):meas.rindex('}') + 1])
    ln = (f"loudnorm=I=-14:TP=-2:LRA=11:measured_I={js['input_i']}:measured_TP={js['input_tp']}:"
          f"measured_LRA={js['input_lra']}:measured_thresh={js['input_thresh']}:offset={js['target_offset']}:linear=true")
    vid = f'{tmp}/v.mp4'
    ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-profile:v', 'high', vid], stdin=subprocess.PIPE)
    nf = 0
    for fr in frames(spec, stills):
        ff.stdin.write(fr.tobytes()); nf += 1
    ff.stdin.close(); ff.wait()
    if ff.returncode: raise SystemExit('ffmpeg video encode failed')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', vid, '-i', wav, '-af', ln, '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
                    '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-t', f'{nf / FPS:.3f}', '-movflags', '+faststart', out], check=True)
    subprocess.run(['rm', '-rf', tmp])
    return nf / FPS

if __name__ == '__main__':
    spec = json.load(open(sys.argv[1]))
    stills = sys.argv[sys.argv.index('--stills') + 1] if '--stills' in sys.argv else None
    print('duration', render(spec, sys.argv[2], stills))
