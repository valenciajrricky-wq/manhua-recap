"""Manhua recap pipeline: zip of pages -> panels -> recap script (Gemini) -> voice (Kokoro) -> MP4.

Runs the same on Windows (testing) and on a Linux runner (GitHub Actions).
CLI:  python pipeline.py chapter.zip --lang en --aspect 9:16 --out recap.mp4
      python pipeline.py chapter.zip --stage script|voice|video --state DIR ...   (split run for CI)
"""
import os, re, io, json, math, time, random, zipfile, shutil, tempfile, subprocess, argparse, bisect
import numpy as np, cv2, soundfile as sf
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
FPS = 24
SR = 24000
TRANS = 0.35      # transition length (s)
FLASH = 0.26      # white flash length (s)
MAX_RATIO = 3.2   # panels taller than this (h/w) are split; anything taller than the screen box is panned
WIDTH = 1200      # working width for panels (sharp at 1080p)
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash"]  # first that answers wins
GEMINI_TIMEOUT = 150  # seconds per try

# language code -> (kokoro lang, name for the script writer, default voice)
LANGS = {
    "en": ("en-us", "English", "am_onyx"),
    "es": ("es", "Spanish", "em_alex"),
    "fr": ("fr-fr", "French", "ff_siwis"),
    "it": ("it", "Italian", "im_nicola"),
    "pt": ("pt-br", "Brazilian Portuguese", "pm_alex"),
    "hi": ("hi", "Hindi", "hm_omega"),
}


def _log(cb, pct, msg):
    (cb or (lambda p, m: print(f"[{p:3d}%] {m}", flush=True)))(pct, msg)


# ---------------------------------------------------------------- pages
def _natkey(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_pages(src, workdir):
    """Accept a .zip or a folder; return page images (RGB numpy) in reading order."""
    if os.path.isfile(src) and src.lower().endswith(".zip"):
        folder = os.path.join(workdir, "pages")
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                if "__MACOSX" in n or os.path.basename(n).startswith(".") or n.endswith("/"):
                    continue
                if not n.lower().endswith(IMG_EXT):
                    continue
                dest = os.path.normpath(os.path.join(folder, n))
                if not dest.startswith(os.path.normpath(folder)):  # zip-slip guard
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with z.open(n) as fi, open(dest, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
    else:
        folder = src
    files = []
    for d, _, fs in os.walk(folder):
        files += [os.path.join(d, f) for f in fs if f.lower().endswith(IMG_EXT)]
    files.sort(key=lambda p: _natkey(os.path.relpath(p, folder)))
    if not files:
        raise ValueError("No images found in the zip (jpg/png/webp).")
    return [np.array(Image.open(f).convert("RGB")) for f in files]


# ---------------------------------------------------------------- panels
def _content_ranges(gray, axis, min_gap, min_piece, bg=None, thr=4.0, tol=28):
    """Ranges along an axis separated by gutters: flat rows/columns in the page's background colour
    (a flat dark area inside a panel is not a gutter if the page background is white)."""
    uni = gray.std(axis=axis) < thr
    if bg is not None:
        uni &= np.abs(gray.mean(axis=axis) - bg) < tol
    n = len(uni)
    ranges, start, gap = [], None, 0
    for i in range(n):
        if not uni[i]:
            if start is None:
                start = i
            gap = 0
        else:
            gap += 1
            if start is not None and gap >= min_gap:
                ranges.append((start, i - gap + 1))
                start = None
    if start is not None:
        end = n
        while end > start and uni[end - 1]:
            end -= 1
        ranges.append((start, end))
    return [(a, b) for a, b in ranges if b - a >= min_piece]


def _page_bg(g):
    """Gutter colour = median of the page border."""
    return float(np.median(np.concatenate([g[:4].ravel(), g[-4:].ravel(), g[:, :4].ravel(), g[:, -4:].ravel()])))


def _xy_cut(img, depth=0, bg=None):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if bg is None:
        bg = _page_bg(g)
    out = []
    for y0, y1 in _content_ranges(g, 1, 12, 40, bg):
        sub = img[y0:y1]
        cols = _content_ranges(g[y0:y1], 0, 12, 80, bg)
        if len(cols) > 1 and depth < 3:
            for x0, x1 in cols:
                out += _xy_cut(sub[:, x0:x1], depth + 1, bg)
        elif cols:
            out.append(sub[:, cols[0][0]:cols[-1][1]])
    return out


def _vstack(a, b):
    if a.shape[1] != b.shape[1]:
        b = cv2.resize(b, (a.shape[1], max(1, round(b.shape[0] * a.shape[1] / b.shape[1]))), interpolation=cv2.INTER_AREA)
    return np.vstack([a, b])


def _split_tall(p, max_ratio=MAX_RATIO, target=2.2):
    """Only very long strips are split, at the calmest row near each cut point."""
    h, w = p.shape[:2]
    if h <= max_ratio * w:
        return [p]
    n = math.ceil(h / (target * w))
    g = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY).astype(np.int16)
    energy = np.abs(np.diff(g, axis=0)).mean(axis=1)
    cuts, win = [0], int(0.35 * w)
    for k in range(1, n):
        t = k * h // n
        a, b = max(cuts[-1] + 50, t - win), min(h - 50, t + win)
        cuts.append(a + int(np.argmin(energy[a:b])) if b > a else t)
    cuts.append(h)
    return [p[cuts[i]:cuts[i + 1]] for i in range(n)]


def cut_panels(pages, width=WIDTH):
    """Cut every page on its gutters. A panel that runs off the bottom of one page and continues
    at the top of the next (no gutter at the seam) is joined back into one image."""
    out, open_tail = [], False
    for p in pages:
        strip = cv2.resize(p, (width, max(1, round(p.shape[0] * width / p.shape[1]))), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY).astype(np.float32)
        bg = _page_bg(g)
        rows = _content_ranges(g, 1, 12, 40, bg)
        pieces = _xy_cut(strip, 0, bg)
        if not rows or not pieces:
            open_tail = False
            continue
        touches_top, touches_bottom = rows[0][0] <= 2, rows[-1][1] >= strip.shape[0] - 2
        if open_tail and touches_top and out and pieces[0].shape[1] >= width * 0.85:
            out[-1] = _vstack(out[-1], pieces[0])
            pieces = pieces[1:]
            if not pieces:
                open_tail = touches_bottom
                continue
        out += pieces
        open_tail = touches_bottom and pieces[-1].shape[1] >= width * 0.85
    final = []
    for p in out:
        final += _split_tall(p)
    return [p for p in final if p.std() > 6 and p.shape[0] >= 60 and p.shape[1] >= 120]


# ---------------------------------------------------------------- script
PROMPT = """You are writing the voice-over for a fast, gripping YouTube/TikTok manhua recap video.
You get the chapter's panels in reading order, labelled "Panel N".
Write the narration in {lang}. Rules:
- Tell the story in third person, present tense, like a hyped recap channel. Name characters once you know their names.
- Do NOT read the speech bubbles word for word; summarise what happens and why it matters.
- 0 to 2 short sentences per panel. Open panel 1 with a strong hook sentence.
- skip: true for panels that add nothing to the story (blank space, credits, logos, watermarks, translator notes, ads).
- continues: true when the panel is the SAME picture as the panel before it, cut in two (one shot split across pages); give it "" text.
- transition for each panel: "fade" normally, "flash" when the story jumps to a new scene/place/time or a big power moment, "cut" on a sudden hit or shocking reveal.
{context}
Return JSON only: {{"panels": [{{"panel": <number>, "text": "...", "transition": "fade|flash|cut", "skip": false, "continues": false}}]}}"""


def _jpeg(p, max_w=768):
    im = Image.fromarray(p)
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=80)
    return b.getvalue()


def write_script(panels, lang_name, cb=None, chunk=40):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT * 1000))
    models = [m for m in [os.environ.get("GEMINI_MODEL")] + GEMINI_MODELS if m]
    lines = [{"panel": i + 1, "text": "", "transition": "fade", "skip": False, "continues": False} for i in range(len(panels))]
    for s in range(0, len(panels), chunk):
        _log(cb, 20 + int(20 * s / len(panels)), f"Writing script (panels {s + 1}-{min(s + chunk, len(panels))})")
        so_far = " ".join(l["text"] for l in lines[:s] if l["text"])[-3000:]
        ctx = f"Story so far (already narrated, continue from here, no repeats): {so_far}" if so_far else ""
        parts = [PROMPT.format(lang=lang_name, context=ctx)]
        for i, p in enumerate(panels[s:s + chunk]):
            parts += [f"Panel {s + i + 1}:", types.Part.from_bytes(data=_jpeg(p), mime_type="image/jpeg")]
        data, errors = None, []
        for attempt, model in enumerate(models * 2):   # busy or slow model -> next one
            try:
                r = client.models.generate_content(
                    model=model, contents=parts,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.7))
                data = json.loads(r.text)
                break
            except Exception as e:
                errors.append(f"{model}: {str(e)[:120]}")
                _log(cb, 20, f"{model} busy, trying another model")
                time.sleep(min(20, 3 * (attempt + 1)))
        if data is None:
            raise RuntimeError("Gemini unavailable: " + " | ".join(errors[-3:]))
        for item in data.get("panels", []):
            k = int(item.get("panel", 0)) - 1
            if 0 <= k < len(lines):
                tr = item.get("transition", "fade")
                lines[k].update(text=str(item.get("text", "")).strip(),
                                transition=tr if tr in ("fade", "flash", "cut") else "fade",
                                skip=bool(item.get("skip", False)),
                                continues=bool(item.get("continues", False)) and k > 0)
    return lines


def make_plan(panels, lines):
    """Decide which pictures make the video: drop skipped panels, glue split shots back together.
    Returns [{"idx": [panel indexes], "text", "transition"}] - stored, so every stage agrees."""
    plan, heights = [], []
    for i, (p, l) in enumerate(zip(panels, lines)):
        if l.get("skip"):
            continue
        if l.get("continues") and plan and plan[-1]["idx"][-1] == i - 1:
            h = heights[-1] + p.shape[0] * WIDTH / p.shape[1]
            if h <= MAX_RATIO * WIDTH * 1.15:
                plan[-1]["idx"].append(i)
                plan[-1]["text"] = (plan[-1]["text"] + " " + l["text"]).strip()
                heights[-1] = h
                continue
        plan.append({"idx": [i], "text": l["text"], "transition": l.get("transition", "fade")})
        heights.append(p.shape[0] * WIDTH / p.shape[1])
    return plan


def plan_images(panels, plan):
    out = []
    for item in plan:
        img = panels[item["idx"][0]]
        for k in item["idx"][1:]:
            img = _vstack(img, panels[k])
        out.append(img)
    return out


# ---------------------------------------------------------------- voice
_KOKORO = None


def tts(texts, lang, voice, speed=1.05, cb=None):
    global _KOKORO
    from kokoro_onnx import Kokoro
    if _KOKORO is None:
        _KOKORO = Kokoro(os.path.join(ROOT, "models", "kokoro-v1.0.int8.onnx"),
                         os.path.join(ROOT, "models", "voices-v1.0.bin"))
    out = []
    for i, t in enumerate(texts):
        _log(cb, 40 + int(25 * i / max(1, len(texts))), f"Recording voice ({i + 1}/{len(texts)})")
        if not t:
            out.append(np.zeros(0, np.float32))
            continue
        a, sr = _KOKORO.create(t, voice=voice, speed=speed, lang=lang)
        out.append(np.asarray(a, np.float32))
    return out


# ---------------------------------------------------------------- captions
def caption_chunks(text, max_words=5, max_chars=26):
    words, chunks, cur = text.split(), [], []
    for w in words:
        if cur and (len(cur) >= max_words or len(" ".join(cur + [w])) > max_chars):
            chunks.append(" ".join(cur)); cur = []
        cur.append(w)
        if re.search(r"[.!?,;:]$", w) and len(cur) >= 2:
            chunks.append(" ".join(cur)); cur = []
    if cur:
        chunks.append(" ".join(cur))
    merged = []  # never flash a lone word: glue it onto the chunk before it
    for c in chunks:
        if merged and len(c.split()) == 1 and len(merged[-1]) + len(c) < max_chars + 10:
            merged[-1] += " " + c
        else:
            merged.append(c)
    return merged


def _font(size):
    for f in (os.path.join(ROOT, "assets", "Anton-Regular.ttf"), "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _caption_band(text, W, y_center, size):
    """Caption as a horizontal band: (y0, rgb float, alpha float) - only that band gets blended."""
    font = _font(size)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    text = text.upper()
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if probe.textlength(trial, font=font) > W * 0.86 and cur:
            lines.append(cur); cur = w
        else:
            cur = trial
    lines.append(cur)
    lh = int(size * 1.15)
    bh = lh * len(lines) + size
    layer = Image.new("RGBA", (W, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    y = size // 2
    for ln in lines:
        tw = d.textlength(ln, font=font)
        d.text(((W - tw) / 2, y), ln, font=font, fill=(255, 255, 255, 255),
               stroke_width=max(4, size // 11), stroke_fill=(0, 0, 0, 255))
        y += lh
    arr = np.array(layer).astype(np.float32)
    return y_center - bh // 2, arr[..., :3], arr[..., 3:4] / 255.0


# ---------------------------------------------------------------- motion
def _ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Shot:
    """One picture on screen: blurred backdrop + the panel moving inside a box.
    Tall panel -> scrolls top to bottom. Wide panel -> pans sideways. Otherwise a slow zoom."""

    def __init__(self, panel, W, H, vertical, rng):
        h, w = panel.shape[:2]
        if vertical:
            bw, bh, cy = W - 60, int(H * 0.68), int(H * 0.43)
        else:
            bw, bh, cy = int(W * 0.9), H - 120, H // 2 - 20
        # backdrop
        s = max(W / w, H / h)
        bg = cv2.resize(panel, (max(1, math.ceil(w * s / 8)), max(1, math.ceil(h * s / 8))), interpolation=cv2.INTER_AREA)
        bg = cv2.GaussianBlur(bg, (0, 0), 6)
        bg = cv2.resize(bg, (math.ceil(w * s), math.ceil(h * s)), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (bg.shape[0] - H) // 2, (bg.shape[1] - W) // 2
        self.bg = (bg[y0:y0 + H, x0:x0 + W].astype(np.float32) * 0.4).astype(np.uint8)
        self.W, self.H = W, H
        if h / w > (bh / bw) * 1.2:                       # tall: scroll down (only if it would get small)
            self.mode = "vpan"
            sw, sh = bw, round(h * bw / w)
            self.img = cv2.resize(panel, (sw, sh), interpolation=cv2.INTER_AREA)
            self.vw, self.vh = sw, bh
            self.travel = sh - bh
        elif w / h > max((bw / bh) * 1.2, 2.0):           # very wide panorama: pan sideways
            self.mode = "hpan"
            sh = int(min(bh, h * bw / w * 1.7))   # enlarge at most 1.7x past fit-to-width, then pan
            sw = round(w * sh / h)
            self.img = cv2.resize(panel, (sw, sh), interpolation=cv2.INTER_AREA)
            self.vw, self.vh = bw, sh
            self.travel = sw - bw
            self.rev = rng.random() < 0.5
        else:                                             # fits: gentle zoom in or out
            self.mode = "zoom"
            f = min(bw / w, bh / h)
            self.vw, self.vh = max(1, int(w * f)), max(1, int(h * f))
            self.zmax = 1.08
            self.img = cv2.resize(panel, (int(self.vw * self.zmax), int(self.vh * self.zmax)),
                                  interpolation=cv2.INTER_AREA if f * self.zmax < 1 else cv2.INTER_CUBIC)
            self.zin = rng.random() < 0.6
            self.drift = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5))
        self.x, self.y = (W - self.vw) // 2, max(0, cy - self.vh // 2)

    def min_duration(self):
        if self.mode == "vpan":
            return 1.2 + 1.6 * self.travel / self.vh
        if self.mode == "hpan":
            return 1.2 + 1.2 * self.travel / self.vw
        return 0

    def frame(self, t, dur):
        p = _ease((t - 0.15) / max(0.3, dur - 0.3))
        f = self.bg.copy()
        if self.mode == "vpan":
            off = int(round(p * self.travel))
            view = self.img[off:off + self.vh]
        elif self.mode == "hpan":
            off = int(round((1 - p if self.rev else p) * self.travel))
            view = self.img[:, off:off + self.vw]
        else:
            z = 1 + (self.zmax - 1) * (p if self.zin else 1 - p)
            Sh, Sw = self.img.shape[:2]
            cw, ch = Sw / z, Sh / z
            cx = Sw / 2 + self.drift[0] * (Sw - cw) * p
            cy = Sh / 2 + self.drift[1] * (Sh - ch) * p
            x0 = int(max(0, min(Sw - cw, cx - cw / 2)))
            y0 = int(max(0, min(Sh - ch, cy - ch / 2)))
            view = cv2.resize(self.img[y0:y0 + int(ch), x0:x0 + int(cw)], (self.vw, self.vh), interpolation=cv2.INTER_LINEAR)
        f[self.y:self.y + view.shape[0], self.x:self.x + view.shape[1]] = view
        return f


# ---------------------------------------------------------------- transitions
POOL = ["fade", "fade", "slide_left", "slide_left", "slide_up", "slide_right", "wipe", "zoom", "zoom"]


def pick_transitions(hints, seed):
    rng, out, last = random.Random(seed), [], None
    for i, h in enumerate(hints):
        if i == 0:
            out.append("none")
        elif h in ("cut", "flash"):
            out.append(h)
        else:
            choice = rng.choice([p for p in POOL if p != last])
            out.append(choice)
        last = out[-1]
    return out


def blend_transition(kind, A, B, e):
    H, W = A.shape[:2]
    if kind == "fade":
        return (A * (1 - e) + B * e).astype(np.uint8)
    if kind in ("slide_left", "slide_right"):
        dx = int(W * e)
        out = np.empty_like(A)
        if kind == "slide_left":
            out[:, :W - dx], out[:, W - dx:] = A[:, dx:], B[:, :dx]
        else:
            out[:, dx:], out[:, :dx] = A[:, :W - dx], B[:, W - dx:]
        return out
    if kind == "slide_up":
        dy = int(H * e)
        out = np.empty_like(A)
        out[:H - dy], out[H - dy:] = A[dy:], B[:dy]
        return out
    if kind == "wipe":
        soft = 80
        edge = e * (W + soft) - soft
        a = np.clip((edge - np.arange(W)) / soft + 1, 0, 1).astype(np.float32)[None, :, None]
        return (A * (1 - a) + B * a).astype(np.uint8)
    if kind == "zoom":
        s = 1.15 - 0.15 * e
        cw, ch = int(W / s), int(H / s)
        x0, y0 = (W - cw) // 2, (H - ch) // 2
        Bz = cv2.resize(B[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
        return (A * (1 - e) + Bz * e).astype(np.uint8)
    if kind == "flash":
        if e < 0.5:
            k = e * 2
            return (A * (1 - k) + 255 * k).astype(np.uint8)
        k = (e - 0.5) * 2
        return (255 * (1 - k) + B * k).astype(np.uint8)
    return B


# ---------------------------------------------------------------- render
def _music_track(n_samples):
    folder = os.path.join(ROOT, "assets", "music")
    tracks = [os.path.join(folder, f) for f in os.listdir(folder)
              if f.lower().endswith((".mp3", ".wav", ".m4a", ".ogg"))] if os.path.isdir(folder) else []
    if not tracks:
        return None
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", random.choice(tracks), "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    m = np.frombuffer(raw, np.float32)
    if len(m) == 0:
        return None
    m = np.tile(m, math.ceil(n_samples / len(m)))[:n_samples].copy()
    fade = min(len(m), 2 * SR)
    m[-fade:] *= np.linspace(1, 0, fade)
    return m


def render(images, plan, voices, out_path, aspect="9:16", workdir=".", cb=None, seed=0):
    vertical = aspect == "9:16"
    W, H = (1080, 1920) if vertical else (1920, 1080)
    rng = random.Random(seed)
    shots = [Shot(img, W, H, vertical, rng) for img in images]
    kinds = pick_transitions([p["transition"] for p in plan], seed)
    lead, tail, silent = 0.12, 0.30, 1.4
    starts, durs, t = [], [], 0.0
    for shot, v in zip(shots, voices):
        d = lead + len(v) / SR + tail if len(v) else silent
        d = max(d, shot.min_duration())
        starts.append(t); durs.append(d); t += d
    total = t

    # audio first (mixed to WAV, then muxed)
    audio = np.zeros(int(total * SR) + SR, np.float32)
    for st, v in zip(starts, voices):
        i = int((st + lead) * SR)
        audio[i:i + len(v)] += v
    peak = np.abs(audio).max() or 1.0
    audio *= 0.89 / peak
    music = _music_track(len(audio))
    if music is not None:
        audio += music * (0.10 / (np.abs(music).max() or 1.0))
    wav = os.path.join(workdir, "mix.wav")
    sf.write(wav, np.clip(audio, -1, 1), SR, subtype="PCM_16")

    # caption timing per shot
    caps = []
    for st, v, item in zip(starts, voices, plan):
        chunks = caption_chunks(item["text"]) if len(v) else []
        weights = [len(c) + 3 for c in chunks]
        acc, spans = st + lead, []
        for c, wgt in zip(chunks, weights):
            d = (len(v) / SR) * wgt / sum(weights)
            spans.append((acc, acc + d, c)); acc += d
        caps.append(spans)
    cap_y, cap_size = (int(H * 0.86), 92) if vertical else (int(H * 0.88), 72)
    band_cache = {}

    def band(i, k):
        if (i, k) not in band_cache:
            if len(band_cache) > 8:
                band_cache.pop(next(iter(band_cache)))
            band_cache[(i, k)] = _caption_band(caps[i][k][2], W, cap_y, cap_size)
        return band_cache[(i, k)]

    def frame_at(t):
        i = max(0, bisect.bisect_right(starts, t) - 1)
        local = t - starts[i]
        f = shots[i].frame(local, durs[i])
        kind = kinds[i]
        tl = FLASH if kind == "flash" else TRANS
        if i > 0 and kind not in ("cut", "none") and local < tl:
            A = shots[i - 1].frame(durs[i - 1] + local, durs[i - 1])
            f = blend_transition(kind, A, f, _ease(local / tl))
        # caption of the line being spoken (stays up until the next one starts)
        spans, k = caps[i], -1
        for j, (a, _, _) in enumerate(spans):
            if t >= a:
                k = j
        if k >= 0:
            y0, rgb, al = band(i, k)
            y1 = y0 + rgb.shape[0]
            region = f[y0:y1].astype(np.float32)
            f[y0:y1] = (region * (1 - al) + rgb * al).astype(np.uint8)
        return f

    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(total * FPS)
    for fi in range(n):
        if fi % (FPS * 5) == 0:
            _log(cb, 65 + int(34 * fi / n), f"Rendering video {int(fi / FPS)}s / {int(total)}s")
        proc.stdin.write(frame_at(fi / FPS).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return total


# ---------------------------------------------------------------- entry points
def _panels_from(src):
    work = tempfile.mkdtemp(prefix="recap_")
    try:
        ps = cut_panels(load_pages(src, work))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not ps:
        raise ValueError("Could not find any panels.")
    return ps


def _write_script_txt(out_path, plan):
    with open(os.path.splitext(out_path)[0] + "_script.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(p["text"] for p in plan if p["text"]))


def make_recap(src, out_path, lang="en", aspect="9:16", voice=None, speed=1.05, cb=None, script=None):
    kl, lang_name, default_voice = LANGS.get(lang, LANGS["en"])
    _log(cb, 2, "Reading pages and finding panels")
    panels = _panels_from(src)
    _log(cb, 18, f"Found {len(panels)} panels")
    lines = script or write_script(panels, lang_name, cb)
    plan = make_plan(panels, lines)
    voices = tts([p["text"] for p in plan], kl, voice or default_voice, speed, cb)
    work = tempfile.mkdtemp(prefix="recap_")
    try:
        total = render(plan_images(panels, plan), plan, voices, out_path, aspect, work, cb, seed=len(panels))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    _write_script_txt(out_path, plan)
    _log(cb, 100, f"Done - {total:.0f}s video, {len(plan)} shots from {len(panels)} panels")
    return out_path


def run_stage(stage, src, out_path, state_dir, lang="en", aspect="9:16", voice=None, speed=1.05):
    """make_recap split in three so a CI runner can show Script / Voice / Video as separate steps."""
    kl, lang_name, default_voice = LANGS.get(lang, LANGS["en"])
    os.makedirs(state_dir, exist_ok=True)
    pj, vz = os.path.join(state_dir, "plan.json"), os.path.join(state_dir, "voices.npz")
    if stage == "script":
        panels = _panels_from(src)
        _log(None, 18, f"Found {len(panels)} panels")
        plan = make_plan(panels, write_script(panels, lang_name))
        json.dump({"n": len(panels), "plan": plan}, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
        _log(None, 40, f"{len(plan)} shots planned")
    elif stage == "voice":
        plan = json.load(open(pj, encoding="utf-8"))["plan"]
        voices = tts([p["text"] for p in plan], kl, voice or default_voice, speed)
        np.savez(vz, *voices)
    elif stage == "video":
        saved = json.load(open(pj, encoding="utf-8"))
        panels = _panels_from(src)
        if len(panels) != saved["n"]:
            raise RuntimeError("Panel count changed between stages")
        z = np.load(vz)
        voices = [z[f"arr_{i}"] for i in range(len(z.files))]
        work = tempfile.mkdtemp(prefix="recap_")
        try:
            total = render(plan_images(panels, saved["plan"]), saved["plan"], voices, out_path, aspect, work, seed=saved["n"])
        finally:
            shutil.rmtree(work, ignore_errors=True)
        _write_script_txt(out_path, saved["plan"])
        _log(None, 100, f"Done - {total:.0f}s video")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="recap.mp4")
    ap.add_argument("--lang", default="en", choices=list(LANGS))
    ap.add_argument("--aspect", default="9:16", choices=["9:16", "16:9"])
    ap.add_argument("--voice")
    ap.add_argument("--speed", type=float, default=1.05)
    ap.add_argument("--stage", choices=["all", "script", "voice", "video"], default="all")
    ap.add_argument("--state", default="state")
    a = ap.parse_args()
    if a.stage == "all":
        make_recap(a.src, a.out, a.lang, a.aspect, a.voice, a.speed)
    else:
        run_stage(a.stage, a.src, a.out, a.state, a.lang, a.aspect, a.voice, a.speed)
