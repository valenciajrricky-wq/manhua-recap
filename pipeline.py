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
TRANS = 0.25      # quick fade length (s)
FLASH = 0.26      # white flash length (s)
MAX_RATIO = 2.2   # stitched pieces may not get taller than this (h/w)
SPLIT_RATIO = 2.0 # taller panels are cut into pieces (no scrolling), each about SPLIT_TARGET tall
SPLIT_TARGET = 1.5
CHAPTER_CARDS = False  # "Chapter N" text cards between merged chapters (off: the user prefers art over on-screen text)
WIDTH = 1200      # working width for panels (sharp at 1080p)
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite"]  # first that answers wins
GEMINI_TIMEOUT = 150  # seconds per try
MOODS4 = ("emotional", "tense", "mysterious", "hopeful")

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


def _extract(zpath, folder):
    """Extract images (and nested chapter archives) from a zip/cbz, with a zip-slip guard."""
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if "__MACOSX" in n or os.path.basename(n).startswith(".") or n.endswith("/"):
                continue
            if not n.lower().endswith(IMG_EXT + (".zip", ".cbz")):
                continue
            dest = os.path.normpath(os.path.join(folder, n))
            if not dest.startswith(os.path.normpath(folder)):
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(n) as fi, open(dest, "wb") as fo:
                shutil.copyfileobj(fi, fo)


def chapter_label(name):
    """'003_Emotion Chip - Ch. 3 Where It Began.cbz' -> ('Chapter 3', 'Where It Began')."""
    base = re.sub(r"(?i)\.(zip|cbz)$", "", os.path.basename(name))
    base = re.sub(r"^\d{3}_", "", base).replace("_", " ").strip()
    m = re.search(r"(?i)\b(?:chapter|chap|ch|episode|ep)[\s.\-]*(\d+(?:\.\d+)?)\s*[:\-–]?\s*(.*)$", base)
    if m:
        return f"Chapter {m.group(1)}", m.group(2).replace("-", " ").strip(" -:")
    m = re.search(r"(\d+(?:\.\d+)?)\s*$", base)
    return (f"Chapter {m.group(1)}", "") if m else (base, "")


def load_chapters(src, workdir):
    """Pages in reading order + which chapter each page belongs to.
    A zip that contains chapter archives (.cbz/.zip, sent by the app's "Combine" option) is several
    chapters in a row; anything else is one chapter."""
    if os.path.isfile(src) and src.lower().endswith((".zip", ".cbz")):
        folder = os.path.join(workdir, "pages")
        _extract(src, folder)
    else:
        folder = src
    nested = []
    for d, _, fs in os.walk(folder):
        nested += [os.path.join(d, f) for f in fs if f.lower().endswith((".zip", ".cbz"))]
    nested.sort(key=lambda p: _natkey(os.path.relpath(p, folder)))
    groups, names = [], []
    if nested:
        for k, arc in enumerate(nested):
            sub = os.path.join(workdir, f"ch{k:03d}")
            try:
                _extract(arc, sub)
            except zipfile.BadZipFile:
                continue
            groups.append(sub)
            names.append(os.path.basename(arc))
    else:
        groups, names = [folder], [""]
    pages, chap = [], []
    for ci, g in enumerate(groups):
        files = []
        for d, _, fs in os.walk(g):
            files += [os.path.join(d, f) for f in fs if f.lower().endswith(IMG_EXT)]
        files.sort(key=lambda p: _natkey(os.path.relpath(p, g)))
        for f in files:
            pages.append(np.array(Image.open(f).convert("RGB")))
            chap.append(ci)
    if not pages:
        raise ValueError("No images found in the zip (jpg/png/webp).")
    return pages, chap, names


def load_pages(src, workdir):
    """Accept a .zip or a folder; return page images (RGB numpy) in reading order."""
    return load_chapters(src, workdir)[0]


# ---------------------------------------------------------------- panels
# A panel is a list of segments [page, x0, y0, x1, y1] on the resized page strips. Keeping the
# boxes (not just pixels) lets us stitch pieces back by cutting the joined area from the page.
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


def _xy_boxes(g, bg, ox=0, oy=0, depth=0):
    out = []
    for y0, y1 in _content_ranges(g, 1, 12, 40, bg):
        sub = g[y0:y1]
        cols = _content_ranges(sub, 0, 12, 80, bg)
        if len(cols) > 1 and depth < 3:
            for x0, x1 in cols:
                out += _xy_boxes(sub[:, x0:x1], bg, ox + x0, oy + y0, depth + 1)
        elif cols:
            out.append((ox + cols[0][0], oy + y0, ox + cols[-1][1], oy + y1))
    return out


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _inter(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _gap(a, b):
    dx = max(0, b[0] - a[2], a[0] - b[2])
    dy = max(0, b[1] - a[3], a[1] - b[3])
    return dx + dy


def _ink(g, b, bg):
    r = g[b[1]:b[3], b[0]:b[2]]
    return float((np.abs(r - bg) > 40).mean()) if r.size else 0.0


def _stitch_fragments(boxes, g, bg, width):
    """Leftover pieces (a lone speech bubble, a sound effect, a sliver of a panel) are joined to the
    nearest panel on the same page; anything the joined area covers is absorbed so nothing shows twice."""
    ref = width * width * 1.4
    def white(b):
        r = g[b[1]:b[3], b[0]:b[2]]
        return float((r > 235).mean()) if r.size else 0.0
    def frag(b):
        a = _area(b)
        return (a < 0.06 * ref or (a < 0.14 * ref and _ink(g, b, bg) < 0.45)
                or (a < 0.30 * ref and white(b) > 0.62))      # a big speech bubble on its own
    boxes, stuck = list(boxes), set()
    while len(boxes) > 1:
        i = next((k for k, b in enumerate(boxes) if frag(b) and b not in stuck), None)
        if i is None:
            break
        b = boxes[i]
        cands = [k for k in (i - 1, i + 1) if 0 <= k < len(boxes)]
        j = min(cands, key=lambda k: _gap(b, boxes[k]))
        u = _union(b, boxes[j])
        if (u[3] - u[1]) > MAX_RATIO * (u[2] - u[0]):
            stuck.add(b)
            continue
        first = min(i, j)
        boxes = [o for k, o in enumerate(boxes) if k not in (i, j) and _inter(o, u) < 0.6 * _area(o)]
        boxes.insert(min(first, len(boxes)), u)
    return boxes


def _split_rows(img, max_ratio=SPLIT_RATIO, target=SPLIT_TARGET):
    """Only very long strips are split, at the calmest row near each cut point. Returns row ranges."""
    h, w = img.shape[:2]
    if h <= max_ratio * w:
        return [(0, h)]
    n = math.ceil(h / (target * w))
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.int16)
    energy = np.abs(np.diff(g, axis=0)).mean(axis=1)
    cuts, win = [0], int(0.35 * w)
    for k in range(1, n):
        t = k * h // n
        a, b = max(cuts[-1] + 50, t - win), min(h - 50, t + win)
        cuts.append(a + int(np.argmin(energy[a:b])) if b > a else t)
    cuts.append(h)
    return [(cuts[i], cuts[i + 1]) for i in range(n)]


def _vstack(a, b):
    if a.shape[1] != b.shape[1]:
        b = cv2.resize(b, (a.shape[1], max(1, round(b.shape[0] * a.shape[1] / b.shape[1]))), interpolation=cv2.INTER_AREA)
    return np.vstack([a, b])


def panel_image(strips, segs):
    img = None
    for pi, x0, y0, x1, y1 in segs:
        crop = strips[pi][y0:y1, x0:x1]
        img = crop if img is None else _vstack(img, crop)
    return img


def merge_segs(a, b):
    """Join two panels: pieces on the same page become one box (cut from the page), others stack."""
    out = [list(s) for s in a]
    for s in b:
        for o in out:
            if o[0] == s[0]:
                o[1:] = _union(o[1:], s[1:])
                break
        else:
            out.append(list(s))
    return sorted(out, key=lambda s: (s[0], s[2]))


def _ratio(segs):
    w0 = segs[0][3] - segs[0][1]
    return sum((s[4] - s[2]) * w0 / max(1, s[3] - s[1]) for s in segs) / max(1, w0)


def cut_panels(pages, width=WIDTH, chapters=None):
    """Returns (strips, panels). Cuts every page on its gutters, stitches fragments back onto their
    panel, and joins a panel that runs off the bottom of one page onto the top of the next."""
    strips, panels, open_tail = [], [], False
    for pi, p in enumerate(pages):
        if chapters and pi > 0 and chapters[pi] != chapters[pi - 1]:
            open_tail = False
        strip = cv2.resize(p, (width, max(1, round(p.shape[0] * width / p.shape[1]))), interpolation=cv2.INTER_AREA)
        strips.append(strip)
        g = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY).astype(np.float32)
        bg, H = _page_bg(g), strip.shape[0]
        boxes = _xy_boxes(g, bg)
        if not boxes:
            open_tail = False
            continue
        boxes = _stitch_fragments(boxes, g, bg, width)
        full = lambda b: b[2] - b[0] >= width * 0.85
        segs = [[[pi, *b]] for b in boxes]
        if open_tail and panels and full(boxes[0]) and boxes[0][1] <= 2:
            panels[-1] = panels[-1] + segs[0]
            segs, boxes = segs[1:], boxes[1:]
            if not segs:
                open_tail = True
                continue
        panels += segs
        open_tail = full(boxes[-1]) and boxes[-1][3] >= H - 2
    final = []
    expanded = []
    for segs in panels:   # a cross-page panel too tall to show whole goes back to its per-page pieces
        expanded += [[s] for s in segs] if len(segs) > 1 and _ratio(segs) > SPLIT_RATIO else [segs]
    for segs in expanded:
        if len(segs) == 1:
            pi, x0, y0, x1, y1 = segs[0]
            for a, b in _split_rows(strips[pi][y0:y1, x0:x1]):
                final.append([[pi, x0, y0 + a, x1, y0 + b]])
        else:
            final.append(segs)
    keep = []
    for segs in final:
        img = panel_image(strips, segs)
        if img is not None and img.std() > 6 and img.shape[0] >= 60 and img.shape[1] >= 120:
            keep.append(segs)
    return strips, keep


# ---------------------------------------------------------------- script
PROMPT = """You are writing the voice-over for a fast, gripping YouTube/TikTok manhua recap video.
You get the chapter's panels in reading order, labelled "Panel N".
Write the narration in {lang}. Rules:
- Tell the story in third person, present tense, like a hyped recap channel. Name characters once you know their names.
- Do NOT read the speech bubbles word for word; summarise what happens and why it matters.
{density}
- skip: true for panels that add nothing to the story (blank space, credits, logos, watermarks, translator notes, ads).
- The video should show the ART, not text. Also set skip: true for panels that are mostly TEXT with little or no drawing: narration/caption boxes on a plain background, black or white pages with only words, title or chapter-heading pages, a speech bubble on its own, a sound effect on its own. Put what that text says into the narration of the nearest drawn panel instead, so nothing of the story is lost.
- continues: true when the panel belongs with the panel right before it - the same picture cut in two, or a speech bubble / sound effect / small inset cut off from it. They will be shown together as one picture; give it "" text and put the narration on the first one.
- transition for each panel: "fade" normally, "flash" when the story jumps to a new scene/place/time or a big power moment, "cut" on a sudden hit or shocking reveal.
- mood: the overall feel of this chapter, and for EVERY panel the feel of that moment, for background music that follows the scenes: emotional, tense, mysterious or hopeful.
{context}
Return JSON only: {{"mood": "emotional|tense|mysterious|hopeful", "panels": [{{"panel": <number>, "text": "...", "transition": "fade|flash|cut", "skip": false, "continues": false, "mood": "emotional|tense|mysterious|hopeful"}}]}}"""


def _jpeg(p, max_w=640):
    im = Image.fromarray(p)
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=80)
    return b.getvalue()


DENSITY = {
    "more": ("- Narrate EVERY panel that shows the story with 1 to 3 sentences (about 20-40 words): what happens, "
             "what the characters say (retold in your own words) and how they feel. Only blank/credit panels stay silent. "
             "Open panel 1 with a strong hook sentence."),
    "normal": "- 0 to 2 short sentences per panel. Open panel 1 with a strong hook sentence.",
}


def _ask_gemini(client, types, models, parts, cb):
    """One JSON answer from the first model that gives a usable one. Blank answers (overload, safety,
    out of room) count as failures and are retried with backoff across models."""
    safety = [types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH") for c in (
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.7,
                                      max_output_tokens=24000, safety_settings=safety)
    errors = []
    for attempt, model in enumerate(models * 3):
        try:
            r = client.models.generate_content(model=model, contents=parts, config=cfg)
            if not r.text:
                why = r.candidates[0].finish_reason if r.candidates else r.prompt_feedback
                raise RuntimeError(f"empty answer ({why})")
            return json.loads(r.text), errors
        except Exception as e:
            errors.append(f"{model}: {str(e)[:140]}")
            _log(cb, 20, f"{model} didn't answer ({str(e)[:60]}), trying again")
            time.sleep(min(30, 4 * (attempt + 1)))
    return None, errors


def write_script(images, lang_name, cb=None, chunk=40, title="", detail="more", starts=None):
    """Returns (lines, mood)."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT * 1000))
    models = [m for m in [os.environ.get("GEMINI_MODEL")] + GEMINI_MODELS if m]
    lines = [{"panel": i + 1, "text": "", "transition": "fade", "skip": False, "continues": False} for i in range(len(images))]
    moods = []
    todo = [(s, min(s + chunk, len(images))) for s in range(0, len(images), chunk)]
    while todo:
        s, e = todo.pop(0)
        _log(cb, 20 + int(20 * s / len(images)), f"Writing script (panels {s + 1}-{e})")
        so_far = " ".join(l["text"] for l in lines[:s] if l["text"])[-3000:]
        ctx = f"Story so far (already narrated, continue from here, no repeats): {so_far}" if so_far else ""
        if title:
            ctx = (f"This is: {title} (series name, then chapter). Mention the series by name in the opening hook "
                   "if it fits; never invent a different title.\n") + ctx
        if starts and len(starts) > 1:
            ctx = ("This video joins several consecutive chapters into ONE continuous story. At a panel marked "
                   "START OF CHAPTER, carry the story on naturally (a short bridge like 'And in the next chapter...' "
                   "is fine) - never restart with a new opening hook or recap what was already said.\n") + ctx
        parts = [PROMPT.format(lang=lang_name, context=ctx, density=DENSITY.get(detail, DENSITY["more"]))]
        for i, p in enumerate(images[s:e]):
            mark = f" (START OF {starts[s + i]})" if starts and (s + i) in starts and len(starts) > 1 else ""
            parts += [f"Panel {s + i + 1}{mark}:", types.Part.from_bytes(data=_jpeg(p), mime_type="image/jpeg")]
        data, errors = _ask_gemini(client, types, models, parts, cb)
        if data is None:
            if e - s > 8:                       # still failing: try the batch in two halves
                mid = (s + e) // 2
                todo[:0] = [(s, mid), (mid, e)]
                _log(cb, 20, f"Splitting panels {s + 1}-{e} into two smaller requests")
                continue
            raise RuntimeError("Gemini unavailable: " + " | ".join(errors[-3:]))
        if data.get("mood") in MOODS4:
            moods.append(data["mood"])
        for item in data.get("panels", []):
            k = int(item.get("panel", 0)) - 1
            if s <= k < e:
                tr = item.get("transition", "fade")
                lines[k].update(text=str(item.get("text", "")).strip(),
                                transition=tr if tr in ("fade", "flash", "cut") else "fade",
                                skip=bool(item.get("skip", False)),
                                continues=bool(item.get("continues", False)) and k > 0,
                                mood=item.get("mood") if item.get("mood") in MOODS4 else None)
    mood = max(set(moods), key=moods.count) if moods else "emotional"
    return lines, mood


def make_plan(panels, lines, pchap=None, names=None):
    """Decide which pictures make the video: drop skipped panels, stitch pieces Gemini says belong
    together, absorb panels already inside a stitched area. Stored in the state so every stage agrees."""
    plan, shown = [], set()
    multi = bool(names) and len(names) > 1 and CHAPTER_CARDS
    for i, (segs, l) in enumerate(zip(panels, lines)):
        if l.get("skip"):
            continue
        ch = pchap[i] if pchap else 0
        if multi and ch not in shown:
            shown.add(ch)
            title, sub = chapter_label(names[ch])
            plan.append({"segs": [], "last": -1, "text": "", "transition": "flash", "mood": None,
                         "card": title, "sub": sub, "chapter": ch})
        if l.get("continues") and plan and plan[-1]["last"] == i - 1 and plan[-1].get("chapter", ch) == ch:
            m = merge_segs(plan[-1]["segs"], segs)
            if _ratio(m) <= MAX_RATIO * 1.15:
                plan[-1].update(segs=m, last=i, text=(plan[-1]["text"] + " " + l["text"]).strip())
                continue
        plan.append({"segs": [list(s) for s in segs], "last": i, "text": l["text"],
                     "transition": l.get("transition", "fade"), "mood": l.get("mood"), "chapter": ch})
    # a stitched area can now contain another planned panel: show it once
    out = []
    for item in plan:
        host = next((o for o in out if len(item["segs"]) == 1 and any(
            s[0] == item["segs"][0][0] and _inter(s[1:], item["segs"][0][1:]) >= 0.6 * _area(item["segs"][0][1:])
            for s in o["segs"])), None)
        if host is not None:
            host["text"] = (host["text"] + " " + item["text"]).strip()
        else:
            out.append(item)
    return out


def card_image(title, sub=""):
    """A chapter title card: big 'CHAPTER 12' on a dark background, chapter name underneath."""
    W, H = 1200, 900
    y = np.linspace(0, 1, H)[:, None]
    bg = (np.array([18, 10, 24]) * (1 - y) + np.array([4, 4, 8]) * y).astype(np.uint8)
    im = Image.fromarray(np.repeat(bg[:, None, :], W, axis=1).reshape(H, W, 3))
    d = ImageDraw.Draw(im)
    big, small = _font(170), _font(64)
    t = title.upper()
    tw = d.textlength(t, font=big)
    top = H * 0.28
    d.text(((W - tw) / 2, top), t, font=big, fill=(255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0))
    bottom = d.textbbox(((W - tw) / 2, top), t, font=big, stroke_width=6)[3]
    d.rectangle([W / 2 - 90, bottom + 22, W / 2 + 90, bottom + 30], fill=(255, 77, 109))
    if sub:
        s = sub.upper()[:40]
        sw = d.textlength(s, font=small)
        d.text(((W - sw) / 2, bottom + 52), s, font=small, fill=(220, 220, 225))
    return np.array(im)


def plan_images(strips, plan):
    return [card_image(item["card"], item.get("sub", "")) if item.get("card") else panel_image(strips, item["segs"])
            for item in plan]


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
    """One picture on screen, recap-channel style: the whole panel, as big as the screen allows (full height
    in 16:9), over a bright blurred copy of itself, with a slow subtle zoom. No scrolling or panning."""

    def __init__(self, panel, W, H, vertical, rng, captions=False):
        h, w = panel.shape[:2]
        # the panel always leaves room for the blurred backdrop at the sides, and grows a little (zoom) without
        # ever being cropped, so speech bubbles and text at the edges stay readable
        self.zmax = 1.04
        if vertical:
            bw, bh, cy = (W - 60, int(H * 0.66), int(H * 0.43)) if captions else (int(W * 0.94), int(H * 0.80), H // 2)
        else:
            bw, bh, cy = int(W * 0.80), int(H / self.zmax), H // 2
        # backdrop: the same panel, filling the frame, heavily blurred and only slightly dimmed
        s = max(W / w, H / h)
        bg = cv2.resize(panel, (max(1, math.ceil(w * s / 12)), max(1, math.ceil(h * s / 12))), interpolation=cv2.INTER_AREA)
        bg = cv2.GaussianBlur(bg, (0, 0), 5)
        bg = cv2.resize(bg, (math.ceil(w * s), math.ceil(h * s)), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (bg.shape[0] - H) // 2, (bg.shape[1] - W) // 2
        self.bg = (bg[y0:y0 + H, x0:x0 + W].astype(np.float32) * 0.82).astype(np.uint8)
        self.W, self.H = W, H
        self.mode = "zoom"
        f = min(bw / w, bh / h)
        self.vw, self.vh = max(1, int(w * f)), max(1, int(h * f))
        self.img = cv2.resize(panel, (int(self.vw * self.zmax), int(self.vh * self.zmax)),
                              interpolation=cv2.INTER_AREA if f * self.zmax < 1 else cv2.INTER_CUBIC)
        self.zin = rng.random() < 0.7
        self.cy = cy

    def min_duration(self):
        return 0

    def frame(self, t, dur):
        p = _ease((t - 0.1) / max(0.3, dur - 0.2))
        f = self.bg.copy()
        z = 1 + (self.zmax - 1) * (p if self.zin else 1 - p)
        tw, th = int(self.vw * z), int(self.vh * z)
        view = cv2.resize(self.img, (tw, th), interpolation=cv2.INTER_AREA)
        x, y = (self.W - tw) // 2, max(0, self.cy - th // 2)
        view = view[:self.H - y, :self.W - x]
        f[y:y + view.shape[0], x:x + view.shape[1]] = view
        return f


# ---------------------------------------------------------------- transitions
POOL = ["cut", "cut", "cut", "fade", "fade"]   # like a recap channel: mostly straight cuts, some quick fades


def pick_transitions(hints, seed):
    rng, out = random.Random(seed), []
    for i, h in enumerate(hints):
        out.append("none" if i == 0 else "cut" if h == "cut" else rng.choice(POOL))
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
USED_MUSIC = []   # tracks used in the last render, for the credit line


def _decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def _locked(v):
    """Series music entry -> {mood: track}. Old entries were a single track path."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and "/" in v:
        return {v.split("/")[0]: v}
    return {}


def series_of(title, known=()):
    """Series name from "Series - Chapter 12"; else a known series whose name appears in the title."""
    t = (title or "").strip()
    if " - " in t:
        return t.split(" - ")[0].strip()
    flat = re.sub(r"[-_.]+", " ", t).lower()
    return next((k for k in known if k.lower() in flat), "")


def _music_track(n_samples, mood, seed, locked=None):
    """Background music for the chapter's mood: tracks in assets/music/<mood>/ first (free/royalty-free
    MP3s), then any in assets/music/, else music written in code. Several tracks are crossfaded to fill."""
    import music as gen
    rng = random.Random(seed)
    base = os.path.join(ROOT, "assets", "music")
    ext = (".mp3", ".wav", ".m4a", ".ogg")
    tracks = []
    for folder in (os.path.join(base, mood), base):
        if os.path.isdir(folder):
            tracks = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(ext)]
        if tracks:
            break
    if not tracks:
        return gen.make_music(n_samples / SR + 0.1, mood, seed)[:n_samples]
    rng.shuffle(tracks)
    lock = os.path.join(base, locked) if locked else None
    if lock and os.path.exists(lock):
        # same series as an earlier video: start with its track, then that track's mood folder
        folder = os.path.dirname(lock)
        rest = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(ext)]
        rng.shuffle(rest)
        tracks = [lock] + [t for t in rest if os.path.normcase(t) != os.path.normcase(lock)]
    out, xf, k = np.zeros(0, np.float32), 2 * SR, 0
    USED_MUSIC.clear()
    while len(out) < n_samples and k < 50:
        USED_MUSIC.append(tracks[k % len(tracks)])
        m = _decode(tracks[k % len(tracks)]); k += 1
        if len(m) < SR:
            continue
        m /= (np.sqrt((m ** 2).mean()) or 1.0)
        if len(out) >= xf:
            out[-xf:] = out[-xf:] * np.linspace(1, 0, xf) + m[:xf] * np.linspace(0, 1, xf)
            out = np.concatenate([out, m[xf:]])
        else:
            out = np.concatenate([out, m])
    out = np.pad(out, (0, max(0, n_samples - len(out))))[:n_samples]
    fi, fo = min(len(out), 2 * SR), min(len(out), 3 * SR)
    out[:fi] *= np.linspace(0, 1, fi); out[-fo:] *= np.linspace(1, 0, fo)
    return out.astype(np.float32)


MOODS4 = ("emotional", "tense", "mysterious", "hopeful")
MIN_SCENE = 18.0      # seconds a music section lasts at least, so the score doesn't flip on one panel
SCENE_XF = 2.5        # crossfade between scene tracks (s)
CHOSEN_MUSIC = {}     # mood -> track used in the last render (relative to assets/music)


def _scene_music(starts, durs, moods, n, default_mood, seed, locked=None):
    """Background music that follows the story: a track per mood, switching (crossfade) when the scene's
    mood changes. A series keeps the same track for each mood across chapters (locked); a track that comes
    back later in the video continues where it stopped instead of restarting."""
    import music as gen
    rng = random.Random(seed)
    base = os.path.join(ROOT, "assets", "music")
    ext = (".mp3", ".wav", ".m4a", ".ogg")
    moods = [m if m in MOODS4 else default_mood for m in moods]
    segs = []
    for st, d, m in zip(starts, durs, moods):
        if segs and (m == segs[-1][0] or st - segs[-1][1] < MIN_SCENE):
            segs[-1][2] = st + d
        else:
            segs.append([m, st, st + d])
    if len(segs) > 1 and segs[-1][2] - segs[-1][1] < MIN_SCENE:
        segs[-2][2] = segs[-1][2]
        segs.pop()
    chosen = dict(locked or {})
    out, xf = np.zeros(n, np.float32), int(SCENE_XF * SR)
    cache, pos = {}, {}
    USED_MUSIC.clear()
    for i, (m, a, b) in enumerate(segs):
        rel = chosen.get(m)
        full = os.path.join(base, rel) if rel else None
        if not full or not os.path.exists(full):
            folder = os.path.join(base, m)
            cands = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(ext)] \
                if os.path.isdir(folder) else []
            full = rng.choice(cands) if cands else None
            if full:
                chosen[m] = os.path.relpath(full, base).replace(os.sep, "/")
        last = i == len(segs) - 1
        i0 = max(0, int(a * SR) - (xf if i > 0 else 0))
        i1 = n if last else min(n, int(b * SR) + xf)
        L = i1 - i0
        if L <= 0:
            continue
        if full:
            if full not in cache:
                t = _decode(full)
                cache[full] = t / (np.sqrt((t ** 2).mean()) or 1.0)
            trk = cache[full]
            off = pos.get(full, 0)
            seg = trk[(off + np.arange(L)) % len(trk)]
            pos[full] = (off + L) % len(trk)
            USED_MUSIC.append(full)
        else:
            seg = gen.make_music(L / SR + 0.1, m, seed + i)[:L]
            seg = seg / (np.sqrt((seg ** 2).mean()) or 1.0)
        env = np.ones(L, np.float32)
        if i > 0:
            k = min(xf, L)
            env[:k] = np.linspace(0, 1, k)
        if not last:
            k = min(xf, L)
            env[-k:] *= np.linspace(1, 0, k)
        out[i0:i1] += seg * env
    fi, fo = min(n, 2 * SR), min(n, 3 * SR)
    out[:fi] *= np.linspace(0, 1, fi)
    out[-fo:] *= np.linspace(1, 0, fo)
    CHOSEN_MUSIC.clear()
    CHOSEN_MUSIC.update(chosen)
    return out


def render(images, plan, voices, out_path, aspect="9:16", workdir=".", cb=None, seed=0, mood="emotional", locked=None,
           captions=False):
    vertical = aspect == "9:16"
    W, H = (1080, 1920) if vertical else (1920, 1080)
    rng = random.Random(seed)
    shots = [Shot(img, W, H, vertical, rng, captions) for img in images]
    kinds = pick_transitions([p["transition"] for p in plan], seed)
    lead, tail, silent = 0.12, 0.30, 1.4
    starts, durs, t = [], [], 0.0
    for shot, v, item in zip(shots, voices, plan):
        d = lead + len(v) / SR + tail if len(v) else (2.0 if item.get("card") else silent)
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
    # subtle music: ~16 dB under the voice, dipping a further ~6 dB while the narrator talks
    import music as gen
    bgm = _scene_music(starts, durs, [p.get("mood") for p in plan], len(audio), mood, seed, locked)
    voiced = np.abs(audio) > 0.02
    v_rms = np.sqrt((audio[voiced] ** 2).mean()) if voiced.any() else 0.1
    m_rms = np.sqrt((bgm ** 2).mean()) or 1.0
    bgm = gen.duck(bgm * (v_rms * 10 ** (-16 / 20) / m_rms), audio, depth=0.5)
    audio = audio + bgm
    audio *= min(1.0, 0.97 / (np.abs(audio).max() or 1.0))
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
        spans, k = (caps[i] if captions else []), -1
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
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "superfast", "-crf", "21",
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


def make_wide(src, dst):
    """YouTube 16:9 copy of a finished 9:16 video: the vertical video in the middle, the sides filled with a
    blurred, stretched copy of itself (the usual recap-channel look). Audio is copied, nothing re-rendered."""
    fc = ("[0:v]split[a][b];"
          "[a]scale=480:270:force_original_aspect_ratio=increase,crop=480:270,gblur=sigma=12,"
          "scale=1920:1080,eq=brightness=-0.04[bg];"
          "[b]scale=-2:1080[fg];[bg][fg]overlay=(W-w)/2:0,format=yuv420p")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-filter_complex", fc, "-c:v", "libx264",
                    "-preset", "superfast", "-crf", "21", "-c:a", "copy", "-movflags", "+faststart", dst], check=True)


def _wide_name(out_path):
    root, ext = os.path.splitext(out_path)
    return root + "_169" + ext


# ---------------------------------------------------------------- entry points
def _panels_from(src, with_chapters=False):
    work = tempfile.mkdtemp(prefix="recap_")
    try:
        pages, chap, names = load_chapters(src, work)
        strips, ps = cut_panels(pages, chapters=chap)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not ps:
        raise ValueError("Could not find any panels.")
    if not with_chapters:
        return strips, ps
    pchap = [chap[segs[0][0]] for segs in ps]
    starts, seen = {}, set()
    for i, c in enumerate(pchap):
        if c not in seen:
            seen.add(c)
            starts[i] = chapter_label(names[c])[0] if names[c] else "Chapter"
    return strips, ps, pchap, names, starts


def _music_credits():
    """Credit lines for the tracks used (the free licence asks for this in the video description)."""
    path = os.path.join(ROOT, "assets", "music", "CREDITS.txt")
    if not USED_MUSIC or not os.path.exists(path):
        return []
    table = dict(l.rstrip("\n").split("\t", 1) for l in open(path, encoding="utf-8") if "\t" in l)
    base = os.path.join(ROOT, "assets", "music")
    out = []
    for t in dict.fromkeys(USED_MUSIC):
        key = os.path.relpath(t, base).replace(os.sep, "/")
        if key in table:
            out.append(table[key])
    return out


def _write_script_txt(out_path, plan):
    with open(os.path.splitext(out_path)[0] + "_script.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(p["text"] for p in plan if p["text"]))
        credits = _music_credits()
        if credits:
            f.write("\n\n--- Music (paste in your video description) ---\n" + "\n".join(credits) + "\n")


def make_recap(src, out_path, lang="en", aspect="9:16", voice=None, speed=1.05, cb=None, script=None):
    kl, lang_name, default_voice = LANGS.get(lang, LANGS["en"])
    _log(cb, 2, "Reading pages and finding panels")
    strips, panels, pchap, names, starts = _panels_from(src, True)
    _log(cb, 18, f"Found {len(panels)} panels in {len(names)} chapter(s)")
    if script:
        lines, mood = script, "emotional"
    else:
        lines, mood = write_script([panel_image(strips, p) for p in panels], lang_name, cb, starts=starts)
    plan = make_plan(panels, lines, pchap, names)
    voices = tts([p["text"] for p in plan], kl, voice or default_voice, speed, cb)
    work = tempfile.mkdtemp(prefix="recap_")
    try:
        total = render(plan_images(strips, plan), plan, voices, out_path, "9:16" if aspect == "both" else aspect,
                       work, cb, seed=len(panels), mood=mood)
        if aspect == "both":
            make_wide(out_path, _wide_name(out_path))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    _write_script_txt(out_path, plan)
    _log(cb, 100, f"Done - {total:.0f}s video, {len(plan)} shots from {len(panels)} panels, mood {mood}")
    return out_path


def run_stage(stage, src, out_path, state_dir, lang="en", aspect="9:16", voice=None, speed=1.05, title="",
              detail="more", series_file="", captions=False):
    """make_recap split in three so a CI runner can show Script / Voice / Video as separate steps."""
    kl, lang_name, default_voice = LANGS.get(lang, LANGS["en"])
    os.makedirs(state_dir, exist_ok=True)
    pj, vz = os.path.join(state_dir, "plan.json"), os.path.join(state_dir, "voices.npz")
    if stage == "script":
        strips, panels, pchap, names, starts = _panels_from(src, True)
        _log(None, 18, f"Found {len(panels)} panels in {len(names)} chapter(s)")
        lines, mood = write_script([panel_image(strips, p) for p in panels], lang_name, title=title, detail=detail,
                                   starts=starts)
        plan = make_plan(panels, lines, pchap, names)
        json.dump({"n": len(panels), "mood": mood, "plan": plan}, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
        _log(None, 40, f"{len(plan)} shots planned, mood: {mood}")
    elif stage == "voice":
        plan = json.load(open(pj, encoding="utf-8"))["plan"]
        voices = tts([p["text"] for p in plan], kl, voice or default_voice, speed)
        np.savez(vz, *voices)
    elif stage == "video":
        saved = json.load(open(pj, encoding="utf-8"))
        strips, panels = _panels_from(src)
        if len(panels) != saved["n"]:
            raise RuntimeError("Panel count changed between stages")
        z = np.load(vz)
        voices = [z[f"arr_{i}"] for i in range(len(z.files))]
        smap = {}
        if series_file and os.path.exists(series_file):
            try:
                smap = json.load(open(series_file, encoding="utf-8"))
            except Exception:
                smap = {}
        series = series_of(title, smap.keys())
        work = tempfile.mkdtemp(prefix="recap_")
        try:
            total = render(plan_images(strips, saved["plan"]), saved["plan"], voices, out_path,
                           "9:16" if aspect == "both" else aspect, work,
                           seed=saved["n"], mood=saved.get("mood", "emotional"), locked=_locked(smap.get(series)),
                           captions=captions)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if series and series_file and CHOSEN_MUSIC and _locked(smap.get(series)) != CHOSEN_MUSIC:
            smap[series] = dict(CHOSEN_MUSIC)
            json.dump(smap, open(series_file, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            _log(None, 100, f"Music for '{series}': {smap[series]}")
        if aspect == "both":
            _log(None, 99, "Making the 16:9 YouTube copy")
            make_wide(out_path, _wide_name(out_path))
        _write_script_txt(out_path, saved["plan"])
        _log(None, 100, f"Done - {total:.0f}s video")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="recap.mp4")
    ap.add_argument("--lang", default="en", choices=list(LANGS))
    ap.add_argument("--aspect", default="9:16", choices=["9:16", "16:9", "both"])
    ap.add_argument("--voice")
    ap.add_argument("--speed", type=float, default=1.05)
    ap.add_argument("--stage", choices=["all", "script", "voice", "video"], default="all")
    ap.add_argument("--state", default="state")
    ap.add_argument("--title", default="")
    ap.add_argument("--detail", default="more", choices=["more", "normal"])
    ap.add_argument("--series-file", default="")
    ap.add_argument("--captions", action="store_true")
    ap.add_argument("--meta", default="", help="JSON with lang/aspect/voice/title/detail (overrides the flags)")
    a = ap.parse_args()
    if a.meta and os.path.exists(a.meta):
        m = json.load(open(a.meta, encoding="utf-8"))
        a.lang = m.get("lang") if m.get("lang") in LANGS else a.lang
        a.aspect = m.get("aspect") if m.get("aspect") in ("9:16", "16:9", "both") else a.aspect
        a.voice = m.get("voice") or a.voice
        a.title = m.get("title") or a.title
        a.detail = m.get("detail") if m.get("detail") in ("more", "normal") else a.detail
        a.captions = bool(m.get("captions", a.captions))
    if a.stage == "all":
        make_recap(a.src, a.out, a.lang, a.aspect, a.voice, a.speed)
    else:
        run_stage(a.stage, a.src, a.out, a.state, a.lang, a.aspect, a.voice, a.speed, a.title, a.detail, a.series_file,
                  a.captions)
