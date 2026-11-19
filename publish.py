"""Publishing worker: finished videos uploaded from the phone -> details -> YouTube (scheduled).

Runs on GitHub Actions (workflow publish.yml): when the phone finishes an upload, and every 2 hours as a
safety net. Each run looks at every video waiting in the "jobs" release, does whatever is still left to do
for it, and saves the result. A step that is already done is never done again, so running it twice is safe.

Everything in the public release is encrypted with the account key K (AES-256, same MRC1 format as crypto.py):
  acct.enc        {"w": K wrapped with public_key.pem, "d": MRC1(K, {"youtube": {client_id, client_secret,
                  refresh_token}})}   - written by the phone
  p<id>.mp4.enc   the video                                                  - phone
  p<id>.json.enc  what the phone wants: series, episode, mode, publish time, paused, thumbnail revision
  p<id>.jpg.enc   thumbnail picked on the phone (optional)                  - phone
  status.enc      what this worker has done for every video                 - this worker

Modes: "store" = only keep it in the cloud; "details" = also write title/description/tags;
"full" = details + upload to YouTube with the batch's publish time (YouTube itself publishes it then,
so nothing has to be awake at that moment).

Safety: a video is only sent to YouTube in "full" mode, and only with a publish time or an explicit
"publish now". With a publish time it goes up as private + publishAt.

CLI: python publish.py
"""
import os, sys, json, re, time, base64, datetime, subprocess, tempfile, shutil
import requests
import crypto, post

API = "https://api.github.com/"
REPO = os.environ.get("GITHUB_REPOSITORY", "")
GH = {"Authorization": f"Bearer {os.environ.get('GH_TOKEN', '')}", "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28"}
ID_RE = re.compile(r"^(p[a-f0-9]{12})\.json\.enc$")
MODELS = ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite")
WORK = tempfile.mkdtemp(prefix="pub_")


# ------------------------------------------------------------------ release storage
def release():
    r = requests.get(API + f"repos/{REPO}/releases/tags/jobs", headers=GH, timeout=60)
    r.raise_for_status()
    return r.json()


def assets(rel):
    out, page = {}, 1
    while True:
        r = requests.get(API + f"repos/{REPO}/releases/{rel['id']}/assets?per_page=100&page={page}", headers=GH, timeout=60)
        r.raise_for_status()
        batch = r.json()
        for a in batch:
            if a.get("state", "uploaded") == "uploaded":
                out[a["name"]] = a
        if len(batch) < 100:
            return out
        page += 1


def download(asset, dest):
    h = dict(GH, Accept="application/octet-stream")
    with requests.get(asset["url"], headers=h, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def fetch_bytes(asset):
    h = dict(GH, Accept="application/octet-stream")
    r = requests.get(asset["url"], headers=h, timeout=300)
    r.raise_for_status()
    return r.content


def delete_asset(asset):
    requests.delete(API + f"repos/{REPO}/releases/assets/{asset['id']}", headers=GH, timeout=60)


def upload_bytes(rel, name, data, existing):
    if name in existing:
        delete_asset(existing[name])
    url = rel["upload_url"].split("{")[0] + f"?name={name}"
    r = requests.post(url, headers=dict(GH, **{"Content-Type": "application/octet-stream"}), data=data, timeout=300)
    r.raise_for_status()
    existing[name] = r.json()


# ------------------------------------------------------------------ account + status
def load_account(found):
    if "acct.enc" not in found:
        return None, None
    a = json.loads(fetch_bytes(found["acct.enc"]))
    key = crypto.unwrap(a["w"])
    data = json.loads(crypto.decrypt_bytes(base64.b64decode(a["d"]), key))
    return key, data


def load_status(found, key):
    if "status.enc" not in found:
        return {}
    try:
        return json.loads(crypto.decrypt_bytes(fetch_bytes(found["status.enc"]), key))
    except Exception as e:   # made with an older account key (app reinstalled): start a fresh record
        print("status unreadable, starting fresh:", type(e).__name__)
        return {}


def save_status(rel, found, key, status):
    upload_bytes(rel, "status.enc", crypto.encrypt_bytes(json.dumps(status, ensure_ascii=False).encode(), key), found)


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def stamp():
    return now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ details (Gemini)
def media_for_gemini(video):
    """Narration audio (mono, small) + 10 frames spread over the video: enough for Gemini to know the story."""
    dur = post._duration(video)
    audio = os.path.join(WORK, "a.mp3")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-t", "3600", "-vn", "-ac", "1", "-ar", "16000",
                    "-b:a", "32k", audio], check=False)
    frames = []
    for k in range(10):
        t = dur * (k + 0.5) / 10 if dur > 0 else k * 5
        f = os.path.join(WORK, f"f{k}.jpg")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
                        "-vf", "scale='min(640,iw)':-2", "-q:v", "5", f], check=False)
        if os.path.exists(f) and os.path.getsize(f) > 0:
            frames.append(f)
    return (audio if os.path.exists(audio) and os.path.getsize(audio) > 0 else None), frames, dur


def clean_filename(series, episode):
    base = re.sub(r'[\\/:*?"<>|]+', " ", series or "Manhua").strip()
    base = re.sub(r"\s+", " ", base)
    if episode not in (None, ""):
        try:
            n = float(episode)
            ep = f"{int(n):03d}" if n == int(n) else f"{n:05.1f}"
        except ValueError:
            ep = str(episode)
        return f"{base} - Episode {ep}.mp4"
    return f"{base}.mp4"


def write_details(video, meta):
    series, ep = meta.get("series", ""), meta.get("episode", "")
    label = f"{series} Episode {ep}".strip() if ep not in (None, "") else series
    fallback = {"title": (f"{label} | Manhua Recap" if label else "Manhua Recap")[:100],
                "description": f"{label} recap.".strip(), "tags": ["manhua", "manhwa", "manhua recap", "manhwa recap"],
                "hashtags": ["#manhua", "#manhwa", "#recap"]}
    audio, frames, dur = media_for_gemini(video)
    from google import genai
    from google.genai import types
    # no hidden retries inside the library (they waited ~8 min on a busy model): one try per model, next model
    # straight away, and a short pause only between whole rounds
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                     http_options=types.HttpOptions(timeout=180000, retry_options=types.HttpRetryOptions(attempts=1)))
    parts = []
    if audio:
        parts.append(types.Part.from_bytes(data=open(audio, "rb").read(), mime_type="audio/mpeg"))
    for f in frames:
        parts.append(types.Part.from_bytes(data=open(f, "rb").read(), mime_type="image/jpeg"))
    parts.append(
        "This is a manhua/manhwa recap video for YouTube (the audio is its narration, the images are frames from it).\n"
        f"Series: {series or 'unknown'}\nEpisode/chapter: {ep or 'unknown'}\nLength: {int(dur // 60)} min\n"
        "Write its YouTube details. Return JSON with:\n"
        '"title": max 90 characters, starts with a strong hook from the story, includes the series name'
        + (f' and "Episode {ep}"' if ep not in (None, "") else "") + ", no false claims;\n"
        '"description": 2-4 short paragraphs that pull the viewer in without spoiling the ending;\n'
        '"tags": 10-15 short search tags (series name, genre, "manhua recap", "manhwa recap" ...);\n'
        '"hashtags": 3-6 hashtags, each starting with #.')
    last = None
    for rnd in range(4):
        if rnd:
            time.sleep(30 * rnd)
        for m in MODELS:
            try:
                r = c.models.generate_content(model=m, contents=parts, config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.6,
                    safety_settings=[types.SafetySetting(category=cat, threshold="BLOCK_ONLY_HIGH") for cat in (
                        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]))
                d = json.loads(r.text or "")
                tags = [str(t).strip()[:30] for t in d.get("tags", []) if str(t).strip()][:15]
                hashtags = [h if h.startswith("#") else "#" + h for h in (str(x).strip().replace(" ", "") for x in d.get("hashtags", [])) if len(h) > 1][:6]
                return {"title": str(d["title"]).strip()[:100], "description": str(d["description"]).strip(),
                        "tags": tags or fallback["tags"], "hashtags": hashtags or fallback["hashtags"], "model": m}
            except Exception as e:
                last = e
                print(f"  {m}: {str(e)[:80]}")
    print("Gemini failed, using simple details:", str(last)[:200])
    fallback["model"] = "fallback"
    return fallback


def full_description(det, meta):
    text = det["description"].strip()
    if det.get("hashtags"):
        text += "\n\n" + " ".join(det["hashtags"])
    footer = (meta.get("footer") or "").strip()
    if footer:
        text += "\n\n" + footer
    return text[:4900]


# ------------------------------------------------------------------ YouTube
YT = "https://www.googleapis.com/youtube/v3/"


class QuotaHit(Exception):
    pass


def yt_error(r):
    try:
        reasons = [e.get("reason", "") for e in r.json()["error"].get("errors", [])]
    except Exception:
        reasons = []
    if r.status_code == 403 and any(x in ("quotaExceeded", "uploadLimitExceeded", "dailyLimitExceeded", "rateLimitExceeded") for x in reasons):
        raise QuotaHit(",".join(reasons))
    if r.status_code == 403 and "insufficientPermissions" in reasons:
        raise RuntimeError("YouTube needs one more permission - open Settings in the app and tap Reconnect YouTube.")
    raise RuntimeError(f"YouTube said {r.status_code}: {r.text[:300]}")


def future(when):
    """The publish time if it is still at least 5 minutes away (RFC 3339 UTC), else ''."""
    if not when:
        return ""
    try:
        t = datetime.datetime.fromisoformat(when.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if t <= now_utc() + datetime.timedelta(minutes=5):
        return ""
    return t.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def yt_upload(video, det, meta):
    """Resumable upload. Scheduled -> private + publishAt; 'publish now' or a slot already passed -> public."""
    tok = post._access_token()
    when = future(meta.get("publish_at", ""))
    status = {"privacyStatus": "private", "publishAt": when} if when else {"privacyStatus": "public"}
    status["selfDeclaredMadeForKids"] = False
    body = {"snippet": {"title": det["title"], "description": full_description(det, meta), "tags": det.get("tags", []),
                        "categoryId": "24"}, "status": status}
    size = os.path.getsize(video)
    r = requests.post("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                      headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json",
                               "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)},
                      data=json.dumps(body), timeout=60)
    if r.status_code >= 300:
        yt_error(r)
    loc = r.headers["Location"]
    for attempt in range(5):
        with open(video, "rb") as f:
            u = requests.put(loc, headers={"Authorization": f"Bearer {tok}", "Content-Type": "video/mp4"}, data=f, timeout=7200)
        if u.status_code in (200, 201):
            v = u.json()
            s = v["status"]
            # Google keeps uploads from an unreviewed API project private: the publish time or "public" is dropped
            locked = (bool(when) and not s.get("publishAt")) or (not when and s["privacyStatus"] == "private")
            return {"yt_id": v["id"], "yt_url": f"https://youtu.be/{v['id']}", "yt_privacy": s["privacyStatus"],
                    "yt_publish_at": s.get("publishAt", "") or ("" if when else stamp()), "yt_locked": locked}
        if u.status_code < 500:
            yt_error(u)
        time.sleep(15 * (attempt + 1))
    raise RuntimeError("YouTube upload kept failing")


def yt_set_schedule(yt_id, meta):
    """Change an uploaded video's publish time, or hold it (paused / no time = stays private)."""
    tok = post._access_token()
    r = requests.get(YT + f"videos?part=status&id={yt_id}", headers={"Authorization": f"Bearer {tok}"}, timeout=60)
    if r.status_code >= 300:
        yt_error(r)
    items = r.json().get("items", [])
    if not items:
        raise RuntimeError("The video is no longer on YouTube.")
    st = items[0]["status"]
    if st.get("privacyStatus") == "public":
        return {"yt_privacy": "public", "yt_publish_at": st.get("publishAt", "") or ""}   # already out: leave it
    when = "" if meta.get("paused") else future(meta.get("publish_at", ""))
    new = {"privacyStatus": "private", "selfDeclaredMadeForKids": False}
    if when:
        new["publishAt"] = when
    elif meta.get("publish_now") and not meta.get("paused"):
        new = {"privacyStatus": "public", "selfDeclaredMadeForKids": False}
    r = requests.put(YT + "videos?part=status", headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
                     data=json.dumps({"id": yt_id, "status": new}), timeout=60)
    if r.status_code >= 300:
        yt_error(r)
    s = r.json()["status"]
    return {"yt_privacy": s["privacyStatus"], "yt_publish_at": s.get("publishAt", "") or ("" if s["privacyStatus"] != "public" else stamp())}


def yt_thumbnail(yt_id, path):
    tok = post._access_token()
    with open(path, "rb") as f:
        r = requests.post(f"https://www.googleapis.com/upload/youtube/v3/thumbnails/set?videoId={yt_id}",
                          headers={"Authorization": f"Bearer {tok}", "Content-Type": "image/jpeg"}, data=f, timeout=120)
    if r.status_code >= 300:
        try:
            reasons = [e.get("reason", "") for e in r.json()["error"].get("errors", [])]
        except Exception:
            reasons = []
        if "forbidden" in reasons or r.status_code == 403:
            raise RuntimeError("YouTube refused the thumbnail - verify your channel at youtube.com/verify, then tap Retry.")
        yt_error(r)


# ------------------------------------------------------------------ one video
def wanted_on_youtube(meta):
    return meta.get("mode") == "full" and not meta.get("paused") and (meta.get("publish_at") or meta.get("publish_now"))


def process(vid, meta, st, key, found, quota):
    """Advance one video as far as it can go. Returns True when the status changed."""
    changed = False
    video = None

    def get_video():
        nonlocal video
        if video:
            return video
        name = f"{vid}.mp4.enc"
        if name not in found:
            raise RuntimeError("The video file is missing from the cloud - upload it again.")
        enc = download(found[name], os.path.join(WORK, name))
        video = os.path.join(WORK, f"{vid}.mp4")
        crypto.decrypt_file(enc, video, key)
        os.remove(enc)
        return video

    if st.get("rev") != meta.get("rev") or st.get("error"):
        st["rev"] = meta.get("rev")
        st["error"] = ""      # a failure sets it again below
        changed = True
    st["filename"] = clean_filename(meta.get("series", ""), meta.get("episode", ""))

    # 1) details
    if meta.get("mode") in ("details", "full") and not st.get("details"):
        st["stage"] = "details"
        det = write_details(get_video(), meta)
        st["details"] = det
        st["details_at"] = stamp()
        changed = True

    # 2) YouTube upload
    if not st.get("yt_id") and wanted_on_youtube(meta):
        if quota["hit"]:
            st["note"] = "Waiting for YouTube's daily upload limit - it continues automatically."
            return True
        try:
            st["stage"] = "youtube"
            st.update(yt_upload(get_video(), st["details"], meta))
            st["uploaded_at"] = stamp()
            st["sched_rev"] = meta.get("sched_rev", 0)
            st["note"] = ""
            changed = True
        except QuotaHit as e:
            quota["hit"] = True
            st["note"] = "Waiting for YouTube's daily upload limit - it continues automatically."
            print("quota:", e)
            return True

    # 3) schedule changes after upload (pause / resume / new time)
    if st.get("yt_id") and st.get("sched_rev", 0) != meta.get("sched_rev", 0):
        st.update(yt_set_schedule(st["yt_id"], meta))
        st["sched_rev"] = meta.get("sched_rev", 0)
        changed = True

    # 4) thumbnail picked on the phone
    if st.get("yt_id") and meta.get("thumb_rev", 0) and st.get("thumb_rev", 0) != meta.get("thumb_rev", 0):
        name = f"{vid}.jpg.enc"
        if name in found:
            jpg = os.path.join(WORK, f"{vid}.jpg")
            crypto.decrypt_file(download(found[name], jpg + ".enc"), jpg, key)
            yt_thumbnail(st["yt_id"], jpg)
            st["thumb_rev"] = meta.get("thumb_rev", 0)
            changed = True

    # the video lives on YouTube now: free the cloud copy
    if st.get("yt_id") and f"{vid}.mp4.enc" in found:
        delete_asset(found.pop(f"{vid}.mp4.enc"))
        st["cloud_copy"] = False
        changed = True
    st["stage"] = "done" if st.get("yt_id") else ("ready" if st.get("details") or meta.get("mode") == "store" else st.get("stage", ""))
    if video and os.path.exists(video):
        os.remove(video)
    return changed


def main():
    rel = release()
    found = assets(rel)
    key, acct = load_account(found)
    if key is None:
        print("no account yet (acct.enc) - nothing to do")
        return
    if (acct.get("youtube") or {}).get("refresh_token"):
        post.AUTH.clear()
        post.AUTH.update(acct["youtube"])
    status = load_status(found, key)
    ids = sorted(m.group(1) for n in found if (m := ID_RE.match(n)))
    metas = {}
    for vid in ids:
        try:
            metas[vid] = json.loads(crypto.decrypt_bytes(fetch_bytes(found[f"{vid}.json.enc"]), key))
        except Exception as e:
            status.setdefault(vid, {})["error"] = "Made with an older install of the app - delete it and upload again."
            print(vid, "unreadable:", type(e).__name__)
    # earliest publish time first, so the daily YouTube limit is spent on what goes out soonest
    order = sorted(metas, key=lambda v: (metas[v].get("publish_at") or "9999", metas[v].get("episode") or 0, v))
    quota = {"hit": False}
    for vid in order:
        meta, st = metas[vid], status.setdefault(vid, {})
        if meta.get("title"):
            print("::add-mask::" + meta["title"])
        if meta.get("series"):
            print("::add-mask::" + meta["series"])
        try:
            if process(vid, meta, st, key, found, quota):
                st["updated"] = stamp()
                save_status(rel, found, key, status)
            print(vid, st.get("stage"), "ok")
        except Exception as e:
            st["error"] = str(e)[:400]
            st["updated"] = stamp()
            save_status(rel, found, key, status)
            print(vid, "failed:", str(e)[:200])
    # forget videos the phone deleted
    gone = [v for v in status if v not in metas and v not in ids]
    if gone:
        for v in gone:
            status.pop(v, None)
        save_status(rel, found, key, status)
    shutil.rmtree(WORK, ignore_errors=True)
    print("done:", len(metas), "videos", "(YouTube daily limit reached)" if quota["hit"] else "")


if __name__ == "__main__":
    main()
