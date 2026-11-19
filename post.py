"""Post a finished recap to YouTube from the render machine (now, or scheduled).

Credentials (GitHub secrets): YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN.
Job settings come from state/meta.json -> "post": {"youtube": {"privacy": "public|unlisted|private",
"publish_at": "2026-11-20T18:00:00Z" or "", "shorts": true}}.
Gemini writes the title / description / tags from the narration; the music credit is appended.
The result (links or errors) is written to out/<job>_post.json, which crypto.py locks for the phone.

CLI: python post.py JOB
"""
import os, sys, json, re, time, datetime, requests

TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
MAX_SHORT = 180          # YouTube Shorts: vertical and at most 3 minutes


def _duration(path):
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


AUTH = {}


def _access_token():
    a = AUTH or {"client_id": os.environ.get("YT_CLIENT_ID", ""), "client_secret": os.environ.get("YT_CLIENT_SECRET", ""),
                 "refresh_token": os.environ.get("YT_REFRESH_TOKEN", "")}
    data = {"client_id": a["client_id"], "refresh_token": a["refresh_token"], "grant_type": "refresh_token"}
    if a.get("client_secret"):
        data["client_secret"] = a["client_secret"]
    r = requests.post(TOKEN_URL, data=data, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def write_metadata(script, title_hint, credits):
    """Title, description and tags from the narration (Gemini). Falls back to simple text if Gemini is down."""
    fallback = {"title": (title_hint or "Manhua Recap")[:95],
                "description": script[:600] + "...", "tags": ["manhua", "manhwa", "recap", "manga recap"]}
    try:
        from google import genai
        from google.genai import types
        c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        q = ("You write YouTube metadata for a manhua/manhwa recap video. Narration:\n" + script[:6000] +
             f"\n\nSeries/chapter: {title_hint or 'unknown'}\n"
             "Return JSON: {\"title\": catchy, max 90 characters, include the series name and chapter if known, no clickbait lies, "
             "\"description\": 2-4 short paragraphs summarising the story without spoiling the ending, then a line of 5-8 hashtags, "
             "\"tags\": 8-15 short search tags}.")
        for m in ("gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.7-flash"):
            try:
                r = c.models.generate_content(model=m, contents=q, config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.6))
                d = json.loads(r.text)
                fallback = {"title": str(d["title"])[:100], "description": str(d["description"]),
                            "tags": [str(t)[:30] for t in d.get("tags", [])][:15]}
                break
            except Exception:
                time.sleep(5)
    except Exception:
        pass
    if credits:
        fallback["description"] += "\n\nMusic:\n" + "\n".join(credits)
    fallback["description"] = fallback["description"][:4900]
    return fallback


def youtube_upload(path, meta, privacy="private", publish_at="", shorts=False):
    tok = _access_token()
    title = meta["title"]
    desc = meta["description"]
    if shorts and "#shorts" not in (title + desc).lower():
        desc = desc + "\n#Shorts"
    status = {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}
    if publish_at:
        status = {"privacyStatus": "private", "publishAt": publish_at, "selfDeclaredMadeForKids": False}
    body = {"snippet": {"title": title, "description": desc, "tags": meta.get("tags", []), "categoryId": "24"},
            "status": status}
    size = os.path.getsize(path)
    r = requests.post(UPLOAD_URL, headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json",
                                           "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)},
                      data=json.dumps(body), timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(f"YouTube refused the upload ({r.status_code}): {r.text[:300]}")
    loc = r.headers["Location"]
    for attempt in range(5):
        with open(path, "rb") as f:
            u = requests.put(loc, headers={"Authorization": f"Bearer {tok}", "Content-Type": "video/mp4"}, data=f, timeout=3600)
        if u.status_code in (200, 201):
            v = u.json()
            return {"id": v["id"], "url": f"https://youtu.be/{v['id']}", "privacy": v["status"]["privacyStatus"],
                    "publish_at": v["status"].get("publishAt", ""), "title": title}
        if u.status_code < 500:
            raise RuntimeError(f"YouTube upload failed ({u.status_code}): {u.text[:300]}")
        time.sleep(10 * (attempt + 1))
    raise RuntimeError("YouTube upload kept failing")


def main(job):
    meta = json.load(open("state/meta.json", encoding="utf-8"))
    post = meta.get("post") or {}
    result = {"youtube": []}
    yt = post.get("youtube")
    if not yt:
        print("nothing to post")
        return
    if (yt.get("auth") or {}).get("refresh_token"):
        AUTH.update(yt["auth"])
    if not AUTH and not os.environ.get("YT_REFRESH_TOKEN"):
        result["error"] = "YouTube isn't connected - tap Connect YouTube in the app."
    else:
        script = open(f"out/{job}_script.txt", encoding="utf-8").read()
        body, _, music = script.partition("--- Music")
        credits = [l for l in music.splitlines() if "Kevin MacLeod" in l]
        md = write_metadata(body, meta.get("title", ""), credits)
        print("::add-mask::" + md["title"])
        privacy = yt.get("privacy", "private") if yt.get("privacy") in ("public", "unlisted", "private") else "private"
        when = yt.get("publish_at", "")
        if when:
            try:   # must be in the future, RFC 3339 UTC
                t = datetime.datetime.fromisoformat(when.replace("Z", "+00:00"))
                if t <= datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5):
                    when = ""
                else:
                    when = t.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                when = ""
        # main upload: the 16:9 video when there is one (long-form), else the only video
        wide = f"out/{job}_169.mp4"
        main_video = wide if os.path.exists(wide) else f"out/{job}.mp4"
        tall = f"out/{job}.mp4"
        try:
            result["youtube"].append(dict(youtube_upload(main_video, md, privacy, when), kind="video"))
        except Exception as e:
            result["youtube"].append({"error": str(e)[:400], "kind": "video"})
        # Shorts: only a vertical video of 3 minutes or less counts as a Short
        if yt.get("shorts", True) and os.path.exists(wide) and _duration(tall) <= MAX_SHORT:
            try:
                result["youtube"].append(dict(youtube_upload(tall, md, privacy, when, shorts=True), kind="short"))
            except Exception as e:
                result["youtube"].append({"error": str(e)[:400], "kind": "short"})
    json.dump(result, open(f"out/{job}_post.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("posting done:", [("ok" if "url" in x else "failed") for x in result["youtube"]])


if __name__ == "__main__":
    main(sys.argv[1])
