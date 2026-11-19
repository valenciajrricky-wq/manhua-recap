"""Encryption for the public render repo: everything uploaded or published is locked.

The phone makes a random AES-256 key per job, encrypts the chapter zip and the job settings with it, and
wraps the key with the repo's RSA public key (public_key.pem). Only this runner can unwrap it, using the
RSA private key kept in the RECAP_PRIVKEY secret. The finished video + script are encrypted with the same
job key, so only the phone that sent the job can open them. series.json (music per series) is encrypted
with a key derived from the private key, so it is readable only here.

File format: b"MRC1" + 8-byte nonce prefix, then records [4-byte length][AES-GCM ciphertext+tag] for 1 MiB
chunks; nonce = prefix + 4-byte chunk index; the final chunk has AAD b"last" (stops truncation).

CLI (used by the workflow):
  python crypto.py unlock JOB   -> in/JOB.zip, state/meta.json, state/series.json, state/jobkey
  python crypto.py lock JOB     -> out/JOB.mp4.enc, out/JOB_script.txt.enc, state/series.json.enc
"""
import os, sys, json, base64, struct, hashlib, io

CH = 1 << 20
MAGIC = b"MRC1"


def _aes(key):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key)


def _priv():
    from cryptography.hazmat.primitives import serialization
    return serialization.load_pem_private_key(os.environ["RECAP_PRIVKEY"].encode(), None)


def unwrap(b64):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    return _priv().decrypt(base64.b64decode(b64), padding.OAEP(
        mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))


def site_key():
    return hashlib.sha256(b"manhua-recap-site|" + os.environ["RECAP_PRIVKEY"].encode()).digest()


def encrypt_stream(f, o, key):
    a, pre = _aes(key), os.urandom(8)
    o.write(MAGIC + pre)
    i, cur = 0, f.read(CH)
    while True:
        nxt = f.read(CH)
        last = not nxt
        c = a.encrypt(pre + struct.pack(">I", i), cur, b"last" if last else b"")
        o.write(struct.pack(">I", len(c)) + c)
        i += 1
        if last:
            break
        cur = nxt


def decrypt_stream(f, o, key):
    a = _aes(key)
    head = f.read(12)
    if head[:4] != MAGIC:
        raise ValueError("not an encrypted Manhua Recap file")
    pre, i = head[4:], 0
    hdr = f.read(4)
    while hdr:
        c = f.read(struct.unpack(">I", hdr)[0])
        nxt = f.read(4)
        o.write(a.decrypt(pre + struct.pack(">I", i), c, b"" if nxt else b"last"))
        i += 1
        hdr = nxt


def encrypt_file(src, dst, key):
    with open(src, "rb") as f, open(dst, "wb") as o:
        encrypt_stream(f, o, key)


def decrypt_file(src, dst, key):
    with open(src, "rb") as f, open(dst, "wb") as o:
        decrypt_stream(f, o, key)


def decrypt_bytes(data, key):
    o = io.BytesIO()
    decrypt_stream(io.BytesIO(data), o, key)
    return o.getvalue()


def encrypt_bytes(data, key):
    o = io.BytesIO()
    encrypt_stream(io.BytesIO(data), o, key)
    return o.getvalue()


def unlock(job):
    os.makedirs("state", exist_ok=True)
    wrapped = os.environ.get("WRAPPED", "")
    if wrapped:
        key = unwrap(wrapped)
        open("state/jobkey", "wb").write(key)
        decrypt_file(f"in/{job}.zip.enc", f"in/{job}.zip", key)
        os.remove(f"in/{job}.zip.enc")
        meta = json.loads(decrypt_bytes(base64.b64decode(os.environ.get("META", "")), key)) if os.environ.get("META") else {}
    else:   # older app version: plain upload, settings as plain inputs
        meta = {k: os.environ.get(e, "") for k, e in
                (("lang", "LANG_CODE"), ("aspect", "ASPECT"), ("voice", "VOICE"), ("title", "TITLE"), ("detail", "DETAIL"))}
    if meta.get("title"):
        print("::add-mask::" + meta["title"])
        series = meta["title"].split(" - ")[0].strip()
        if series:
            print("::add-mask::" + series)
    json.dump(meta, open("state/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    if os.path.exists("state/series.json.enc"):
        open("state/series.json", "wb").write(decrypt_bytes(open("state/series.json.enc", "rb").read(), site_key()))
        os.remove("state/series.json.enc")
    print("unlocked", "encrypted job" if wrapped else "plain job")


def lock(job):
    if os.path.exists("state/jobkey"):
        key = open("state/jobkey", "rb").read()
        for name in (f"{job}.mp4", f"{job}_169.mp4", f"{job}_script.txt", f"{job}_post.json"):
            if os.path.exists(f"out/{name}"):
                encrypt_file(f"out/{name}", f"out/{name}.enc", key)
    if os.path.exists("state/series.json"):
        open("state/series.json.enc", "wb").write(encrypt_bytes(open("state/series.json", "rb").read(), site_key()))
    print("locked")


if __name__ == "__main__":
    {"unlock": unlock, "lock": lock}[sys.argv[1]](sys.argv[2])
