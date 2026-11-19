"""Subtle background music, written in code: soft pad chords + sparse piano + room reverb.
No outside service, no licences - every track is generated fresh for the video.
Moods: emotional, tense, mysterious, hopeful."""
import numpy as np

SR = 24000


def _hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)


# chord progressions as MIDI notes (each chord holds for 2 bars)
MOODS = {
    "emotional":  dict(bpm=64, prog=[[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]], piano="arp", pulse=False),   # Am F C G
    "hopeful":    dict(bpm=74, prog=[[48, 52, 55], [55, 59, 62], [57, 60, 64], [53, 57, 60]], piano="arp", pulse=False),   # C G Am F
    "mysterious": dict(bpm=58, prog=[[52, 55, 59, 62], [48, 52, 55, 59], [45, 48, 52, 55], [47, 50, 54, 57]], piano="sparse", pulse=False),  # Em9 Cmaj7 Am7 Bm7b5
    "tense":      dict(bpm=82, prog=[[50, 53, 57], [46, 50, 53], [48, 51, 55], [49, 52, 56]], piano="none", pulse=True),     # Dm Bb Cm C#dim-ish
}


def _pad_note(f, dur, rng):
    t = np.arange(int(dur * SR)) / SR
    det = rng.uniform(0.12, 0.3)
    w = (np.sin(2 * np.pi * (f - det) * t) + np.sin(2 * np.pi * (f + det) * t)
         + 0.25 * np.sin(2 * np.pi * 2 * f * t) + 0.08 * np.sin(2 * np.pi * 3 * f * t))
    a = min(1.8, dur / 3)
    env = np.minimum(1, t / a) * np.minimum(1, (dur - t) / a)
    return w * np.clip(env, 0, 1) * (1 + 0.06 * np.sin(2 * np.pi * 0.2 * t))


def _piano_note(f, rng):
    dur = 2.6
    t = np.arange(int(dur * SR)) / SR
    w = (np.sin(2 * np.pi * f * t) * np.exp(-t * 1.6)
         + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 3.2)
         + 0.12 * np.sin(2 * np.pi * 3 * f * t) * np.exp(-t * 5.0))
    return w * np.minimum(1, t / 0.006) * rng.uniform(0.75, 1.0)


def _pulse_note(f, dur):
    t = np.arange(int(dur * SR)) / SR
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)) * np.exp(-t * 7) * np.minimum(1, t / 0.004)


def _reverb(x, rng, secs=2.8, wet=0.38):
    n = int(secs * SR)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / SR * (6.9 / secs))
    k = 24                                        # darken the tail
    ir = np.convolve(ir, np.ones(k) / k, mode="same")
    ir /= np.sqrt((ir ** 2).sum())
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[:len(x)]
    return (1 - wet) * x + wet * y * 0.9


def make_music(seconds, mood="emotional", seed=0):
    """Return a mono float32 track of `seconds`, peak 1.0, with fade in/out."""
    m = MOODS.get(mood, MOODS["emotional"])
    rng = np.random.default_rng(seed)
    n = int(seconds * SR) + SR * 3
    out = np.zeros(n)
    beat = 60.0 / m["bpm"]
    chord_len = 8 * beat
    t, ci = 0.0, 0
    while t < seconds + 1:
        chord = m["prog"][ci % len(m["prog"])]
        i0 = int(t * SR)
        # pad: chord + root an octave down, overlapping the next chord a little
        for note in chord + [chord[0] - 12]:
            w = _pad_note(_hz(note), chord_len + 1.5, rng) * (0.20 if note != chord[0] - 12 else 0.26)
            out[i0:i0 + len(w)] += w[:max(0, n - i0)]
        # piano
        if m["piano"] != "none":
            hi = [x + 12 for x in chord]
            steps = range(0, 8) if m["piano"] == "arp" else range(0, 8, 2)
            for b in steps:
                if m["piano"] == "sparse" and rng.random() < 0.35:
                    continue
                note = hi[(b if m["piano"] == "arp" else rng.integers(len(hi))) % len(hi)]
                if m["piano"] == "arp" and b % 4 == 0:
                    note = hi[0]
                w = _piano_note(_hz(note), rng) * 0.22
                j = i0 + int(b * beat * SR)
                out[j:j + len(w)] += w[:max(0, n - j)]
        # low heartbeat pulse for tense scenes
        if m["pulse"]:
            for b in range(16):
                w = _pulse_note(_hz(chord[0] - 24), beat / 2) * (0.5 if b % 2 == 0 else 0.3)
                j = i0 + int(b * beat / 2 * SR)
                out[j:j + len(w)] += w[:max(0, n - j)]
        t += chord_len
        ci += 1
    out = _reverb(out[:int(seconds * SR)], rng)
    fade_in, fade_out = int(2.0 * SR), int(3.0 * SR)
    out[:fade_in] *= np.linspace(0, 1, fade_in)
    out[-fade_out:] *= np.linspace(1, 0, fade_out)
    peak = np.abs(out).max() or 1.0
    return (out / peak).astype(np.float32)


def duck(music, voice, depth=0.5, window=0.35):
    """Lower the music while the narrator speaks (smooth, no pumping)."""
    k = int(window * SR)
    env = np.convolve(np.abs(voice), np.ones(k) / k, mode="same")
    talking = np.clip(env / (env.max() * 0.08 + 1e-9), 0, 1)
    talking = np.convolve(talking, np.ones(k * 2) / (k * 2), mode="same")
    return music * (1 - depth * talking)


if __name__ == "__main__":
    import soundfile as sf, sys
    mood = sys.argv[1] if len(sys.argv) > 1 else "emotional"
    sf.write(f"music_{mood}.wav", make_music(40, mood, 1), SR)
    print("ok")
