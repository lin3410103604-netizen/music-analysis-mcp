
import os
import tempfile
import librosa
import numpy as np
from fastmcp import FastMCP

mcp = FastMCP("Music Analysis")

@mcp.tool()
def analyze_audio(audio_url: str) -> dict:
    """分析音频的时长、BPM、节拍和估计调性。提供可访问的音频URL。"""
    import requests

    response = requests.get(audio_url, timeout=60)
    response.raise_for_status()

    suffix = os.path.splitext(audio_url.split("?")[0])[1] or ".mp3"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(response.content)
        path = f.name

    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_index = int(np.argmax(chroma.mean(axis=1)))

        keys = ["C", "C#", "D", "D#", "E", "F",
                "F#", "G", "G#", "A", "A#", "B"]

        return {
            "duration_seconds": round(len(y) / sr, 2),
            "sample_rate": sr,
            "bpm": round(float(np.asarray(tempo).reshape(-1)[0]), 2),
            "beat_count": int(len(beats)),
            "estimated_key": keys[key_index]
        }
    finally:
        os.remove(path)
if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
