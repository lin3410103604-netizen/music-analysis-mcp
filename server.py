
import os
import tempfile
import librosa
import numpy as np
import requests
from fastmcp import FastMCP

mcp = FastMCP("Music Analysis")


@mcp.tool()
def analyze_audio(audio_url: str) -> dict:
    """下载音频并分析时长、BPM、节拍和估计调性，同时返回下载诊断信息。"""

    path = None

    try:
        response = requests.get(
            audio_url,
            timeout=60,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        response.raise_for_status()

        audio_data = response.content
        content_type = response.headers.get("Content-Type", "unknown")
        file_size = len(audio_data)

        if file_size == 0:
            return {
                "success": False,
                "error": "下载内容为空",
                "http_status": response.status_code,
                "content_type": content_type,
                "file_size": file_size,
                "final_url": response.url
            }

        # 判断下载内容是否可能是网页
        if "text/html" in content_type.lower():
            return {
                "success": False,
                "error": "下载到的是网页，不是音频文件",
                "http_status": response.status_code,
                "content_type": content_type,
                "file_size": file_size,
                "final_url": response.url,
                "content_preview": audio_data[:100].decode(
                    "utf-8", errors="replace"
                )
            }

        suffix = os.path.splitext(
            response.url.split("?")[0]
        )[1].lower()

        if suffix not in [".mp3", ".wav", ".flac", ".ogg", ".m4a"]:
            suffix = ".mp3"

        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as f:
            f.write(audio_data)
            path = f.name

        if not os.path.isfile(path):
            return {
                "success": False,
                "error": "临时音频文件创建失败",
                "file_size": file_size
            }

        # 尝试读取音频
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
        except Exception as e:
            return {
                "success": False,
                "error": "音频解码失败",
                "decode_error": str(e),
                "http_status": response.status_code,
                "content_type": content_type,
                "file_size": file_size,
                "final_url": response.url,
                "file_exists": os.path.isfile(path),
                "file_size_on_disk": os.path.getsize(path)
            }

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_index = int(np.argmax(chroma.mean(axis=1)))

        keys = [
            "C", "C#", "D", "D#", "E", "F",
            "F#", "G", "G#", "A", "A#", "B"
        ]

        return {
            "success": True,
            "duration_seconds": round(len(y) / sr, 2),
            "sample_rate": sr,
            "bpm": round(float(np.asarray(tempo).reshape(-1)[0]), 2),
            "beat_count": int(len(beats)),
            "estimated_key": keys[key_index],
            "http_status": response.status_code,
            "content_type": content_type,
            "file_size": file_size,
            "final_url": response.url
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

    finally:
        if path and os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        stateless_http=True
    )
