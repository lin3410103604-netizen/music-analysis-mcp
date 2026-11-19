
import os
import tempfile
from urllib.parse import urlparse

import requests
from fastmcp import FastMCP

mcp = FastMCP("Music Analysis")

MAX_FILE_SIZE = 25 * 1024 * 1024
MAX_ANALYSIS_SECONDS = 60
ALLOWED_HOSTS = {"tempfile.org", "www.tempfile.org"}


def validate_audio_url(audio_url):
    parsed = urlparse(audio_url)

    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
    ):
        raise ValueError("目前只允许 tempfile.org 的 HTTPS 链接")


def download_audio(audio_url):
    validate_audio_url(audio_url)

    downloaded = 0
    path = None

    response = requests.get(
        audio_url,
        stream=True,
        timeout=(15, 45),
        allow_redirects=False,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        if 300 <= response.status_code < 400:
            raise ValueError("音频链接发生重定向，暂不支持，请使用直接下载链接")

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", "unknown"
        )

        if "text/html" in content_type.lower():
            raise ValueError("下载到的是网页，不是音频文件")

        content_length = response.headers.get("Content-Length")

        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise ValueError("文件超过25MB限制")

        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        ) as f:
            path = f.name

            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue

                downloaded += len(chunk)

                if downloaded > MAX_FILE_SIZE:
                    raise ValueError("文件超过25MB限制")

                f.write(chunk)

        if downloaded == 0:
            raise ValueError("下载内容为空")

        return path, {
            "http_status": response.status_code,
            "content_type": content_type,
            "downloaded_bytes": downloaded,
            "downloaded_MB": round(downloaded / 1024 / 1024, 2),
            "final_url": response.url
        }

    except Exception:
        if path and os.path.exists(path):
            os.remove(path)
        raise

    finally:
        response.close()


@mcp.tool()
def test_audio_download(audio_url: str) -> dict:
    """测试音频能否下载，不进行音乐分析。"""

    path = None

    try:
        path, info = download_audio(audio_url)

        return {
            "success": True,
            **info,
            "message": "下载读取完成，未进行音频分析"
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


@mcp.tool()
def analyze_audio(audio_url: str) -> dict:
    """分析音频前60秒，返回时长、BPM、节拍数量和估计调性。"""

    path = None

    try:
        path, info = download_audio(audio_url)

        # 延迟导入，避免服务启动时加载分析库
        import librosa
        import numpy as np

        # 只读取前60秒，统一降采样到16000Hz
        y, sr = librosa.load(
            path,
            sr=16000,
            mono=True,
            duration=MAX_ANALYSIS_SECONDS
        )

        if len(y) == 0:
            raise ValueError("没有读取到有效音频数据")

        # BPM和节拍分析
        tempo, beats = librosa.beat.beat_track(
            y=y,
            sr=sr
        )

        # 调性估计
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr
        )

        key_index = int(np.argmax(chroma.mean(axis=1)))

        keys = [
            "C", "C#", "D", "D#", "E", "F",
            "F#", "G", "G#", "A", "A#", "B"
        ]

        return {
            "success": True,
            **info,
            "analyzed_seconds": round(len(y) / sr, 2),
            "sample_rate": sr,
            "bpm": round(float(np.asarray(tempo).reshape(-1)[0]), 2),
            "beat_count": int(len(beats)),
            "estimated_key": keys[key_index],
            "message": "音乐分析完成，仅分析音频前60秒"
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
