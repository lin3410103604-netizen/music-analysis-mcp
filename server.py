
import os
import tempfile
from urllib.parse import urlparse

import requests
import numpy as np
from fastmcp import FastMCP

mcp = FastMCP("Music Analysis")

# -----------------------------
# 基础配置：适配 Render Free
# -----------------------------

MAX_FILE_SIZE = 25 * 1024 * 1024
MAX_ANALYSIS_SECONDS = 600
SAMPLE_RATE = 11025
HOP_LENGTH = 512

ALLOWED_HOSTS = {
    "tempfile.org",
    "www.tempfile.org"
}


# -----------------------------
# 音频下载与安全检查
# -----------------------------

def validate_audio_url(audio_url):
    parsed = urlparse(audio_url)

    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "目前只允许 tempfile.org 的 HTTPS 音频链接"
        )


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
            raise ValueError(
                "音频链接发生重定向，请使用直接下载链接"
            )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", "unknown"
        )

        if "text/html" in content_type.lower():
            raise ValueError("下载到的是网页，不是音频文件")

        content_length = response.headers.get("Content-Length")

        if content_length:
            if int(content_length) > MAX_FILE_SIZE:
                raise ValueError("文件超过25MB限制")

        with tempfile.NamedTemporaryFile(
            suffix=".audio",
            delete=False
        ) as f:
            path = f.name

            for chunk in response.iter_content(
                chunk_size=64 * 1024
            ):
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
            "downloaded_MB": round(
                downloaded / 1024 / 1024, 2
            ),
            "final_url": response.url
        }

    except Exception:
        if path and os.path.exists(path):
            os.remove(path)
        raise

    finally:
        response.close()


# -----------------------------
# 音频读取
# -----------------------------

def load_audio_features(path):
    import librosa

    y, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=MAX_ANALYSIS_SECONDS
    )

    if len(y) == 0:
        raise ValueError("没有读取到有效音频数据")

    duration = len(y) / sr

    # 基础特征
    rms = librosa.feature.rms(
        y=y,
        hop_length=HOP_LENGTH
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y=y,
        hop_length=HOP_LENGTH
    )[0]

    onset = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH
    )

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH
    )

    # BPM 和节拍
    tempo, beats = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        trim=False
    )

    bpm = float(
        np.asarray(tempo).reshape(-1)[0]
    )

    beat_times = librosa.frames_to_time(
        beats,
        sr=sr,
        hop_length=HOP_LENGTH
    )

    # 调性估计：同时比较大调和小调
    major_profile = np.array([
        6.35, 2.23, 3.48, 2.33,
        4.38, 4.09, 2.52, 5.19,
        2.39, 3.66, 2.29, 2.88
    ])

    minor_profile = np.array([
        6.33, 2.68, 3.52, 5.38,
        2.60, 3.53, 2.54, 4.75,
        3.98, 2.69, 3.34, 3.17
    ])

    chroma_mean = chroma.mean(axis=1)

    key_names = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ]

    candidates = []

    for i in range(12):
        rotated = np.roll(chroma_mean, -i)

        for mode, profile in [
            ("major", major_profile),
            ("minor", minor_profile)
        ]:
            correlation = np.corrcoef(
                rotated, profile
            )[0, 1]

            if np.isfinite(correlation):
                candidates.append((
                    float(correlation),
                    key_names[i],
                    mode
                ))

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    if candidates:
        best = candidates[0]
        estimated_key = best[1] + " " + best[2]
        key_confidence = round(best[0], 3)
    else:
        estimated_key = "unknown"
        key_confidence = None

    # 时间轴
    frame_times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=HOP_LENGTH
    )

    # 统一帧数，避免不同特征长度不一致
    feature_arrays = {
        "rms": rms,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "zcr": zcr
    }

    return {
        "y": y,
        "sr": sr,
        "duration": duration,
        "bpm": bpm,
        "beats": beat_times,
        "key": estimated_key,
        "key_confidence": key_confidence,
        "frame_times": frame_times,
        "features": feature_arrays,
        "chroma": chroma,
        "onset": onset
    }


# -----------------------------
# 分段特征统计
# -----------------------------

def build_segments(data, window_seconds=10):
    frame_times = data["frame_times"]
    features = data["features"]
    beats = data["beats"]
    duration = data["duration"]

    results = []

    start = 0.0

    while start < duration:
        end = min(
            start + window_seconds,
            duration
        )

        mask = (
            (frame_times >= start)
            & (frame_times < end)
        )

        if not np.any(mask):
            start = end
            continue

        def mean_value(name):
            return float(
                np.mean(features[name][mask])
            )

        rms_value = mean_value("rms")

        segment_beats = beats[
            (beats >= start) & (beats < end)
        ]

        results.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "rms_energy": round(rms_value, 5),
            "energy_dbfs_approx": round(
                20 * np.log10(max(rms_value, 1e-8)),
                2
            ),
            "spectral_centroid_hz": round(
                mean_value("centroid"), 2
            ),
            "spectral_bandwidth_hz": round(
                mean_value("bandwidth"), 2
            ),
            "spectral_rolloff_hz": round(
                mean_value("rolloff"), 2
            ),
            "zero_crossing_rate": round(
                mean_value("zcr"), 5
            ),
            "beat_count": int(len(segment_beats)),
            "beat_density_per_second": round(
                len(segment_beats) / max(end - start, 0.01),
                3
            )
        })

        start = end

    return results


# -----------------------------
# 音乐变化检测
# -----------------------------

def find_section_candidates(
    data,
    window_seconds=5,
    min_gap_seconds=12,
    max_candidates=12
):
    import librosa

    y = data["y"]
    sr = data["sr"]

    # MFCC 用于描述不同时间段的音色特征
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        hop_length=HOP_LENGTH
    )

    chroma = data["chroma"]
    frame_times = data["frame_times"]
    duration = data["duration"]

    vectors = []
    window_starts = []

    start = 0.0

    while start + window_seconds <= duration:
        end = start + window_seconds

        mask = (
            (frame_times >= start)
            & (frame_times < end)
        )

        if np.any(mask):
            mfcc_mean = mfcc[:, mask].mean(axis=1)
            chroma_mean = chroma[:, mask].mean(axis=1)

            vector = np.concatenate([
                mfcc_mean,
                chroma_mean
            ])

            vectors.append(vector)
            window_starts.append(start)

        start += window_seconds

    if len(vectors) < 3:
        return []

    vectors = np.asarray(vectors)

    # 对特征向量进行标准化
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0) + 1e-6

    normalized = (vectors - mean) / std

    scores = []

    for i in range(1, len(normalized)):
        a = normalized[i - 1]
        b = normalized[i]

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator == 0:
            score = 0.0
        else:
            score = 1 - float(
                np.dot(a, b) / denominator
            )

        scores.append({
            "time": window_starts[i],
            "score": max(0.0, score)
        })

    if not scores:
        return []

    score_values = np.array([
        item["score"] for item in scores
    ])

    threshold = (
        float(np.median(score_values))
        + 0.75 * float(np.std(score_values))
    )

    candidates = [
        item for item in scores
        if item["score"] >= threshold
    ]

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    selected = []

    for item in candidates:
        if all(
            abs(item["time"] - chosen["time"])
            >= min_gap_seconds
            for chosen in selected
        ):
            selected.append(item)

        if len(selected) >= max_candidates:
            break

    selected.sort(key=lambda item: item["time"])

    return [
        {
            "time": round(item["time"], 2),
            "feature_change_score": round(
                item["score"], 4
            ),
            "label": "possible_section_change"
        }
        for item in selected
    ]


# -----------------------------
# 工具1：下载测试
# -----------------------------

@mcp.tool()
def test_audio_download(audio_url: str) -> dict:
    """测试音频能否下载，不进行音乐分析。"""

    path = None

    try:
        path, info = download_audio(audio_url)

        return {
            "success": True,
            **info,
            "message": "下载读取完成，未进行音乐分析"
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


# -----------------------------
# 工具2：全曲音乐分析
# -----------------------------

@mcp.tool()
def analyze_audio(audio_url: str) -> dict:
    """
    分析整首歌曲，返回 BPM、节拍、调性、
    能量、音色及频谱特征。
    """

    path = None

    try:
        path, info = download_audio(audio_url)
        data = load_audio_features(path)

        y = data["y"]

        return {
            "success": True,
            **info,
            "analyzed_seconds": round(
                data["duration"], 2
            ),
            "sample_rate": data["sr"],
            "bpm": round(data["bpm"], 2),
            "beat_count": int(len(data["beats"])),
            "beat_times_seconds": [
                round(float(t), 2)
                for t in data["beats"][:500]
            ],
            "estimated_key": data["key"],
            "key_confidence": data["key_confidence"],
            "average_rms_energy": round(
                float(np.mean(data["features"]["rms"])),
                5
            ),
            "average_spectral_centroid_hz": round(
                float(np.mean(
                    data["features"]["centroid"]
                )), 2
            ),
            "average_spectral_bandwidth_hz": round(
                float(np.mean(
                    data["features"]["bandwidth"]
                )), 2
            ),
            "average_spectral_rolloff_hz": round(
                float(np.mean(
                    data["features"]["rolloff"]
                )), 2
            ),
            "average_zero_crossing_rate": round(
                float(np.mean(
                    data["features"]["zcr"]
                )), 5
            ),
            "message": (
                "全曲基础音乐特征分析完成。"
                "调性为算法估计，需结合听感核验。"
            )
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


# -----------------------------
# 工具3：分段音乐特征
# -----------------------------

@mcp.tool()
def analyze_audio_segments(
    audio_url: str,
    window_seconds: int = 10
) -> dict:
    """
    按时间窗口分析音乐能量、音色、
    频谱和节拍密度变化。
    """

    path = None

    try:
        if window_seconds < 5 or window_seconds > 30:
            raise ValueError(
                "window_seconds 必须在5到30秒之间"
            )

        path, info = download_audio(audio_url)
        data = load_audio_features(path)

        segments = build_segments(
            data,
            window_seconds
        )

        return {
            "success": True,
            **info,
            "analyzed_seconds": round(
                data["duration"], 2
            ),
            "window_seconds": window_seconds,
            "segment_count": len(segments),
            "segments": segments,
            "message": (
                "分段音乐特征分析完成。"
                "能量值为RMS近似值，不是LUFS响度。"
            )
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


# -----------------------------
# 工具4：音乐结构变化检测
# -----------------------------

@mcp.tool()
def detect_music_sections(
    audio_url: str,
    window_seconds: int = 5,
    min_gap_seconds: int = 12,
    max_candidates: int = 12
) -> dict:
    """
    根据MFCC和色度特征变化寻找疑似音乐段落边界。
    不直接判断主歌、副歌等语义标签。
    """

    path = None

    try:
        if window_seconds < 3 or window_seconds > 15:
            raise ValueError(
                "window_seconds 必须在3到15秒之间"
            )

        if min_gap_seconds < 5:
            raise ValueError(
                "min_gap_seconds 不能小于5秒"
            )

        if max_candidates < 1 or max_candidates > 20:
            raise ValueError(
                "max_candidates 必须在1到20之间"
            )

        path, info = download_audio(audio_url)
        data = load_audio_features(path)

        candidates = find_section_candidates(
            data,
            window_seconds=window_seconds,
            min_gap_seconds=min_gap_seconds,
            max_candidates=max_candidates
        )

        return {
            "success": True,
            **info,
            "analyzed_seconds": round(
                data["duration"], 2
            ),
            "section_candidates": candidates,
            "candidate_count": len(candidates),
            "warning": (
                "这些是声学特征变化候选点，"
                "不等于已经确认的主歌、副歌或间奏边界。"
            )
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


# -----------------------------
# 工具5：生成音乐时间轴
# -----------------------------

@mcp.tool()
def generate_music_timeline(
    audio_url: str,
    window_seconds: int = 10
) -> dict:
    """
    整合全曲基础分析、分段特征和音乐变化候选点，
    生成供Claude进行音乐理解的时间轴。
    """

    path = None

    try:
        if window_seconds < 5 or window_seconds > 30:
            raise ValueError(
                "window_seconds 必须在5到30秒之间"
            )

        path, info = download_audio(audio_url)
        data = load_audio_features(path)

        segments = build_segments(
            data,
            window_seconds
        )

        candidates = find_section_candidates(
            data,
            window_seconds=5,
            min_gap_seconds=12,
            max_candidates=12
        )

        timeline = []

        for segment in segments[:60]:
            timeline.append({
                "time": segment["start"],
                "type": "segment_summary",
                "energy": segment["rms_energy"],
                "spectral_centroid_hz": (
                    segment["spectral_centroid_hz"]
                ),
                "beat_density": (
                    segment["beat_density_per_second"]
                )
            })

        for candidate in candidates:
            timeline.append({
                "time": candidate["time"],
                "type": "possible_section_change",
                "score": candidate[
                    "feature_change_score"
                ]
            })

        timeline.sort(key=lambda item: item["time"])

        return {
            "success": True,
            **info,
            "duration_seconds": round(
                data["duration"], 2
            ),
            "global_analysis": {
                "bpm": round(data["bpm"], 2),
                "beat_count": int(len(data["beats"])),
                "estimated_key": data["key"],
                "key_confidence": data["key_confidence"]
            },
            "segments": segments[:60],
            "section_candidates": candidates,
            "timeline": timeline,
            "message": (
                "音乐时间轴生成完成。"
                "请结合歌词和听感进一步确认音乐段落及情绪。"
            )
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


# -----------------------------
# MCP 服务启动
# -----------------------------

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        stateless_http=True
    )
