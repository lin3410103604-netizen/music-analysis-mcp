
import os
import time
import logging
import tempfile
import threading
from urllib.parse import urlparse

import requests
import numpy as np
import librosa

from fastmcp import FastMCP


# =========================================================
# 1. 基础配置
# =========================================================

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))

MAX_FILE_SIZE = 25 * 1024 * 1024
MAX_ANALYSIS_SECONDS = 600

SAMPLE_RATE = 11025
HOP_LENGTH = 512
N_FFT = 2048

ALLOWED_HOSTS = {
    "tempfile.org",
    "www.tempfile.org",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger("music-analysis")

mcp = FastMCP(
    "music-analysis",
    host=HOST,
    port=PORT,
    stateless_http=True,
)

# 免费实例资源有限，同一时间只运行一个重型分析任务
ANALYSIS_LOCK = threading.Lock()


# =========================================================
# 2. 日志与内存监控
# =========================================================

def memory_mb():
    """读取 Linux 进程当前常驻内存。"""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass

    return None


def log_stage(stage, started):
    elapsed = round(time.perf_counter() - started, 2)
    mem = memory_mb()

    logger.info(
        "[music-analysis] stage=%s elapsed=%ss memory=%sMB",
        stage,
        elapsed,
        mem if mem is not None else "unknown",
    )


def acquire_analysis_lock():
    return ANALYSIS_LOCK.acquire(blocking=False)


# =========================================================
# 3. 下载音频
# =========================================================

def download_audio(audio_url):
    parsed = urlparse(audio_url)

    if parsed.scheme != "https":
        raise ValueError("只允许 HTTPS 音频链接")

    if parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(
            "当前只允许 tempfile.org 音频链接"
        )

    if parsed.username or parsed.password:
        raise ValueError("音频链接格式不合法")

    response = requests.get(
        audio_url,
        stream=True,
        timeout=(15, 90),
        allow_redirects=False,
        headers={
            "User-Agent": "MusicAnalysisMCP/1.0"
        },
    )

    try:
        if response.status_code != 200:
            raise RuntimeError(
                f"音频下载失败，HTTP 状态码：{response.status_code}"
            )

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if "text/html" in content_type:
            raise ValueError(
                "链接返回的是网页，不是音频文件"
            )

        content_length = response.headers.get(
            "Content-Length"
        )

        if content_length:
            if int(content_length) > MAX_FILE_SIZE:
                raise ValueError("音频文件超过 25MB 限制")

        temp_path = None
        total_size = 0

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3",
        ) as temp_file:

            temp_path = temp_file.name

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if not chunk:
                    continue

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise ValueError(
                        "音频文件超过 25MB 限制"
                    )

                temp_file.write(chunk)

        if total_size == 0:
            raise ValueError("下载到的音频文件为空")

        return {
            "path": temp_path,
            "size_bytes": total_size,
            "content_type": content_type,
        }

    except Exception:
        if "temp_path" in locals() and temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise

    finally:
        response.close()


# =========================================================
# 4. 音乐调性估计
# =========================================================

KEY_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]

MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88
], dtype=np.float32)

MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17
], dtype=np.float32)


def estimate_key(chroma):
    chroma_mean = np.mean(chroma, axis=1)

    if np.std(chroma_mean) < 1e-8:
        return {
            "key": None,
            "mode": None,
            "confidence": 0,
            "note": "音高特征不足，无法可靠估计调性",
        }

    scores = []

    for i in range(12):
        major = np.roll(MAJOR_PROFILE, i)
        minor = np.roll(MINOR_PROFILE, i)

        major_corr = np.corrcoef(
            chroma_mean, major
        )[0, 1]

        minor_corr = np.corrcoef(
            chroma_mean, minor
        )[0, 1]

        scores.append((
            major_corr,
            KEY_NAMES[i],
            "major",
        ))

        scores.append((
            minor_corr,
            KEY_NAMES[i],
            "minor",
        ))

    scores = [
        item for item in scores
        if np.isfinite(item[0])
    ]

    if not scores:
        return {
            "key": None,
            "mode": None,
            "confidence": 0,
            "note": "调性估计失败",
        }

    scores.sort(key=lambda x: x[0], reverse=True)

    best = scores[0]
    second = scores[1]

    confidence = max(
        0.0,
        float(best[0] - second[0])
    )

    return {
        "key": best[1],
        "mode": best[2],
        "key_name": (
            f"{best[1]} {'大调' if best[2] == 'major' else '小调'}"
        ),
        "confidence": round(confidence, 4),
        "note": "算法估计结果，不等同于人工听辨确认",
    }


# =========================================================
# 5. 提取整首歌曲特征
# =========================================================

def load_audio_features(audio_path):
    started = time.perf_counter()

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        dtype=np.float32,
    )

    original_duration = len(y) / sr
    truncated = False

    if original_duration > MAX_ANALYSIS_SECONDS:
        y = y[:int(MAX_ANALYSIS_SECONDS * sr)]
        truncated = True

    duration = len(y) / sr

    if duration < 1:
        raise ValueError("音频时长过短，无法分析")

    log_stage("audio_loaded", started)

    # RMS 音量
    rms = librosa.feature.rms(
        y=y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    log_stage("rms", started)

    # 频谱中心
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    log_stage("spectral_centroid", started)

    # 频谱带宽
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    log_stage("spectral_bandwidth", started)

    # 频谱滚降
    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    log_stage("spectral_rolloff", started)

    # 过零率
    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    log_stage("zero_crossing_rate", started)

    # 色度特征
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    log_stage("chroma", started)

    # 节拍检测
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP_LENGTH,
        trim=False,
    )

    del onset_env

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    if isinstance(tempo, np.ndarray):
        bpm = float(np.asarray(tempo).reshape(-1)[0])
    else:
        bpm = float(tempo)

    log_stage("beat_tracking", started)

    # 统一各特征的帧数，避免时间轴错位
    feature_lengths = [
        len(rms),
        len(centroid),
        len(bandwidth),
        len(rolloff),
        len(zcr),
        chroma.shape[1],
    ]

    frame_count = min(feature_lengths)

    rms = rms[:frame_count]
    centroid = centroid[:frame_count]
    bandwidth = bandwidth[:frame_count]
    rolloff = rolloff[:frame_count]
    zcr = zcr[:frame_count]
    chroma = chroma[:, :frame_count]

    frame_times = librosa.frames_to_time(
        np.arange(frame_count),
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    key_result = estimate_key(chroma)

    log_stage("feature_extraction_complete", started)

    return {
        "y": y,
        "sr": sr,
        "duration": duration,
        "original_duration": original_duration,
        "truncated": truncated,
        "bpm": round(bpm, 2),
        "beat_times": beat_times,
        "beat_count": int(len(beat_times)),
        "rms": rms,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "zcr": zcr,
        "chroma": chroma,
        "frame_times": frame_times,
        "key": key_result,
    }


# =========================================================
# 6. 特征统计
# =========================================================

def safe_mean(values):
    if len(values) == 0:
        return 0.0

    return float(np.mean(values))


def build_segments(data, window_seconds=10):
    window_seconds = max(
        5, min(int(window_seconds), 30)
    )

    duration = data["duration"]
    frame_times = data["frame_times"]

    segments = []

    start = 0.0

    while start < duration:
        end = min(
            start + window_seconds,
            duration,
        )

        mask = (
            (frame_times >= start)
            & (frame_times < end)
        )

        if np.any(mask):
            rms_mean = safe_mean(
                data["rms"][mask]
            )

            segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),

                "rms": round(rms_mean, 5),

                "approx_dbfs": round(
                    20 * np.log10(
                        max(rms_mean, 1e-8)
                    ),
                    2,
                ),

                "spectral_centroid_hz": round(
                    safe_mean(data["centroid"][mask]),
                    2,
                ),

                "spectral_bandwidth_hz": round(
                    safe_mean(data["bandwidth"][mask]),
                    2,
                ),

                "spectral_rolloff_hz": round(
                    safe_mean(data["rolloff"][mask]),
                    2,
                ),

                "zero_crossing_rate": round(
                    safe_mean(data["zcr"][mask]),
                    5,
                ),

                "beat_count": int(
                    np.sum(
                        (data["beat_times"] >= start)
                        & (data["beat_times"] < end)
                    )
                ),
            })

        start = end

    return segments


# =========================================================
# 7. 音乐结构变化候选点
# =========================================================

def find_section_candidates(
    data,
    window_seconds=5,
    min_gap_seconds=12,
    max_candidates=12,
):
    window_seconds = max(
        5, min(int(window_seconds), 30)
    )

    min_gap_seconds = max(
        3, int(min_gap_seconds)
    )

    max_candidates = max(
        1, min(int(max_candidates), 30)
    )

    y = data["y"]
    sr = data["sr"]
    duration = data["duration"]

    started = time.perf_counter()

    # MFCC 只计算一次，不在每个时间窗口重复提取
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    log_stage("mfcc", started)

    frame_times = data["frame_times"]

    common_length = min(
        mfcc.shape[1],
        data["chroma"].shape[1],
        len(frame_times),
    )

    mfcc = mfcc[:, :common_length]
    chroma = data["chroma"][:, :common_length]
    frame_times = frame_times[:common_length]

    windows = []
    start = 0.0

    while start < duration:
        end = min(
            start + window_seconds,
            duration,
        )

        mask = (
            (frame_times >= start)
            & (frame_times < end)
        )

        if np.any(mask):
            mfcc_mean = np.mean(
                mfcc[:, mask],
                axis=1,
            )

            chroma_mean = np.mean(
                chroma[:, mask],
                axis=1,
            )

            vector = np.concatenate([
                mfcc_mean,
                chroma_mean,
            ]).astype(np.float32)

            norm = np.linalg.norm(vector)

            if norm > 1e-8:
                vector = vector / norm

            windows.append({
                "start": start,
                "end": end,
                "vector": vector,
            })

        start = end

    del mfcc

    if len(windows) < 2:
        return []

    changes = []

    for i in range(1, len(windows)):
        previous = windows[i - 1]["vector"]
        current = windows[i]["vector"]

        similarity = float(
            np.dot(previous, current)
        )

        distance = 1.0 - similarity

        changes.append({
            "time": round(
                windows[i]["start"], 2
            ),
            "distance": round(
                float(distance), 4
            ),
        })

    if not changes:
        return []

    distances = np.array(
        [item["distance"] for item in changes],
        dtype=np.float32,
    )

    threshold = float(
        np.mean(distances)
        + 0.5 * np.std(distances)
    )

    candidates = [
        item for item in changes
        if item["distance"] >= threshold
    ]

    candidates.sort(
        key=lambda item: item["distance"],
        reverse=True,
    )

    selected = []

    for item in candidates:
        if all(
            abs(item["time"] - old["time"])
            >= min_gap_seconds
            for old in selected
        ):
            selected.append(item)

        if len(selected) >= max_candidates:
            break

    selected.sort(
        key=lambda item: item["time"]
    )

    return [
        {
            "time": item["time"],
            "change_score": item["distance"],
            "description": "可能的音乐段落变化点，需结合听感确认",
        }
        for item in selected
    ]


# =========================================================
# 8. 统一错误处理
# =========================================================

def error_result(exc):
    logger.exception(
        "[music-analysis] analysis_failed"
    )

    return {
        "success": False,
        "error": str(exc),
        "memory_mb": memory_mb(),
    }


def cleanup_audio(path):
    if path:
        try:
            os.remove(path)
        except OSError:
            logger.warning(
                "临时音频文件清理失败"
            )


# =========================================================
# 9. MCP 工具一：测试音频下载
# =========================================================

@mcp.tool()
def test_audio_download(audio_url: str):
    """
    测试音频链接是否可访问，并检查文件大小和类型。
    """

    path = None

    try:
        result = download_audio(audio_url)
        path = result["path"]

        return {
            "success": True,
            "message": "音频下载成功",
            "size_bytes": result["size_bytes"],
            "size_mb": round(
                result["size_bytes"] / 1024 / 1024,
                2,
            ),
            "content_type": result["content_type"],
            "memory_mb": memory_mb(),
        }

    except Exception as exc:
        return error_result(exc)

    finally:
        cleanup_audio(path)


# =========================================================
# 10. MCP 工具二：全曲基础分析
# =========================================================

@mcp.tool()
def analyze_audio(audio_url: str):
    """
    分析整首音频的时长、BPM、节拍数量、调性和基础音乐特征。
    """

    if not acquire_analysis_lock():
        return {
            "success": False,
            "error": "已有分析任务正在运行，请稍后再试",
        }

    path = None

    try:
        started = time.perf_counter()

        downloaded = download_audio(audio_url)
        path = downloaded["path"]

        log_stage("download_complete", started)

        data = load_audio_features(path)

        result = {
            "success": True,
            "duration_seconds": round(
                data["duration"], 2
            ),
            "original_duration_seconds": round(
                data["original_duration"], 2
            ),
            "truncated": data["truncated"],
            "sample_rate": SAMPLE_RATE,
            "bpm": data["bpm"],
            "beat_count": data["beat_count"],
            "key": data["key"],
            "average_rms": round(
                safe_mean(data["rms"]), 5
            ),
            "average_centroid_hz": round(
                safe_mean(data["centroid"]), 2
            ),
            "average_bandwidth_hz": round(
                safe_mean(data["bandwidth"]), 2
            ),
            "average_rolloff_hz": round(
                safe_mean(data["rolloff"]), 2
            ),
            "average_zero_crossing_rate": round(
                safe_mean(data["zcr"]), 5
            ),
            "beat_times_seconds": [
                round(float(t), 3)
                for t in data["beat_times"][:500]
            ],
            "notes": [
                "BPM 和调性属于算法估计值",
                "节拍检测可能存在漏检或误检",
                "本工具不负责歌词识别",
            ],
            "memory_mb": memory_mb(),
        }

        log_stage("analysis_complete", started)

        del data

        return result

    except Exception as exc:
        return error_result(exc)

    finally:
        cleanup_audio(path)
        ANALYSIS_LOCK.release()


# =========================================================
# 11. MCP 工具三：分段音乐特征分析
# =========================================================

@mcp.tool()
def analyze_audio_segments(
    audio_url: str,
    window_seconds: int = 10,
):
    """
    按时间窗口分析音量、频谱、节拍密度等音乐特征。
    """

    if not acquire_analysis_lock():
        return {
            "success": False,
            "error": "已有分析任务正在运行，请稍后再试",
        }

    path = None

    try:
        started = time.perf_counter()

        downloaded = download_audio(audio_url)
        path = downloaded["path"]

        data = load_audio_features(path)

        segments = build_segments(
            data,
            window_seconds,
        )

        result = {
            "success": True,
            "duration_seconds": round(
                data["duration"], 2
            ),
            "window_seconds": max(
                5, min(int(window_seconds), 30)
            ),
            "segment_count": len(segments),
            "segments": segments,
            "memory_mb": memory_mb(),
        }

        log_stage("segments_complete", started)

        del data

        return result

    except Exception as exc:
        return error_result(exc)

    finally:
        cleanup_audio(path)
        ANALYSIS_LOCK.release()


# =========================================================
# 12. MCP 工具四：音乐结构变化检测
# =========================================================

@mcp.tool()
def detect_music_sections(
    audio_url: str,
    window_seconds: int = 5,
    min_gap_seconds: int = 12,
    max_candidates: int = 12,
):
    """
    检测音乐特征变化较明显的位置，提供可能的段落变化候选点。
    不会直接断定主歌、副歌或桥段。
    """

    if not acquire_analysis_lock():
        return {
            "success": False,
            "error": "已有分析任务正在运行，请稍后再试",
        }

    path = None

    try:
        started = time.perf_counter()

        downloaded = download_audio(audio_url)
        path = downloaded["path"]

        data = load_audio_features(path)

        candidates = find_section_candidates(
            data,
            window_seconds=window_seconds,
            min_gap_seconds=min_gap_seconds,
            max_candidates=max_candidates,
        )

        result = {
            "success": True,
            "duration_seconds": round(
                data["duration"], 2
            ),
            "candidate_count": len(candidates),
            "section_candidates": candidates,
            "warning": (
                "变化点不等于正式音乐结构标签。"
                "主歌、副歌、前奏等需要结合听感或其他信息判断。"
            ),
            "memory_mb": memory_mb(),
        }

        log_stage("section_detection_complete", started)

        del data

        return result

    except Exception as exc:
        return error_result(exc)

    finally:
        cleanup_audio(path)
        ANALYSIS_LOCK.release()


# =========================================================
# 13. MCP 工具五：生成完整音乐时间轴
# =========================================================

@mcp.tool()
def generate_music_timeline(
    audio_url: str,
    window_seconds: int = 10,
):
    """
    一次性生成整首音乐的基础分析、分段特征和结构变化候选点。
    """

    if not acquire_analysis_lock():
        return {
            "success": False,
            "error": "已有分析任务正在运行，请稍后再试",
        }

    path = None

    try:
        started = time.perf_counter()

        downloaded = download_audio(audio_url)
        path = downloaded["path"]

        log_stage("download_complete", started)

        # 只加载和分析一次
        data = load_audio_features(path)

        log_stage("features_ready", started)

        segments = build_segments(
            data,
            window_seconds,
        )

        log_stage("segments_ready", started)

        candidates = find_section_candidates(
            data,
            window_seconds=5,
            min_gap_seconds=12,
            max_candidates=20,
        )

        log_stage("section_candidates_ready", started)

        timeline = []

        for segment in segments:
            timeline.append({
                "time": segment["start"],
                "type": "audio_segment",
                "data": segment,
            })

        for candidate in candidates:
            timeline.append({
                "time": candidate["time"],
                "type": "possible_section_change",
                "data": candidate,
            })

        timeline.sort(
            key=lambda item: item["time"]
        )

        result = {
            "success": True,

            "song": {
                "duration_seconds": round(
                    data["duration"], 2
                ),
                "original_duration_seconds": round(
                    data["original_duration"], 2
                ),
                "truncated": data["truncated"],
                "sample_rate": SAMPLE_RATE,
            },

            "music_overview": {
                "bpm": data["bpm"],
                "beat_count": data["beat_count"],
                "key": data["key"],
                "average_rms": round(
                    safe_mean(data["rms"]), 5
                ),
                "average_centroid_hz": round(
                    safe_mean(data["centroid"]), 2
                ),
                "average_bandwidth_hz": round(
                    safe_mean(data["bandwidth"]), 2
                ),
            },

            "segments": segments,

            "section_candidates": candidates,

            "timeline": timeline,

            "limitations": [
                "音乐变化候选点不等于正式段落标签",
                "不能仅凭频谱特征准确判断歌词内容",
                "情绪和叙事含义需要结合歌词、听感及背景资料分析",
                "本工具不输出完整歌词",
            ],

            "memory_mb": memory_mb(),
        }

        log_stage("timeline_complete", started)

        del data

        return result

    except Exception as exc:
        return error_result(exc)

    finally:
        cleanup_audio(path)
        ANALYSIS_LOCK.release()


# =========================================================
# 14. 启动 MCP 服务
# =========================================================

if __name__ == "__main__":
    logger.info(
        "Starting music-analysis MCP on %s:%s",
        HOST,
        PORT,
    )

    mcp.run(transport="http")
