import os
import time
import uuid
import logging
import tempfile
import threading
import json
import csv
import io
import secrets
from pathlib import Path
from urllib.parse import urlparse, quote

import requests
import numpy as np
import librosa
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from psycopg import connect as pg_connect
from psycopg.rows import dict_row


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

ALLOWED_HOSTS = {"tempfile.org", "www.tempfile.org"}

# PostgreSQL 保存任务；内存字典仅作为当前进程的快速缓存。
# 默认保留任务 30 天，避免短时间内因清理规则丢失结果；可通过环境变量调整。
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", str(30 * 24 * 60 * 60)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("music-analysis")

mcp = FastMCP("music-analysis")

# 同一时间只运行一个重型分析任务，避免免费实例内存被多个任务占满。
ANALYSIS_LOCK = threading.Lock()

# 异步任务状态保存在当前 Python 进程内。
JOBS = {}
JOBS_LOCK = threading.Lock()

# Render Free 的本地文件系统不作为持久化存储。
# 所有任务记录和导出文件内容均保存到 Neon PostgreSQL。
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
DOWNLOAD_TTL_SECONDS = int(os.environ.get("DOWNLOAD_TTL_SECONDS", str(30 * 24 * 60 * 60)))

# Keep MCP tool responses comfortably below common gateway/client limits.
MAX_INLINE_RESULT_CHARS = 12000


def db_connect():
    if not DATABASE_URL:
        raise RuntimeError("缺少 DATABASE_URL 环境变量，无法连接 Neon PostgreSQL")
    # Neon pooled connection string 使用 SSL；关闭客户端 prepared statements
    # 以兼容 PgBouncer transaction pooling。每次操作使用短连接，避免空闲连接堆积。
    return pg_connect(
        DATABASE_URL,
        row_factory=dict_row,
        connect_timeout=10,
        prepare_threshold=None,
        application_name="music-analysis-mcp",
    )


def init_database():
    with db_connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            stage TEXT,
            progress INTEGER,
            created_at DOUBLE PRECISION,
            updated_at DOUBLE PRECISION,
            window_seconds INTEGER,
            result_json TEXT,
            error TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS exports (
            token TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            content BYTEA NOT NULL,
            created_at DOUBLE PRECISION NOT NULL
        )""")
        conn.commit()
    logger.info("[music-analysis] PostgreSQL schema ready; export content stored in database")


def persist_job(job):
    result_json = json.dumps(job.get("result"), ensure_ascii=False, separators=(",", ":")) if job.get("result") is not None else None
    with db_connect() as conn:
        conn.execute("""INSERT INTO jobs
            (job_id,status,stage,progress,created_at,updated_at,window_seconds,result_json,error)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(job_id) DO UPDATE SET
              status=excluded.status, stage=excluded.stage, progress=excluded.progress,
              updated_at=excluded.updated_at, window_seconds=excluded.window_seconds,
              result_json=COALESCE(excluded.result_json, jobs.result_json), error=excluded.error
        """, (job["job_id"], job.get("status", "queued"), job.get("stage"),
              job.get("progress", 0), job.get("created_at", time.time()),
              job.get("updated_at", time.time()), job.get("window_seconds", 10),
              result_json, job.get("error")))
        conn.commit()


def load_persisted_jobs():
    try:
        with db_connect() as conn:
            rows = conn.execute("SELECT * FROM jobs").fetchall()
        now = time.time()
        with JOBS_LOCK:
            for row in rows:
                if row["status"] in ("queued", "running"):
                    # 进程重启后，后台线程不可能继续运行；明确标记为中断。
                    status = "interrupted"
                    stage = "interrupted_after_restart"
                    error = "服务进程重启，原后台任务已中断；请使用已有结果或重新启动分析。"
                else:
                    status, stage, error = row["status"], row["stage"], row["error"]
                result = json.loads(row["result_json"]) if row["result_json"] else None
                JOBS[row["job_id"]] = {
                    "job_id": row["job_id"], "status": status, "stage": stage,
                    "progress": row["progress"] or 0, "created_at": row["created_at"],
                    "updated_at": row["updated_at"], "window_seconds": row["window_seconds"] or 10,
                    "result": result, "error": error,
                }
                if status == "interrupted":
                    persist_job(JOBS[row["job_id"]])
    except Exception:
        logger.exception("[music-analysis] failed_to_load_persisted_jobs")


init_database()


# =========================================================
# 2. 日志、内存监控与任务状态
# =========================================================

def memory_mb():
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


def cleanup_old_jobs():
    """清理过期任务与下载记录。"""
    now = time.time()
    with JOBS_LOCK:
        expired = [job_id for job_id, job in JOBS.items()
                   if job.get("status") in ("done", "failed", "interrupted")
                   and now - job.get("updated_at", now) > JOB_TTL_SECONDS]
        for job_id in expired:
            JOBS.pop(job_id, None)
    if expired:
        with db_connect() as conn:
            conn.executemany("DELETE FROM jobs WHERE job_id=%s", [(x,) for x in expired])
            conn.commit()
    with db_connect() as conn:
        conn.execute("DELETE FROM exports WHERE created_at < %s", (now - DOWNLOAD_TTL_SECONDS,))
        conn.commit()


def update_job(job_id, **updates):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            job.update(updates)
            job["updated_at"] = time.time()
            snapshot = dict(job)
        else:
            snapshot = None
    if snapshot is not None:
        persist_job(snapshot)


# =========================================================
# 3. 下载音频
# =========================================================

def download_audio(audio_url):
    parsed = urlparse(audio_url)

    if parsed.scheme != "https":
        raise ValueError("只允许 HTTPS 音频链接")
    if parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError("当前只允许 tempfile.org 音频链接")
    if parsed.username or parsed.password:
        raise ValueError("音频链接格式不合法")

    response = requests.get(
        audio_url,
        stream=True,
        timeout=(15, 90),
        allow_redirects=False,
        headers={"User-Agent": "MusicAnalysisMCP/1.0"},
    )

    temp_path = None
    try:
        if response.status_code != 200:
            raise RuntimeError(
                f"音频下载失败，HTTP 状态码：{response.status_code}"
            )

        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            raise ValueError("链接返回的是网页，不是音频文件")

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise ValueError("音频文件超过 25MB 限制")

        total_size = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_path = temp_file.name
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise ValueError("音频文件超过 25MB 限制")
                temp_file.write(chunk)

        if total_size == 0:
            raise ValueError("下载到的音频文件为空")

        return {
            "path": temp_path,
            "size_bytes": total_size,
            "content_type": content_type,
        }
    except Exception:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise
    finally:
        response.close()


def cleanup_audio(path):
    if path:
        try:
            os.remove(path)
        except OSError:
            logger.warning("临时音频文件清理失败")


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
            "key": None, "mode": None, "confidence": 0,
            "note": "音高特征不足，无法可靠估计调性",
        }

    scores = []
    for i in range(12):
        major_corr = np.corrcoef(
            chroma_mean, np.roll(MAJOR_PROFILE, i)
        )[0, 1]
        minor_corr = np.corrcoef(
            chroma_mean, np.roll(MINOR_PROFILE, i)
        )[0, 1]
        scores.append((major_corr, KEY_NAMES[i], "major"))
        scores.append((minor_corr, KEY_NAMES[i], "minor"))

    scores = [item for item in scores if np.isfinite(item[0])]
    if not scores:
        return {
            "key": None, "mode": None, "confidence": 0,
            "note": "调性估计失败",
        }

    scores.sort(key=lambda x: x[0], reverse=True)
    best, second = scores[0], scores[1]
    return {
        "key": best[1],
        "mode": best[2],
        "key_name": f"{best[1]} {'大调' if best[2] == 'major' else '小调'}",
        "confidence": round(max(0.0, float(best[0] - second[0])), 4),
        "note": "算法估计结果，不等同于人工听辨确认",
    }


# =========================================================
# 5. 提取整首歌曲特征
# =========================================================

def load_audio_features(audio_path):
    started = time.perf_counter()

    y, sr = librosa.load(
        audio_path, sr=SAMPLE_RATE, mono=True, dtype=np.float32
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

    rms = librosa.feature.rms(
        y=y, frame_length=N_FFT, hop_length=HOP_LENGTH
    )[0]
    log_stage("rms", started)

    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )[0]
    log_stage("spectral_centroid", started)

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )[0]
    log_stage("spectral_bandwidth", started)

    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )[0]
    log_stage("spectral_rolloff", started)

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=N_FFT, hop_length=HOP_LENGTH
    )[0]
    log_stage("zero_crossing_rate", started)

    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_stage("chroma", started)

    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP_LENGTH,
        trim=False,
    )
    beat_times = librosa.frames_to_time(
        beat_frames, sr=sr, hop_length=HOP_LENGTH
    )
    bpm = (
        float(np.asarray(tempo).reshape(-1)[0])
        if isinstance(tempo, np.ndarray)
        else float(tempo)
    )
    log_stage("beat_tracking", started)

    frame_count = min(
        len(rms), len(centroid), len(bandwidth),
        len(rolloff), len(zcr), chroma.shape[1],
    )
    rms = rms[:frame_count]
    centroid = centroid[:frame_count]
    bandwidth = bandwidth[:frame_count]
    rolloff = rolloff[:frame_count]
    zcr = zcr[:frame_count]
    chroma = chroma[:, :frame_count]

    frame_times = librosa.frames_to_time(
        np.arange(frame_count), sr=sr, hop_length=HOP_LENGTH
    )
    key_result = estimate_key(chroma)
    log_stage("feature_extraction_complete", started)

    return {
        "y": y, "sr": sr, "duration": duration,
        "original_duration": original_duration,
        "truncated": truncated,
        "bpm": round(bpm, 2),
        "beat_times": beat_times,
        "onset_env": onset_env,
        "beat_frames": beat_frames,
        "beat_count": int(len(beat_times)),
        "rms": rms, "centroid": centroid,
        "bandwidth": bandwidth, "rolloff": rolloff,
        "zcr": zcr, "chroma": chroma,
        "frame_times": frame_times, "key": key_result,
    }


# =========================================================
# 6. 节拍、重音与低能量区间分析
# =========================================================

def analyze_rhythm_details(data):
    """输出节拍时间点、相对强弱、重音候选和低能量区间。

    强弱标签仅表示本曲内部的相对起音强度，不代表已识别出准确拍号/小节首拍。
    """
    beat_times = np.asarray(data["beat_times"], dtype=np.float32)
    onset_env = np.asarray(data["onset_env"], dtype=np.float32)
    rms = np.asarray(data["rms"], dtype=np.float32)
    frame_times = np.asarray(data["frame_times"], dtype=np.float32)
    duration = float(data["duration"])

    if len(beat_times) and len(onset_env):
        beat_frames = np.clip(
            np.rint(beat_times * SAMPLE_RATE / HOP_LENGTH).astype(int),
            0, len(onset_env) - 1,
        )
        beat_scores = onset_env[beat_frames]
        strong_threshold = float(np.percentile(beat_scores, 70))
        beat_points = [{
            "time": round(float(t), 3),
            "relative_strength": round(float(score), 4),
            "accent_class": "相对强拍候选" if score >= strong_threshold else "相对弱拍候选",
        } for t, score in zip(beat_times, beat_scores)]
    else:
        beat_points = []
        beat_scores = np.array([], dtype=np.float32)
        strong_threshold = 0.0

    # 起音包络局部峰值：作为音乐重音候选，不直接等同于鼓点或小节重拍。
    accent_candidates = []
    if len(onset_env) >= 3:
        threshold = float(np.percentile(onset_env, 85))
        min_distance_frames = max(1, int(round(0.25 * SAMPLE_RATE / HOP_LENGTH)))
        last_kept = -min_distance_frames
        for idx in range(1, len(onset_env) - 1):
            value = float(onset_env[idx])
            if (value >= threshold and value >= onset_env[idx - 1]
                    and value > onset_env[idx + 1]
                    and idx - last_kept >= min_distance_frames):
                t = float(frame_times[idx]) if idx < len(frame_times) else idx * HOP_LENGTH / SAMPLE_RATE
                accent_candidates.append({
                    "time": round(t, 3),
                    "strength": round(value, 4),
                    "description": "起音强度峰值候选，需结合听感确认",
                })
                last_kept = idx

    # 用约 0.5 秒滑动平均 RMS 寻找持续低能量区间，避免把单帧波动当作停顿。
    low_energy_intervals = []
    if len(rms) and len(frame_times):
        frame_step = HOP_LENGTH / SAMPLE_RATE
        window_frames = max(1, int(round(0.5 / frame_step)))
        kernel = np.ones(window_frames, dtype=np.float32) / window_frames
        smooth_rms = np.convolve(rms, kernel, mode="same")
        baseline = float(np.percentile(smooth_rms, 20))
        silence_threshold = max(0.002, baseline * 0.35)
        quiet = smooth_rms <= silence_threshold
        starts = np.where(quiet & ~np.r_[False, quiet[:-1]])[0]
        ends = np.where(quiet & ~np.r_[quiet[1:], False])[0] + 1
        for start_idx, end_idx in zip(starts, ends):
            start_t = float(frame_times[min(start_idx, len(frame_times) - 1)])
            end_idx = min(end_idx, len(frame_times) - 1)
            end_t = float(frame_times[end_idx]) + frame_step
            if end_t - start_t >= 0.5:
                low_energy_intervals.append({
                    "start": round(max(0.0, start_t), 3),
                    "end": round(min(duration, end_t), 3),
                    "duration": round(min(duration, end_t) - max(0.0, start_t), 3),
                    "description": "持续低能量区间，可能是停顿、尾奏或弱音段，需听感确认",
                })

    last_accent = max(accent_candidates, key=lambda x: x["time"]) if accent_candidates else None
    return {
        "beat_points": beat_points,
        "beat_point_count": len(beat_points),
        "relative_strong_beat_threshold": round(strong_threshold, 4),
        "accent_candidates": accent_candidates,
        "low_energy_intervals": low_energy_intervals,
        "last_accent_candidate": last_accent,
        "notes": [
            "强拍/弱拍是根据本曲节拍位置的相对起音强度分类，不代表准确拍号或小节首拍。",
            "重音候选来自起音强度峰值，可能包含鼓点、和弦起音或其他瞬态。",
            "低能量区间不一定是真正静音，也可能是弱音、尾奏或编曲留白。",
            "最后一个重音候选不一定等于歌曲最后一个重要音乐事件。",
        ],
    }


# =========================================================
# 7. 特征统计
# =========================================================

def safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def build_segments(data, window_seconds=10):
    window_seconds = max(5, min(int(window_seconds), 30))
    duration = data["duration"]
    frame_times = data["frame_times"]
    segments = []
    start = 0.0

    while start < duration:
        end = min(start + window_seconds, duration)
        mask = (frame_times >= start) & (frame_times < end)

        if np.any(mask):
            rms_mean = safe_mean(data["rms"][mask])
            segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "rms": round(rms_mean, 5),
                "approx_dbfs": round(
                    20 * np.log10(max(rms_mean, 1e-8)), 2
                ),
                "spectral_centroid_hz": round(
                    safe_mean(data["centroid"][mask]), 2
                ),
                "spectral_bandwidth_hz": round(
                    safe_mean(data["bandwidth"][mask]), 2
                ),
                "spectral_rolloff_hz": round(
                    safe_mean(data["rolloff"][mask]), 2
                ),
                "zero_crossing_rate": round(
                    safe_mean(data["zcr"][mask]), 5
                ),
                "beat_count": int(np.sum(
                    (data["beat_times"] >= start)
                    & (data["beat_times"] < end)
                )),
            })
        start = end

    return segments


# =========================================================
# 7. 音乐结构变化候选点
# =========================================================

def find_section_candidates(
    data, window_seconds=5, min_gap_seconds=12, max_candidates=12
):
    window_seconds = max(5, min(int(window_seconds), 30))
    min_gap_seconds = max(3, int(min_gap_seconds))
    max_candidates = max(1, min(int(max_candidates), 30))

    y, sr, duration = data["y"], data["sr"], data["duration"]
    started = time.perf_counter()

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_stage("mfcc", started)

    common_length = min(
        mfcc.shape[1], data["chroma"].shape[1],
        len(data["frame_times"]),
    )
    mfcc = mfcc[:, :common_length]
    chroma = data["chroma"][:, :common_length]
    frame_times = data["frame_times"][:common_length]

    windows = []
    start = 0.0
    while start < duration:
        end = min(start + window_seconds, duration)
        mask = (frame_times >= start) & (frame_times < end)

        if np.any(mask):
            vector = np.concatenate([
                np.mean(mfcc[:, mask], axis=1),
                np.mean(chroma[:, mask], axis=1),
            ]).astype(np.float32)
            norm = np.linalg.norm(vector)
            if norm > 1e-8:
                vector /= norm
            windows.append({
                "start": start,
                "vector": vector,
            })
        start = end

    del mfcc
    if len(windows) < 2:
        return []

    changes = []
    for i in range(1, len(windows)):
        distance = 1.0 - float(np.dot(
            windows[i - 1]["vector"], windows[i]["vector"]
        ))
        changes.append({
            "time": round(windows[i]["start"], 2),
            "distance": round(float(distance), 4),
        })

    if not changes:
        return []

    distances = np.array(
        [item["distance"] for item in changes], dtype=np.float32
    )
    threshold = float(np.mean(distances) + 0.5 * np.std(distances))
    candidates = [
        item for item in changes if item["distance"] >= threshold
    ]
    candidates.sort(key=lambda item: item["distance"], reverse=True)

    selected = []
    for item in candidates:
        if all(
            abs(item["time"] - old["time"]) >= min_gap_seconds
            for old in selected
        ):
            selected.append(item)
        if len(selected) >= max_candidates:
            break

    selected.sort(key=lambda item: item["time"])
    return [{
        "time": item["time"],
        "change_score": item["distance"],
        "description": "可能的音乐段落变化点，需结合听感确认",
    } for item in selected]


# =========================================================
# 8. 统一分析逻辑与错误处理
# =========================================================

def error_result(exc):
    logger.exception("[music-analysis] analysis_failed")
    return {
        "success": False,
        "error": str(exc),
        "memory_mb": memory_mb(),
    }


def run_music_timeline_analysis(audio_url, window_seconds=10, job_id=None):
    """实际执行完整分析。可由同步工具或后台线程调用。"""
    path = None
    started = time.perf_counter()

    try:
        if job_id:
            update_job(job_id, stage="downloading", progress=5)

        downloaded = download_audio(audio_url)
        path = downloaded["path"]
        log_stage("download_complete", started)

        if job_id:
            update_job(job_id, stage="extracting_audio_features", progress=15)

        data = load_audio_features(path)
        if job_id:
            update_job(job_id, stage="building_segments", progress=65)

        segments = build_segments(data, window_seconds)
        rhythm_details = analyze_rhythm_details(data)
        log_stage("segments_ready", started)

        if job_id:
            update_job(job_id, stage="detecting_section_changes", progress=75)

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
        timeline.sort(key=lambda item: item["time"])

        result = {
            "success": True,
            "song": {
                "duration_seconds": round(data["duration"], 2),
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
                "average_rms": round(safe_mean(data["rms"]), 5),
                "average_centroid_hz": round(
                    safe_mean(data["centroid"]), 2
                ),
                "average_bandwidth_hz": round(
                    safe_mean(data["bandwidth"]), 2
                ),
            },
            "segments": segments,
            "rhythm_analysis": rhythm_details,
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
        if job_id:
            update_job(job_id, stage="completed", progress=100)
        del data
        return result

    except Exception as exc:
        return error_result(exc)
    finally:
        cleanup_audio(path)


# =========================================================
# 9. MCP 工具一：测试音频下载
# =========================================================

@mcp.tool()
def test_audio_download(audio_url: str):
    """测试音频链接是否可访问，并检查文件大小和类型。"""
    path = None
    try:
        result = download_audio(audio_url)
        path = result["path"]
        return {
            "success": True,
            "message": "音频下载成功",
            "size_bytes": result["size_bytes"],
            "size_mb": round(result["size_bytes"] / 1024 / 1024, 2),
            "content_type": result["content_type"],
            "memory_mb": memory_mb(),
        }
    except Exception as exc:
        return error_result(exc)
    finally:
        cleanup_audio(path)


# =========================================================
# 10. MCP 工具二：全曲基础分析（旧同步工具，保留）
# =========================================================

@mcp.tool()
def analyze_audio(audio_url: str):
    """同步分析整首音频；长音频可能超过客户端等待时间，建议改用异步工具。"""
    if not acquire_analysis_lock():
        return {"success": False, "error": "已有分析任务正在运行，请稍后再试"}

    path = None
    try:
        started = time.perf_counter()
        downloaded = download_audio(audio_url)
        path = downloaded["path"]
        log_stage("download_complete", started)
        data = load_audio_features(path)

        result = {
            "success": True,
            "duration_seconds": round(data["duration"], 2),
            "original_duration_seconds": round(data["original_duration"], 2),
            "truncated": data["truncated"],
            "sample_rate": SAMPLE_RATE,
            "bpm": data["bpm"],
            "beat_count": data["beat_count"],
            "key": data["key"],
            "average_rms": round(safe_mean(data["rms"]), 5),
            "average_centroid_hz": round(safe_mean(data["centroid"]), 2),
            "average_bandwidth_hz": round(safe_mean(data["bandwidth"]), 2),
            "average_rolloff_hz": round(safe_mean(data["rolloff"]), 2),
            "average_zero_crossing_rate": round(safe_mean(data["zcr"]), 5),
            "beat_times_seconds": [
                round(float(t), 3) for t in data["beat_times"][:1500]
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
# 11. MCP 工具三：分段音乐特征分析（旧同步工具，保留）
# =========================================================

@mcp.tool()
def analyze_audio_segments(audio_url: str, window_seconds: int = 10):
    """按时间窗口分析音量、频谱、节拍密度等音乐特征。"""
    if not acquire_analysis_lock():
        return {"success": False, "error": "已有分析任务正在运行，请稍后再试"}

    path = None
    try:
        started = time.perf_counter()
        downloaded = download_audio(audio_url)
        path = downloaded["path"]
        data = load_audio_features(path)
        segments = build_segments(data, window_seconds)
        result = {
            "success": True,
            "duration_seconds": round(data["duration"], 2),
            "window_seconds": max(5, min(int(window_seconds), 30)),
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
# 12. MCP 工具四：音乐结构变化检测（旧同步工具，保留）
# =========================================================

@mcp.tool()
def detect_music_sections(
    audio_url: str,
    window_seconds: int = 5,
    min_gap_seconds: int = 12,
    max_candidates: int = 12,
):
    """检测音乐特征变化位置，不直接断定主歌、副歌或桥段。"""
    if not acquire_analysis_lock():
        return {"success": False, "error": "已有分析任务正在运行，请稍后再试"}

    path = None
    try:
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
            "duration_seconds": round(data["duration"], 2),
            "candidate_count": len(candidates),
            "section_candidates": candidates,
            "warning": (
                "变化点不等于正式音乐结构标签。"
                "主歌、副歌、前奏等需要结合听感或其他信息判断。"
            ),
            "memory_mb": memory_mb(),
        }
        log_stage("section_detection_complete", time.perf_counter())
        del data
        return result
    except Exception as exc:
        return error_result(exc)
    finally:
        cleanup_audio(path)
        ANALYSIS_LOCK.release()


# =========================================================
# 13-14. 统一使用后台分析，避免长时间同步请求超时
# =========================================================

def analysis_worker(job_id, audio_url, window_seconds):
    """后台线程执行完整分析；MCP 请求只负责快速返回 job_id。"""
    try:
        update_job(
            job_id,
            status="running",
            stage="starting",
            progress=1,
        )
        result = run_music_timeline_analysis(
            audio_url=audio_url,
            window_seconds=window_seconds,
            job_id=job_id,
        )

        if result.get("success"):
            update_job(
                job_id,
                status="done",
                stage="completed",
                progress=100,
                result=result,
                error=None,
            )
        else:
            update_job(
                job_id,
                status="failed",
                stage="failed",
                error=result.get("error", "分析失败"),
                result=result,
            )
    except Exception as exc:
        logger.exception("[music-analysis] background_job_failed")
        try:
            update_job(
                job_id,
                status="failed",
                stage="failed",
                error=str(exc),
                result={
                    "success": False,
                    "error": str(exc),
                    "memory_mb": memory_mb(),
                },
            )
        except Exception:
            logger.exception("[music-analysis] failed_to_persist_job_error")
    finally:
        ANALYSIS_LOCK.release()


def _start_background_analysis(audio_url: str, window_seconds: int = 10):
    """创建并启动一个完整音频分析任务；只返回轻量任务凭证。"""
    cleanup_old_jobs()

    try:
        normalized_window = max(5, min(int(window_seconds), 30))
    except (TypeError, ValueError):
        return {
            "success": False,
            "status": "invalid_arguments",
            "error": "window_seconds 必须是 5 到 30 之间的整数。",
        }

    if not acquire_analysis_lock():
        return {
            "success": False,
            "status": "busy",
            "error": "已有音乐分析任务正在运行，请先查询该任务或稍后再试。",
        }

    job_id = "music_" + uuid.uuid4().hex[:12]
    now = time.time()
    job = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "created_at": now,
        "updated_at": now,
        "window_seconds": normalized_window,
        "result": None,
        "error": None,
    }

    with JOBS_LOCK:
        JOBS[job_id] = job

    try:
        # 先持久化任务，再启动线程，避免任务刚返回就无法查询。
        persist_job(dict(job))
        thread = threading.Thread(
            target=analysis_worker,
            args=(job_id, audio_url, normalized_window),
            daemon=True,
            name=f"music-analysis-{job_id}",
        )
        thread.start()
    except Exception as exc:
        with JOBS_LOCK:
            failed_job = JOBS.get(job_id)
            if failed_job is not None:
                failed_job.update({
                    "status": "failed",
                    "stage": "failed",
                    "error": f"后台任务启动失败：{exc}",
                    "updated_at": time.time(),
                })
                failed_snapshot = dict(failed_job)
            else:
                failed_snapshot = None
        if failed_snapshot is not None:
            try:
                persist_job(failed_snapshot)
            except Exception:
                logger.exception("[music-analysis] failed_to_persist_startup_error")
        ANALYSIS_LOCK.release()
        logger.exception("[music-analysis] background_thread_start_failed")
        return {
            "success": False,
            "job_id": job_id,
            "status": "failed",
            "error": f"后台任务启动失败：{exc}",
        }

    return {
        "success": True,
        "message": "完整音乐分析任务已在后台启动",
        "job_id": job_id,
        "status": "queued",
        "window_seconds": normalized_window,
        "execution_mode": "background",
        "next_step": (
            "稍后调用 get_analysis_status(job_id) 查询进度；"
            "完成后调用 get_analysis_summary 或 get_analysis_timeline 分页读取结果。"
            "完整原始数据请使用 export_analysis_data 下载。"
        ),
    }


@mcp.tool()
def generate_music_timeline(audio_url: str, window_seconds: int = 10):
    """
    兼容旧工具名称，但改为后台执行，立即返回 job_id。
    避免 2 分钟以上的同步请求被 4Router 或中间网关超时中断。
    """
    return _start_background_analysis(audio_url, window_seconds)


@mcp.tool()
def start_audio_analysis(audio_url: str, window_seconds: int = 10):
    """
    启动后台完整音乐分析，立即返回 job_id。
    后续使用 get_analysis_status 查询进度。
    """
    return _start_background_analysis(audio_url, window_seconds)


# =========================================================
# 15. 新 MCP 工具七：查询异步分析状态
# =========================================================

@mcp.tool()
def get_analysis_status(job_id: str):
    """查询后台任务状态。此工具会快速返回，不会等待分析完成。"""
    cleanup_old_jobs()

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return {
                "success": False,
                "error": "找不到该任务。可能任务 ID 不正确，或 Render 服务重启后内存任务记录已清空。",
                "job_id": job_id,
            }

        return {
            "success": True,
            "job_id": job["job_id"],
            "status": job["status"],
            "stage": job["stage"],
            "progress": job["progress"],
            "created_at_unix": int(job["created_at"]),
            "updated_at_unix": int(job["updated_at"]),
            "error": job.get("error"),
            "memory_mb": memory_mb(),
        }


# =========================================================
# 16. 新 MCP 工具八：获取异步分析结果
# =========================================================

def _compact_result_view(result):
    """构造适合 MCP/LLM 客户端传输的精简结果，不修改数据库中的原始结果。"""
    result = result or {}
    rhythm = result.get("rhythm_analysis") or {}
    beats = rhythm.get("beat_points") or []
    accents = rhythm.get("accent_candidates") or []
    low_energy = rhythm.get("low_energy_intervals") or []
    sections = result.get("section_candidates") or []
    segments = result.get("segments") or []
    timeline = result.get("timeline") or []

    strongest_accents = sorted(
        accents,
        key=lambda item: float(item.get("strength", 0) or 0),
        reverse=True,
    )[:10]
    strongest_accents = sorted(
        [
            {"time": item.get("time"), "strength": item.get("strength")}
            for item in strongest_accents
        ],
        key=lambda item: float(item.get("time", 0) or 0),
    )

    last_accent = rhythm.get("last_accent_candidate")
    if isinstance(last_accent, dict):
        last_accent = {
            "time": last_accent.get("time"),
            "strength": last_accent.get("strength"),
        }

    return {
        "success": bool(result.get("success", True)),
        "song": result.get("song") or {},
        "music_overview": result.get("music_overview") or {},
        "data_counts": {
            "segment_count": len(segments),
            "timeline_event_count": len(timeline) or (len(segments) + len(sections)),
            "section_candidate_count": len(sections),
            "beat_count": len(beats),
            "accent_candidate_count": len(accents),
            "low_energy_interval_count": len(low_energy),
        },
        "section_candidates": [
            {
                "time": item.get("time"),
                "change_score": item.get("change_score"),
            }
            for item in sections[:20]
        ],
        "rhythm_overview": {
            "relative_strong_beat_threshold": rhythm.get("relative_strong_beat_threshold"),
            "strongest_accent_candidates": strongest_accents,
            "low_energy_intervals": [
                {
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "duration": item.get("duration"),
                    "description": item.get("description"),
                }
                for item in low_energy[:20]
            ],
            "last_accent_candidate": last_accent,
            "notes": (rhythm.get("notes") or [])[:4],
        },
        "limitations": (result.get("limitations") or [])[:4],
    }


@mcp.tool()
def get_analysis_result(job_id: str, include_full_result: bool = False):
    """
    获取分析结果。默认返回精简结果，避免数万字符响应被客户端截断。
    include_full_result=true 仅在完整 JSON 不超过 MAX_INLINE_RESULT_CHARS 时内联返回；
    更大的原始结果请使用 export_analysis_data 下载。
    """
    cleanup_old_jobs()

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return {
                "success": False,
                "error": "数据库和当前服务缓存中都找不到该任务。",
                "job_id": job_id,
            }

        status = job.get("status")
        if status in ("queued", "running"):
            return {
                "success": True,
                "job_id": job_id,
                "status": status,
                "stage": job.get("stage"),
                "progress": job.get("progress"),
                "message": "分析尚未完成，请稍后再次查询。",
            }

        if status != "done":
            return {
                "success": False,
                "job_id": job_id,
                "status": status,
                "error": job.get("error") or "分析结果不可用",
            }

        result = job.get("result")

    if not isinstance(result, dict):
        return {
            "success": False,
            "job_id": job_id,
            "status": "done",
            "error": "任务状态为 done，但数据库中没有可读取的结果对象。",
        }

    serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    result_chars = len(serialized)

    if include_full_result and result_chars <= MAX_INLINE_RESULT_CHARS:
        return {
            "success": True,
            "job_id": job_id,
            "status": "done",
            "result_mode": "full",
            "result_chars": result_chars,
            "result": result,
        }

    compact = _compact_result_view(result)
    return {
        "success": True,
        "job_id": job_id,
        "status": "done",
        "result_mode": "compact",
        "result_chars": result_chars,
        "result": compact,
        "full_result_available": True,
        "full_result_delivery": (
            "完整原始结果保留在 PostgreSQL。请调用 export_analysis_data(job_id, format='json') "
            "获取下载链接；节拍使用 get_beats 分页读取；时间轴使用 get_analysis_timeline 分页读取。"
        ),
        "full_result_inline_limit_chars": MAX_INLINE_RESULT_CHARS,
    }


# =========================================================
# 17. 精简查询、分页读取、报告与文件导出
# =========================================================

def get_job_snapshot(job_id):
    cleanup_old_jobs()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def low_energy_flag(timestamp, intervals):
    return any(float(item.get("start", 0)) <= timestamp <= float(item.get("end", 0))
               for item in (intervals or []))


@mcp.tool()
def get_analysis_summary(job_id: str):
    """返回精简音乐摘要；完整节拍和原始分析结果仍保留在服务端。"""
    job = get_job_snapshot(job_id)
    if not job:
        return {"success": False, "error": "找不到任务", "job_id": job_id}

    if job.get("status") != "done" or not job.get("result"):
        return {
            "success": True,
            "job_id": job_id,
            "status": job.get("status"),
            "stage": job.get("stage"),
            "progress": job.get("progress"),
            "error": job.get("error"),
        }

    result = job["result"]
    rhythm = result.get("rhythm_analysis") or {}

    # 音乐结构候选点最多保留20个，只保留真实存在的字段。
    section_candidates = []
    for item in (result.get("section_candidates") or [])[:20]:
        section_candidates.append({
            "time": item.get("time"),
            "change_score": item.get("change_score"),
        })

    # 先按强度选出最强10个，再按时间排序，便于阅读时间轴。
    all_accents = rhythm.get("accent_candidates") or []
    strongest_accents = sorted(
        all_accents,
        key=lambda item: float(item.get("strength", 0) or 0),
        reverse=True,
    )[:10]
    accent_candidates = [
        {"time": item.get("time"), "strength": item.get("strength")}
        for item in sorted(
            strongest_accents,
            key=lambda item: float(item.get("time", 0) or 0),
        )
    ]

    # 最长的8个低能量区间优先保留，最后按开始时间排序。
    all_low_energy = rhythm.get("low_energy_intervals") or []
    longest_intervals = sorted(
        all_low_energy,
        key=lambda item: float(item.get("duration", 0) or 0),
        reverse=True,
    )[:8]
    low_energy_intervals = [
        {
            "start": item.get("start"),
            "end": item.get("end"),
            "duration": item.get("duration"),
        }
        for item in sorted(
            longest_intervals,
            key=lambda item: float(item.get("start", 0) or 0),
        )
    ]

    # 均匀抽取最多12个节拍样本，并保留原始节拍序号。
    beats = rhythm.get("beat_points") or []
    max_samples = 12
    if len(beats) <= max_samples:
        sample_indices = list(range(len(beats)))
    elif max_samples == 1:
        sample_indices = [0]
    else:
        sample_indices = sorted({
            round(i * (len(beats) - 1) / (max_samples - 1))
            for i in range(max_samples)
        })

    sampled = []
    for index in sample_indices:
        beat = beats[index]
        sampled.append({
            "beat_index": index + 1,
            "time": beat.get("time"),
            "relative_strength": beat.get("relative_strength"),
            "accent_class": beat.get("accent_class"),
        })

    last_accent = rhythm.get("last_accent_candidate")
    compact_last_accent = None
    if isinstance(last_accent, dict):
        compact_last_accent = {
            "time": last_accent.get("time"),
            "strength": last_accent.get("strength"),
        }

    return {
        "success": True,
        "job_id": job_id,
        "status": "done",
        "song": result.get("song", {}),
        "music_overview": result.get("music_overview", {}),
        "beat_count": len(beats),
        "section_candidates_total": len(result.get("section_candidates") or []),
        "section_candidates": section_candidates,
        "beat_sampling": {
            "assumption": "均匀抽样最多12个检测节拍；未检测拍号",
            "sample_count": len(sampled),
            "items": sampled,
        },
        "accent_candidates_total": len(all_accents),
        "accent_candidates": accent_candidates,
        "low_energy_intervals_total": len(all_low_energy),
        "low_energy_intervals": low_energy_intervals,
        "last_accent_candidate": compact_last_accent,
        "limitations": (result.get("limitations") or [])[:3],
    }


@mcp.tool()
def get_analysis_timeline(job_id: str, offset: int = 0, limit: int = 12):
    """
    分页读取完整音乐时间轴事件。
    默认每页12项，最多12项；所有分段和变化候选点均可通过 offset 完整读取。
    全部节拍请使用 get_beats 分页读取。
    """
    job = get_job_snapshot(job_id)
    if not job:
        return {"success": False, "error": "找不到任务", "job_id": job_id}

    if job.get("status") != "done" or not job.get("result"):
        return {
            "success": False,
            "job_id": job_id,
            "status": job.get("status"),
            "stage": job.get("stage"),
            "progress": job.get("progress"),
            "error": job.get("error"),
            "message": "分析尚未完成或结果不可用，请先使用 get_analysis_status 查询任务状态。",
        }

    result = job["result"]
    segments = result.get("segments") or []
    section_candidates = result.get("section_candidates") or []
    rhythm = result.get("rhythm_analysis") or {}

    timeline = result.get("timeline") or []
    if not timeline:
        timeline = []
        for segment in segments:
            timeline.append({
                "time": segment.get("start", 0),
                "type": "audio_segment",
                "data": segment,
            })
        for candidate in section_candidates:
            timeline.append({
                "time": candidate.get("time", 0),
                "type": "possible_section_change",
                "data": candidate,
            })
        timeline.sort(key=lambda item: float(item.get("time", 0) or 0))

    try:
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 12))
    except (TypeError, ValueError):
        return {
            "success": False,
            "job_id": job_id,
            "error": "offset 必须是非负整数，limit 必须是 1 到 12 之间的整数。",
        }

    page = timeline[offset:offset + limit]
    next_offset = offset + len(page)
    all_accents = rhythm.get("accent_candidates") or []
    strongest_accents = sorted(
        all_accents,
        key=lambda item: float(item.get("strength", 0) or 0),
        reverse=True,
    )[:10]
    compact_accents = sorted(
        [
            {"time": item.get("time"), "strength": item.get("strength")}
            for item in strongest_accents
        ],
        key=lambda item: float(item.get("time", 0) or 0),
    )

    all_low_energy = rhythm.get("low_energy_intervals") or []
    compact_low_energy = [
        {
            "start": item.get("start"),
            "end": item.get("end"),
            "duration": item.get("duration"),
            "description": item.get("description"),
        }
        for item in all_low_energy[:20]
    ]

    last_accent = rhythm.get("last_accent_candidate")
    if isinstance(last_accent, dict):
        last_accent = {
            "time": last_accent.get("time"),
            "strength": last_accent.get("strength"),
        }

    return {
        "success": True,
        "job_id": job_id,
        "status": "done",
        "song": result.get("song") or {},
        "music_overview": result.get("music_overview") or {},
        "segment_count": len(segments),
        "section_candidate_count": len(section_candidates),
        "timeline_event_count": len(timeline),
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "next_offset": next_offset if next_offset < len(timeline) else None,
        "timeline": page,
        "rhythm_overview": {
            "beat_count": len(rhythm.get("beat_points") or []),
            "relative_strong_beat_threshold": rhythm.get("relative_strong_beat_threshold"),
            "accent_candidates_total": len(all_accents),
            "strongest_accent_candidates": compact_accents,
            "low_energy_intervals_total": len(all_low_energy),
            "low_energy_intervals": compact_low_energy,
            "last_accent_candidate": last_accent,
            "notes": (rhythm.get("notes") or [])[:4],
        },
        "limitations": (result.get("limitations") or [])[:4],
        "omitted_data": {
            "beat_points": "本页不返回全部节拍点；请使用 get_beats(job_id, offset, limit) 分页读取。",
            "full_json": "完整原始 JSON 请使用 export_analysis_data(job_id, format='json') 下载。",
        },
    }


@mcp.tool()
def get_beats(job_id: str, offset: int = 0, limit: int = 50):
    """分页读取节拍点。limit最大100；offset从0开始。"""
    job = get_job_snapshot(job_id)
    if not job:
        return {"success": False, "error": "找不到任务", "job_id": job_id}
    if job["status"] != "done" or not job.get("result"):
        return {"success": False, "job_id": job_id, "status": job["status"],
                "message": "分析尚未完成或结果不可用", "error": job.get("error")}
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 100))
    rhythm = job["result"].get("rhythm_analysis", {})
    beats = rhythm.get("beat_points", [])
    page = beats[offset:offset + limit]
    intervals = rhythm.get("low_energy_intervals", [])
    items = [{**item, "in_low_energy_interval": low_energy_flag(float(item.get("time", 0)), intervals)}
             for item in page]
    return {"success": True, "job_id": job_id, "offset": offset, "limit": limit,
            "returned": len(items), "total": len(beats),
            "next_offset": offset + len(items) if offset + len(items) < len(beats) else None,
            "items": items}


def register_export(job_id, filename, content_bytes):
    safe_name = Path(filename).name
    token = secrets.token_urlsafe(32)
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO exports(token,filename,content,created_at) VALUES(%s,%s,%s,%s)",
            (token, safe_name, content_bytes, time.time()),
        )
        conn.commit()
    base_url = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    if not base_url:
        return {"success": False, "error": "文件已保存到 PostgreSQL，但尚未配置 PUBLIC_BASE_URL，无法构造外部下载链接。",
                "filename": safe_name, "storage": "postgresql",
                "required_env": "PUBLIC_BASE_URL，例如 https://你的Render服务.onrender.com"}
    return {"success": True, "filename": safe_name,
            "download_url": f"{base_url}/download?token={token}",
            "storage": "postgresql",
            "expires_in_seconds": DOWNLOAD_TTL_SECONDS,
            "note": "文件内容保存在 PostgreSQL；下载令牌在有效期内可用，请勿公开分享。"}


@mcp.tool()
def export_analysis_data(job_id: str, format: str = "json"):
    """在MCP服务端生成JSON或CSV文件，返回下载链接；不把完整数据交给模型。"""
    job = get_job_snapshot(job_id)
    if not job:
        return {"success": False, "error": "找不到任务", "job_id": job_id}
    if job["status"] != "done" or not job.get("result"):
        return {"success": False, "job_id": job_id, "status": job["status"],
                "error": "分析尚未完成或结果不可用"}
    fmt = (format or "json").lower().strip()
    result = job["result"]
    if fmt == "json":
        payload = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
        filename = f"{job_id}_analysis.json"
    elif fmt == "csv":
        rhythm = result.get("rhythm_analysis", {})
        intervals = rhythm.get("low_energy_intervals", [])
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=["time", "relative_strength", "accent_class", "in_low_energy_interval"])
        writer.writeheader()
        for item in rhythm.get("beat_points", []):
            t = float(item.get("time", 0))
            writer.writerow({"time": item.get("time"),
                             "relative_strength": item.get("relative_strength"),
                             "accent_class": item.get("accent_class"),
                             "in_low_energy_interval": low_energy_flag(t, intervals)})
        payload = output.getvalue().encode("utf-8-sig")
        filename = f"{job_id}_beats.csv"
    else:
        return {"success": False, "error": "format仅支持json或csv"}
    return register_export(job_id, filename, payload)


@mcp.tool()
def generate_song_report(job_id: str):
    """基于现有分析结果在服务端生成song.md，不重新分析音频。"""
    job = get_job_snapshot(job_id)
    if not job:
        return {"success": False, "error": "找不到任务", "job_id": job_id}
    if job["status"] != "done" or not job.get("result"):
        return {"success": False, "job_id": job_id, "status": job["status"],
                "error": "分析尚未完成或结果不可用"}
    result = job["result"]
    song = result.get("song", {})
    overview = result.get("music_overview", {})
    rhythm = result.get("rhythm_analysis", {})
    lines = [f"# 音乐分析报告：{job_id}", "",
             "## 基础信息", f"- 时长：{song.get('duration_seconds', '未知')} 秒",
             f"- BPM：{overview.get('bpm', '未知')}",
             f"- 节拍总数：{rhythm.get('beat_point_count', overview.get('beat_count', '未知'))}",
             f"- 调性估计：{overview.get('key', {})}", "",
             "## 音乐变化候选点"]
    for item in result.get("section_candidates", []):
        lines.append(f"- {item.get('time')} 秒：变化分数 {item.get('change_score')}（需听感确认）")
    lines += ["", "## 重音候选"]
    for item in rhythm.get("accent_candidates", [])[:30]:
        lines.append(f"- {item.get('time')} 秒：强度 {item.get('strength')}（需听感确认）")
    lines += ["", "## 低能量区间"]
    for item in rhythm.get("low_energy_intervals", []):
        lines.append(f"- {item.get('start')}–{item.get('end')} 秒：{item.get('description', '需听感确认')}")
    lines += ["", "## 导演分析待补充", "- 前奏、主歌、副歌、桥段和尾奏：需结合实际听感与转写进一步标注。",
              "- 情绪与叙事：当前算法没有直接识别情绪，不应仅凭音频特征臆测。",
              "- 镜头与转场建议：待结合人工确认的段落和歌词内容补充。", "",
              "## 分析限制"]
    for item in result.get("limitations", []):
        lines.append(f"- {item}")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return register_export(job_id, f"{job_id}_song.md", payload)


@mcp.custom_route("/download", methods=["GET"], include_in_schema=False)
async def download_export(request: Request):
    """随机令牌保护的短期下载路由。自定义路由不继承MCP认证，令牌本身即下载凭证。"""
    token = request.query_params.get("token", "")
    if not token or len(token) > 128:
        return JSONResponse({"success": False, "error": "缺少或无效的下载令牌"}, status_code=400)
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM exports WHERE token=%s", (token,)).fetchone()
    if not row or time.time() - row["created_at"] > DOWNLOAD_TTL_SECONDS:
        return JSONResponse({"success": False, "error": "下载链接不存在或已过期"}, status_code=404)
    filename = row["filename"]
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"}
    return Response(
        content=bytes(row["content"]),
        media_type="application/octet-stream",
        headers=headers,
    )


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health_check(request: Request):
    try:
        with db_connect() as conn:
            conn.execute("SELECT 1").fetchone()
        return JSONResponse({
            "status": "healthy",
            "service": "music-analysis",
            "database_backend": "postgresql",
            "database_connected": True,
            "storage_backend": "postgresql",
            "job_retention_seconds": JOB_TTL_SECONDS,
            "export_link_ttl_seconds": DOWNLOAD_TTL_SECONDS,
        })
    except Exception as exc:
        logger.exception("[music-analysis] health_check_database_failed")
        return JSONResponse({
            "status": "unhealthy",
            "service": "music-analysis",
            "database_backend": "postgresql",
            "database_connected": False,
            "error": "数据库连接失败，请检查 Render 的 DATABASE_URL 和 Neon 状态。",
        }, status_code=503)


# =========================================================
# 17. 启动 MCP 服务
# =========================================================

if __name__ == "__main__":
    load_persisted_jobs()
    logger.info("Starting music-analysis MCP on %s:%s using Neon PostgreSQL", HOST, PORT)
    mcp.run(
        transport="http",
        host=HOST,
        port=PORT,
        stateless_http=True,
    )
