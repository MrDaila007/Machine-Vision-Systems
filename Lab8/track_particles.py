"""Трекинг красных частиц в клетке: детекции и связывание ближайших центров."""

from functools import lru_cache
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
SCALE = 0.5
MAX_DIST = 14.0
MAX_MISS = 12
MIN_TRACK = 15
# Маска тела клетки — самый дорогой шаг (открытие 45x45), а за 5 кадров
# (1/6 с) клетка меняется меньше, чем маска шумит от кадра к кадру.
MASK_EVERY = 5
PREVIEW_FRAMES = {0, 900, 2000, 3500}


def cell_body(blue: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(blue, (0, 0), 3)
    bright = blur[blur > 15]
    threshold = max(30.0, float(np.percentile(bright, 35))) if bright.size else 30.0
    raw = (blur > threshold).astype(np.uint8)
    # 21x21 не рвёт тонкие выросты (филоподии): они остаются тонким "мостиком"
    # к основному телу клетки, largest-component подбирает их вместе с телом,
    # и детектор частиц затем ищет точки на кончиках этих отростков. 45x45
    # надёжно отделяет такие мостики на всех кадрах ролика (проверено: после
    # открытия остаётся 85-92% площади тела, без разрывов самого тела).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    if count <= 1:
        return opened
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == index).astype(np.uint8)


def detect_particles(frame_bgr: np.ndarray, inside: np.ndarray) -> np.ndarray:
    _blue, green, red = cv2.split(frame_bgr)
    red_f = red.astype(np.float32)
    green_f = green.astype(np.float32)
    if int(inside.sum()) < 500:
        return np.zeros((0, 2), dtype=np.float32)
    highpass = np.clip(red_f - cv2.GaussianBlur(red_f, (0, 0), 1.8), 0, None)
    floor = max(16.0, float(np.percentile(red_f[inside], 99.0)) * 0.62)
    mask = inside & (red_f >= floor) & (red_f > green_f + 8.0) & (highpass >= 7.0)
    score = np.where(mask, highpass, 0).astype(np.float32)
    dilated = cv2.dilate(score, np.ones((11, 11), np.float32))
    ys, xs = np.where((score > 0) & (score >= dilated - 1e-6))
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([xs, ys], axis=1).astype(np.float32)


class Track:
    def __init__(self, track_id: int, frame_idx: int, point: np.ndarray) -> None:
        self.track_id = track_id
        self.points = [(frame_idx, float(point[0]), float(point[1]))]
        self.miss = 0
        self.alive = True


class CentroidTracker:
    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self.live: list[Track] = []
        self.fading: list[Track] = []
        self.next_id = 1
        self.max_alive = 0

    def update(self, frame_idx: int, detections: np.ndarray) -> None:
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 2)
        active = self.live
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        if active and len(detections):
            previous = np.array([[track.points[-1][1], track.points[-1][2]] for track in active], dtype=np.float32)
            cost = np.linalg.norm(previous[:, None, :] - detections[None, :, :], axis=2)
            cost[cost > MAX_DIST] = 1e5
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if cost[row, col] >= 1e5:
                    continue
                point = detections[col]
                active[row].points.append((frame_idx, float(point[0]), float(point[1])))
                active[row].miss = 0
                matched_tracks.add(row)
                matched_dets.add(int(col))
        for index, track in enumerate(active):
            if index not in matched_tracks:
                track.miss += 1
                if track.miss > MAX_MISS:
                    track.alive = False
        self.fading.extend(track for track in active if not track.alive)
        self.live = [track for track in active if track.alive]
        for index, point in enumerate(detections):
            if index not in matched_dets:
                track = Track(self.next_id, frame_idx, point)
                self.tracks.append(track)
                self.live.append(track)
                self.next_id += 1
        self.max_alive = max(self.max_alive, len(self.live))

    def recent(self, frame_idx: int, window: int = 20) -> list[Track]:
        self.fading = [track for track in self.fading if frame_idx - track.points[-1][0] <= window]
        visible = [
            track
            for track in self.live + self.fading
            if len(track.points) >= 2 and frame_idx - track.points[-1][0] <= window
        ]
        return sorted(visible, key=lambda track: track.track_id)


def tail_points(points: list[tuple[int, float, float]], frame_idx: int, window: int) -> list[tuple[int, float, float]]:
    tail = []
    for point in reversed(points):
        if frame_idx - point[0] > window:
            break
        tail.append(point)
    tail.reverse()
    return tail


@lru_cache(maxsize=None)
def color_for(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id + 3)
    return tuple(int(value) for value in rng.integers(60, 255, size=3))


def draw(frame: np.ndarray, tracker: CentroidTracker, frame_idx: int) -> np.ndarray:
    canvas = frame.copy()
    for track in tracker.recent(frame_idx):
        color = color_for(track.track_id)
        tail = tail_points(track.points, frame_idx, 45)
        for start, end in zip(tail, tail[1:]):
            cv2.line(canvas, (int(start[1]), int(start[2])), (int(end[1]), int(end[2])), color, 1)
        x, y = track.points[-1][1], track.points[-1][2]
        cv2.circle(canvas, (int(x), int(y)), 3, color, 1)
    alive = len(tracker.live)
    cv2.putText(canvas, f"f={frame_idx} tracks={alive}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def nvenc_available() -> bool:
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:d=0.1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        return subprocess.run(command, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def open_writer(path: Path, fps: float):
    import imageio.v2 as imageio

    if nvenc_available():
        return imageio.get_writer(
            str(path),
            fps=fps,
            codec="h264_nvenc",
            quality=None,
            pixelformat="yuv420p",
            macro_block_size=1,
            output_params=["-preset", "p4", "-cq", "23"],
        )
    # quality=6 у imageio-ffmpeg уже подставляет -pix_fmt yuv420p для libx264;
    # повторное явное указание того же флага только дублирует его в команде
    # ffmpeg и даёт безобидное, но лишнее предупреждение "Multiple -pix_fmt".
    return imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=6,
        macro_block_size=1,
    )


def track_speed(track: Track, fps: float) -> float:
    if len(track.points) < 2:
        return 0.0
    shifts = []
    for start, end in zip(track.points, track.points[1:]):
        dt = end[0] - start[0]
        if dt <= 0:
            continue
        shifts.append(np.hypot(end[1] - start[1], end[2] - start[2]) / dt * fps)
    return float(np.mean(shifts)) if shifts else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(DATA / "MKCell.mp4"))
    if not capture.isOpened():
        raise SystemExit("не открывается data/MKCell.mp4")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = open_writer(OUT / "tracked.mp4", fps)
    tracker = CentroidTracker()
    previews = []
    detections_per_frame = []
    started = time.perf_counter()
    frame_idx = 0
    inside = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        if frame_idx % MASK_EVERY == 0:
            inside = cell_body(small[:, :, 0].astype(np.float32)).astype(bool)
        points = detect_particles(small, inside)
        tracker.update(frame_idx, points)
        canvas = draw(small, tracker, frame_idx)
        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        detections_per_frame.append(len(points))
        if frame_idx in PREVIEW_FRAMES:
            previews.append(canvas.copy())
        if frame_idx % 500 == 0:
            print(f"кадр {frame_idx}, детекций {len(points)}", flush=True)
        frame_idx += 1
    capture.release()
    writer.close()
    elapsed = time.perf_counter() - started
    confirmed = [track for track in tracker.tracks if len(track.points) >= MIN_TRACK]
    speeds = [track_speed(track, fps) for track in confirmed]
    if previews:
        cv2.imwrite(str(OUT / "preview.jpg"), np.hstack(previews), [cv2.IMWRITE_JPEG_QUALITY, 85])
    summary = (
        f"кадров: {frame_idx}\n"
        f"время обработки: {elapsed:.1f} с\n"
        f"скорость: {frame_idx / elapsed:.2f} кадр/с\n"
        f"среднее число детекций на кадр: {np.mean(detections_per_frame):.1f}\n"
        f"подтверждённых треков (>={MIN_TRACK} точек): {len(confirmed)}\n"
        f"максимум одновременных треков: {tracker.max_alive}\n"
        f"медианная скорость трека: {np.median(speeds) if speeds else 0:.2f} px/с\n"
    )
    (OUT / "stats.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
