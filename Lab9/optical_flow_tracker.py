"""Свой трекер частиц на оптическом потоке Лукаса–Канаде."""

from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
SCALE = 0.5
MAX_JUMP = 18.0
MIN_TRACK = 15
REFRESH_EVERY = 10
PREVIEW_FRAMES = {0, 900, 2000, 3500}
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)


def cell_body(blue: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(blue, (0, 0), 3)
    bright = blur[blur > 15]
    threshold = max(30.0, float(np.percentile(bright, 35))) if bright.size else 30.0
    raw = (blur > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    if count <= 1:
        return opened
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == index).astype(np.uint8)


def detect_particles(frame_bgr: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(frame_bgr)
    red_f = red.astype(np.float32)
    blue_f = blue.astype(np.float32)
    green_f = green.astype(np.float32)
    inside = cell_body(blue_f).astype(bool)
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


class FlowTracker:
    def __init__(self) -> None:
        self.prev_red = None
        self.points = None
        self.ids: list[int] = []
        self.tracks: dict[int, list[tuple[int, float, float]]] = {}
        self.next_id = 1
        self.max_alive = 0

    def _add(self, frame_idx: int, x: float, y: float) -> int:
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = [(frame_idx, x, y)]
        return track_id

    def step(self, frame_idx: int, red: np.ndarray, detections: np.ndarray) -> None:
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 2)
        if self.prev_red is None or self.points is None or len(self.ids) == 0:
            kept = []
            self.ids = []
            for x, y in detections:
                self.ids.append(self._add(frame_idx, float(x), float(y)))
                kept.append([float(x), float(y)])
            self.points = np.array(kept, dtype=np.float32).reshape(-1, 1, 2) if kept else None
            self.prev_red = red
            self.max_alive = max(self.max_alive, len(self.ids))
            return

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_red,
            red,
            self.points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=LK_CRITERIA,
        )
        kept_pts = []
        kept_ids = []
        status = status.reshape(-1) if status is not None else np.zeros(len(self.ids), dtype=np.uint8)
        height, width = red.shape
        for index, track_id in enumerate(self.ids):
            if nxt is None or status[index] == 0:
                continue
            x, y = float(nxt[index, 0, 0]), float(nxt[index, 0, 1])
            ox, oy = float(self.points[index, 0, 0]), float(self.points[index, 0, 1])
            if not (0 <= x < width and 0 <= y < height):
                continue
            if np.hypot(x - ox, y - oy) > MAX_JUMP:
                continue
            self.tracks[track_id].append((frame_idx, x, y))
            kept_pts.append([x, y])
            kept_ids.append(track_id)

        if frame_idx % REFRESH_EVERY == 0 and len(detections):
            existing = np.array(kept_pts, dtype=np.float32).reshape(-1, 2) if kept_pts else np.zeros((0, 2), np.float32)
            for x, y in detections:
                if len(existing) and np.linalg.norm(existing - np.array([x, y]), axis=1).min() <= 10.0:
                    continue
                track_id = self._add(frame_idx, float(x), float(y))
                kept_pts.append([float(x), float(y)])
                kept_ids.append(track_id)
                existing = np.array(kept_pts, dtype=np.float32)

        self.ids = kept_ids
        self.points = np.array(kept_pts, dtype=np.float32).reshape(-1, 1, 2) if kept_pts else None
        self.prev_red = red
        self.max_alive = max(self.max_alive, len(self.ids))


def color_for(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id + 11)
    return tuple(int(value) for value in rng.integers(60, 255, size=3))


def draw(frame: np.ndarray, tracker: FlowTracker, frame_idx: int) -> np.ndarray:
    canvas = frame.copy()
    for track_id, points in tracker.tracks.items():
        if len(points) < 2 or frame_idx - points[-1][0] > 20:
            continue
        color = color_for(track_id)
        tail = [point for point in points if frame_idx - point[0] <= 45]
        for start, end in zip(tail, tail[1:]):
            cv2.line(canvas, (int(start[1]), int(start[2])), (int(end[1]), int(end[2])), color, 1)
        x, y = points[-1][1], points[-1][2]
        cv2.circle(canvas, (int(x), int(y)), 3, color, 1)
    cv2.putText(
        canvas,
        f"f={frame_idx} flow={len(tracker.ids)}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return canvas


def track_speed(points: list[tuple[int, float, float]], fps: float) -> float:
    if len(points) < 2:
        return 0.0
    shifts = []
    for start, end in zip(points, points[1:]):
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
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        str(OUT / "flow_tracked.mp4"),
        fps=fps,
        codec="libx264",
        quality=6,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    tracker = FlowTracker()
    previews = []
    started = time.perf_counter()
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        red = cv2.GaussianBlur(small[:, :, 2], (3, 3), 0)
        points = detect_particles(small)
        tracker.step(frame_idx, red, points)
        canvas = draw(small, tracker, frame_idx)
        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        if frame_idx in PREVIEW_FRAMES:
            previews.append(canvas.copy())
        if frame_idx % 500 == 0:
            print(f"кадр {frame_idx}, точек потока {len(tracker.ids)}", flush=True)
        frame_idx += 1
    capture.release()
    writer.close()
    elapsed = time.perf_counter() - started
    confirmed = [points for points in tracker.tracks.values() if len(points) >= MIN_TRACK]
    speeds = [track_speed(points, fps) for points in confirmed]
    if previews:
        cv2.imwrite(str(OUT / "preview.jpg"), np.hstack(previews), [cv2.IMWRITE_JPEG_QUALITY, 85])
    lab8_stats = Path(__file__).resolve().parents[1] / "Lab8" / "output" / "stats.txt"
    comparison = ""
    if lab8_stats.exists():
        comparison = "Lab8:\n" + lab8_stats.read_text(encoding="utf-8")
    summary = (
        f"кадров: {frame_idx}\n"
        f"время обработки: {elapsed:.1f} с\n"
        f"скорость: {frame_idx / elapsed:.2f} кадр/с\n"
        f"подтверждённых треков (>={MIN_TRACK} точек): {len(confirmed)}\n"
        f"максимум одновременных точек: {tracker.max_alive}\n"
        f"медианная скорость трека: {np.median(speeds) if speeds else 0:.2f} px/с\n"
    )
    (OUT / "stats.txt").write_text(summary + ("\n" + comparison if comparison else ""), encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
