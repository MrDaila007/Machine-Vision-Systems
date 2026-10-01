"""Выделение теплохода прямыми линиями (Canny + Хаф)."""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"


def line_angle_deg(x1: int, y1: int, x2: int, y2: int) -> float:
    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    return float(min(angle, 180.0 - angle))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(str(DATA / "boat1.jpg"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit("нет data/boat1.jpg")
    height, width = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=90,
        minLineLength=70,
        maxLineGap=16,
    )

    all_lines = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ship = all_lines.copy()
    total = 0
    kept = 0
    if lines is not None:
        lines = np.asarray(lines).reshape(-1, 4)
        # Мост слева и дом справа лежат в той же полосе по высоте, что и
        # корпус, и визуально не отделяются от него ни плотностью рёбер
        # Канни (вода под всеми тремя даёт сплошной тёмный фон), ни яркостью
        # (дом и корпус одинаково тёмные). Поэтому полоса по x, как и полоса
        # по высоте выше, задаётся явно по пикселям теплохода на кадре
        # 1972x1291: нос слева x≈460, корма второго судна справа x≈1705;
        # мост заканчивается левее x≈400, дом начинается правее x≈1650.
        x_min = 0.225 * width
        x_max = 0.860 * width

        for x1, y1, x2, y2 in lines:
            total += 1
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = line_angle_deg(int(x1), int(y1), int(x2), int(y2))
            mid_y = 0.5 * (y1 + y2)
            cv2.line(all_lines, (int(x1), int(y1)), (int(x2), int(y2)), (0, 180, 0), 2)
            in_band = 0.30 * height <= mid_y <= 0.58 * height
            in_hull_span = x_min <= min(x1, x2) and max(x1, x2) <= x_max
            horizontal = angle <= 16.0 and length >= 110.0
            vertical = angle >= 72.0 and length >= 35.0
            if in_band and in_hull_span and (horizontal or vertical):
                cv2.line(ship, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                kept += 1

    cv2.imwrite(str(OUT / "edges.jpg"), edges)
    cv2.imwrite(str(OUT / "all_lines.jpg"), all_lines, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(OUT / "ship_lines.jpg"), ship, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"кадр {width}x{height}, линий Хафа: {total}, линий корпуса: {kept}")
    print(f"результат: {OUT / 'ship_lines.jpg'}")


if __name__ == "__main__":
    main()
