"""Площадь, периметр, угол и оси объектов на sample30.jpg."""

from pathlib import Path

import cv2
import numpy as np
from skimage.measure import label, regionprops

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"


def axis_lengths(prop) -> tuple[float, float]:
    major = getattr(prop, "axis_major_length", None)
    minor = getattr(prop, "axis_minor_length", None)
    if major is None:
        major = prop.major_axis_length
    if minor is None:
        minor = prop.minor_axis_length
    return float(major), float(minor)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(str(DATA / "sample30.jpg"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit("нет data/sample30.jpg")
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labels = label(binary > 0, connectivity=2)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rows = []
    for prop in regionprops(labels):
        if prop.area < 2:
            continue
        major, minor = axis_lengths(prop)
        cy, cx = prop.centroid
        angle = float(np.degrees(prop.orientation))
        rows.append((prop.area, prop.perimeter, angle, major, minor, cx, cy))
        center = (int(round(cx)), int(round(cy)))
        axes = (max(int(round(major / 2)), 1), max(int(round(minor / 2)), 1))
        # orientation в skimage отсчитывается от вертикали против часовой.
        # У cv2.ellipse нулевой угол лежит на оси x и растёт по часовой.
        cv2.ellipse(canvas, center, axes, 90.0 - angle, 0, 360, (0, 220, 0), 1)
        tip_x = int(round(cx - np.sin(prop.orientation) * major / 2.0))
        tip_y = int(round(cy - np.cos(prop.orientation) * major / 2.0))
        cv2.line(canvas, center, (tip_x, tip_y), (0, 0, 255), 1)

    rows.sort(key=lambda item: item[5])
    header = "area,perimeter,angle_deg,axis_major,axis_minor,center_x,center_y"
    body = "\n".join(
        f"{area:.0f},{perim:.3f},{angle:.3f},{major:.3f},{minor:.3f},{cx:.3f},{cy:.3f}"
        for area, perim, angle, major, minor, cx, cy in rows
    )
    (OUT / "features.csv").write_text(header + "\n" + body + "\n", encoding="utf-8")
    cv2.imwrite(str(OUT / "objects.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    reference = np.genfromtxt(DATA / "result.txt", delimiter="\t", names=True)
    detected = np.array([(cx, cy, area) for area, _, _, _, _, cx, cy in rows], dtype=np.float64)
    ref_xy = np.column_stack([reference["CenterX"], reference["CenterY"]])
    if len(detected) and len(ref_xy):
        distances = np.linalg.norm(detected[:, None, :2] - ref_xy[None, :, :], axis=2)
        nearest = distances.argmin(axis=1)
        area_ratio = detected[:, 2] / reference["Area"][nearest]
        median_shift = float(np.median(distances.min(axis=1)))
        median_ratio = float(np.median(area_ratio))
    else:
        median_shift = float("nan")
        median_ratio = float("nan")
    summary = (
        f"объектов: {len(rows)} (в result.txt: {len(reference)})\n"
        f"медианное смещение центра к ближайшему эталону: {median_shift:.2f} px\n"
        f"медианное отношение площадей: {median_ratio:.3f}\n"
        "угол — orientation skimage, градусы от вертикали против часовой стрелки\n"
    )
    (OUT / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"таблица: {OUT / 'features.csv'}")


if __name__ == "__main__":
    main()
