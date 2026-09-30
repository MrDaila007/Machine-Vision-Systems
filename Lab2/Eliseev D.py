"""Частотная и растровая фильтрация."""

from pathlib import Path

import cv2
import numpy as np
from fpdf import FPDF

SURNAME = "Eliseev D"

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def to_uint8(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (0.5, 99.5))
    if high - low < 1e-8:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def gaussian_lowpass(image: np.ndarray, sigma: float) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(image.astype(np.float64)))
    height, width = image.shape
    yy, xx = np.ogrid[:height, :width]
    radius2 = (yy - height // 2) ** 2 + (xx - width // 2) ** 2
    mask = np.exp(-radius2 / (2.0 * sigma ** 2))
    restored = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
    return restored


def spectrum_image(image: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(image.astype(np.float64)))
    return to_uint8(np.log1p(np.abs(spectrum)))


def notch_periodic(image: np.ndarray, notch_sigma: float = 3.2) -> tuple[np.ndarray, np.ndarray]:
    """Гасит полосу спектра вдоль самой яркой периодической помехи.

    Пик ищется в стороне от осей и от нулевой частоты. Помеха и её гармоники
    лежат на одной прямой через центр спектра, поэтому режекция идёт вдоль
    этой прямой.
    """
    spectrum = np.fft.fftshift(np.fft.fft2(image.astype(np.float64)))
    magnitude = np.log1p(np.abs(spectrum))
    height, width = image.shape
    yy, xx = np.mgrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    search = magnitude.copy()
    search[radius < 8] = 0
    search[np.abs(xx - center_x) <= 2] = 0
    search[np.abs(yy - center_y) <= 2] = 0
    peak_y, peak_x = np.unravel_index(int(np.argmax(search)), magnitude.shape)
    angle = np.arctan2(peak_y - center_y, peak_x - center_x)
    distance = np.abs((xx - center_x) * np.sin(angle) - (yy - center_y) * np.cos(angle))
    notch = 1.0 - np.exp(-(distance ** 2) / (2.0 * notch_sigma ** 2))
    notch[radius < 6] = 1.0
    restored = np.fft.ifft2(np.fft.ifftshift(spectrum * notch)).real
    return restored, notch


class Report(FPDF):
    def __init__(self) -> None:
        super().__init__()
        self.add_font("DejaVu", "", FONT)
        self.add_font("DejaVu", "B", FONT_BOLD)
        self.set_auto_page_break(auto=True, margin=15)

    def heading(self, text: str) -> None:
        self.set_font("DejaVu", "B", 16)
        self.multi_cell(0, 8, text)
        self.ln(2)

    def paragraph(self, text: str) -> None:
        self.set_font("DejaVu", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def figure(self, path: Path, caption: str) -> None:
        if self.get_y() > 180:
            self.add_page()
        self.image(str(path), w=180)
        self.ln(1)
        self.set_font("DejaVu", "", 10)
        self.multi_cell(0, 5, caption)
        self.ln(3)


def save_jpg(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), to_uint8(image), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample6 = cv2.imread(str(DATA / "sample6.jpg"), cv2.IMREAD_GRAYSCALE)
    sample1 = cv2.imread(str(DATA / "sample1.jpg"), cv2.IMREAD_GRAYSCALE)
    if sample6 is None or sample1 is None:
        raise SystemExit("не удалось прочитать изображения из data/")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    raster = cv2.morphologyEx(sample6, cv2.MORPH_CLOSE, kernel)
    frequency = gaussian_lowpass(sample6, sigma=16.0)
    sequential = gaussian_lowpass(raster, sigma=16.0)

    periodic, notch = notch_periodic(sample1)
    before_spectrum = spectrum_image(sample1)
    after_spectrum = spectrum_image(periodic)

    save_jpg(OUT / "sample6_original.jpg", sample6)
    save_jpg(OUT / "sample6_raster.jpg", raster)
    save_jpg(OUT / "sample6_frequency.jpg", frequency)
    save_jpg(OUT / "sample6_raster_then_frequency.jpg", sequential)
    save_jpg(OUT / "sample1_original.jpg", sample1)
    save_jpg(OUT / "sample1_notch.jpg", periodic)
    save_jpg(OUT / "sample1_spectrum_before.jpg", before_spectrum)
    save_jpg(OUT / "sample1_spectrum_after.jpg", after_spectrum)
    np.save(OUT / "sample1_notch_mask.npy", notch)

    pdf_path = ROOT / f"{SURNAME}.pdf"
    pdf = Report()
    pdf.add_page()
    pdf.heading(f"Практика 2. Частотная фильтрация. {SURNAME}.")
    pdf.paragraph(
        "На sample6.jpg мелкие тёмные точки убраны двумя способами. "
        "Растровая фильтрация — морфологическое закрытие эллипсом 15×15: "
        "оно заполняет тёмные точки, не размывая границы клеток. "
        "Частотная — гауссов фильтр низких частот через БПФ (sigma = 16 отсчётов спектра). "
        "Третий кадр — закрытие, затем тот же фильтр низких частот."
    )
    pdf.figure(OUT / "sample6_original.jpg", "sample6.jpg, исходное изображение.")
    pdf.figure(OUT / "sample6_raster.jpg", "Растровая фильтрация: морфологическое закрытие.")
    pdf.figure(OUT / "sample6_frequency.jpg", "Частотный фильтр низких частот.")
    pdf.figure(OUT / "sample6_raster_then_frequency.jpg", "Сначала закрытие, затем фильтр низких частот.")
    pdf.paragraph(
        "На sample1.jpg периодическая помеха — наклонные полосы "
        "(в тексте задания она названа горизонтальной). "
        "В спектре это яркий пик в стороне от осей и от нулевой частоты; "
        "гармоники лежат на той же прямой. "
        "Режекторный фильтр гасит эту прямую и оставляет окрестность нуля."
    )
    pdf.figure(OUT / "sample1_original.jpg", "sample1.jpg, исходное изображение.")
    pdf.figure(OUT / "sample1_notch.jpg", "После режекторного фильтра.")
    pdf.figure(OUT / "sample1_spectrum_before.jpg", "Логарифм модуля спектра до фильтрации.")
    pdf.figure(OUT / "sample1_spectrum_after.jpg", "Логарифм модуля спектра после фильтрации.")
    pdf.output(str(pdf_path))
    print(f"PDF: {pdf_path}")
    print(f"кадры: {OUT}")


if __name__ == "__main__":
    main()
