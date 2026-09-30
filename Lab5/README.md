# Практика 5. Особые точки

Детектор Харриса (`cv2.cornerHarris`). Параметры из задания: `blockSize` — размер окрестности угла, `ksize` — апертура Собеля.

## Задание 1

На `fingerprints.jpg` перебираются `blockSize`, `ksize` и относительный порог. Берётся набор, ближе всего к красным точкам на `resultfp.jpg` (F1 при допуске 6 пикселей). Выбранные значения пишутся в `output/harris_params.txt`.

## Задание 2

`star.jpg` и `fp2.jpg` поворачиваются на полный оборот с шагом 1°. На каждом угле считаются точки Харриса. В `output/` — контактный лист через каждые 30° и график числа точек от угла.

Вопрос, откуда берутся точки, не устойчивые к повороту, в отчёт не входит: он для обсуждения на занятии.

## Задание 3

`fingerprints.jpg` режется на девять шаблонов по светлым промежуткам. Для `fp2.jpg` и `fp3.jpg` считаются ключевые точки SIFT, ближайший шаблон — тот, у которого больше инлайеров после RANSAC. Номера шаблонов идут по строкам слева направо, сверху вниз.

## Данные

- `data/fingerprints.jpg`, `data/resultfp.jpg`
- `data/star.jpg`, `data/fp2.jpg`, `data/fp3.jpg`

## Запуск

```bash
conda activate mvs
python keypoints.py
```

## Результат

`output/harris_overlay.jpg`, `output/harris_vs_reference.jpg`, `output/star_sheet.jpg`, `output/fp2_sheet.jpg`, графики `*_counts.png`, `output/match_fp2.jpg`, `output/match_fp3.jpg`, `output/matches.txt`.
