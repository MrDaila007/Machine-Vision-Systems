# Системы машинного зрения

Девять практических работ. Исходные кадры лежат в `Lab*/data`, скрипт каждой работы запускается из своей папки и пишет результат в `Lab*/output`.

Условие заданий собрано в [tasks.md](tasks.md).

## Окружение

Нужен [Miniconda](https://docs.conda.io/) или Anaconda.

```bash
conda env create -f environment.yml
conda activate mvs
```

Окружение называется `mvs` (Python 3.11, NumPy, OpenCV, scikit-image, pydicom и остальные пакеты из `environment.yml`).

## Запуск

```bash
conda activate mvs
cd Lab1 && python correlation_filter.py
cd ../Lab2 && python "Eliseev D.py"
cd ../Lab3 && python histogram.py
cd ../Lab4 && python lines.py
cd ../Lab5 && python keypoints.py
cd ../Lab6 && python morphology.py
cd ../Lab7 && python features.py
cd ../Lab8 && python track_particles.py
cd ../Lab9 && python optical_flow_tracker.py
```

Видео `MKCell.mp4` длится около 150 секунд. Скрипты Lab8 и Lab9 обрабатывают его целиком в половинном разрешении, на это уходит несколько минут.

В практике 2 код и PDF называются `Eliseev D`.

Файл `boat2 (1).png` был точной копией `boat2.png` и в работу не входит.
