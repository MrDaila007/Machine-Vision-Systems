# Graph Report - Machine_vision_systems  (2026-09-30)

## Corpus Check
- 20 files · ~992,109 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 159 nodes · 244 edges · 19 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- keypoints.py
- Фамилия.py
- track_particles.py
- optical_flow_tracker.py
- correlation_filter.py
- morphology.py
- Практика 5. Особые точки
- histogram.py
- Практика 6. Математическая морфология
- Практика 1. Корреляционный фильтр
- Практика 2. Частотная фильтрация
- Практика 3. Гистограмма
- Практика 4. Выделение линий
- Практика 7. Характеристики объектов
- Практика 8. Анализ видео
- Практика 9. Свой трекер на оптическом потоке
- Системы машинного зрения
- lines.py
- features.py

## God Nodes (most connected - your core abstractions)
1. `main()` - 9 edges
2. `Report` - 7 edges
3. `rotation_study()` - 7 edges
4. `match_queries()` - 7 edges
5. `Практика 5. Особые точки` - 7 edges
6. `main()` - 6 edges
7. `main()` - 6 edges
8. `CentroidTracker` - 6 edges
9. `main()` - 6 edges
10. `FlowTracker` - 6 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (19 total, 0 thin omitted)

### Community 0 - "keypoints.py"
Cohesion: 0.30
Nodes (18): draw_points(), f1_score(), harris_points(), ink_bands(), load_gray(), main(), make_sheet(), match_queries() (+10 more)

### Community 1 - "Фамилия.py"
Cohesion: 0.21
Nodes (12): FPDF, gaussian_lowpass(), main(), notch_periodic(), ndarray, Path, Частотная и растровая фильтрация. Имя файла — плейсхолдер. Замените SURNAME на…, Гасит полосу спектра вдоль самой яркой периодической помехи. Пик ищется в… (+4 more)

### Community 2 - "track_particles.py"
Cohesion: 0.23
Nodes (12): cell_body(), CentroidTracker, color_for(), detect_particles(), draw(), main(), open_writer(), ndarray (+4 more)

### Community 3 - "optical_flow_tracker.py"
Cohesion: 0.29
Nodes (9): cell_body(), color_for(), detect_particles(), draw(), FlowTracker, main(), ndarray, Свой трекер частиц на оптическом потоке Лукаса–Канаде. (+1 more)

### Community 4 - "correlation_filter.py"
Cohesion: 0.39
Nodes (11): correlate(), load_gray(), main(), montage(), normalize_kernel(), ndarray, Path, Корреляционный фильтр на NumPy. Сама свёртка с ядром написана только через… (+3 more)

### Community 5 - "morphology.py"
Cohesion: 0.40
Nodes (10): caption(), colorize(), fiber(), grains(), main(), measure_component(), ndarray, Математическая морфология: зёрна и волокно. (+2 more)

### Community 6 - "Практика 5. Особые точки"
Cohesion: 0.25
Nodes (7): Данные, Задание 1, Задание 2, Задание 3, Запуск, Практика 5. Особые точки, Результат

### Community 7 - "histogram.py"
Cohesion: 0.43
Nodes (6): load_xray(), main(), ndarray, Path, Подбор яркостного окна для рентгенограммы сварного шва., window_to_uint8()

### Community 8 - "Практика 6. Математическая морфология"
Cohesion: 0.29
Nodes (6): Волокно, Данные, Запуск, Зёрна, Практика 6. Математическая морфология, Результат

### Community 9 - "Практика 1. Корреляционный фильтр"
Cohesion: 0.40
Nodes (4): Данные, Запуск, Практика 1. Корреляционный фильтр, Результат

### Community 10 - "Практика 2. Частотная фильтрация"
Cohesion: 0.40
Nodes (4): Данные, Запуск, Практика 2. Частотная фильтрация, Результат

### Community 11 - "Практика 3. Гистограмма"
Cohesion: 0.40
Nodes (4): Данные, Запуск, Практика 3. Гистограмма, Результат

### Community 12 - "Практика 4. Выделение линий"
Cohesion: 0.40
Nodes (4): Данные, Запуск, Практика 4. Выделение линий, Результат

### Community 13 - "Практика 7. Характеристики объектов"
Cohesion: 0.40
Nodes (4): Данные, Запуск, Практика 7. Характеристики объектов, Результат

### Community 14 - "Практика 8. Анализ видео"
Cohesion: 0.40
Nodes (4): Данные, Запуск, Практика 8. Анализ видео, Результат

### Community 15 - "Практика 9. Свой трекер на оптическом потоке"
Cohesion: 0.40
Nodes (4): Данные, Запуск, Практика 9. Свой трекер на оптическом потоке, Результат

### Community 16 - "Системы машинного зрения"
Cohesion: 0.40
Nodes (3): Запуск, Окружение, Системы машинного зрения

### Community 17 - "lines.py"
Cohesion: 0.67
Nodes (3): line_angle_deg(), main(), Выделение теплохода прямыми линиями (Canny + Хаф).

### Community 18 - "features.py"
Cohesion: 0.67
Nodes (3): axis_lengths(), main(), Площадь, периметр, угол и оси объектов на sample30.jpg.

## Knowledge Gaps
- **34 isolated node(s):** `Данные`, `Запуск`, `Результат`, `Данные`, `Запуск` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `Данные`, `Запуск`, `Результат` to the rest of the system?**
  _34 weakly-connected nodes found - possible documentation gaps or missing edges._