# trainingyolo
launching training for yolo
Автоматизированный пайплайн для обучения моделей YOLO с поддержкой пользовательских данных

## 🚀 Основные возможности

- **Автоматическая обработка данных**
  - Распаковка архивов с данными
  - Автоматическое разделение на train/val
  - Проверка целостности данных
- **Оптимизация ресурсов**
  - Автовыбор устройства (CPU/GPU/MPS)
  - Контроль использования памяти
- **Гибкая конфигурация**
  - Настройка путей через конфиг
  - Параметры обучения в одном месте
- **Встроенные проверки**
  - Валидация структуры данных
  - Проверка соответствия классов
  - Контроль качества данных

## ⚙️ Требования

- Python 3.9+
- NVIDIA GPU (опционально)
- Ultralytics YOLO 8.3.82+

```bash
# Установка зависимостей
pip install ultralytics torch

🛠 Использование
Подготовка данных

выгрузите из Label Studio  ZIP-архив с именем project-*.zip

Структура архива:
project-XXXX.zip/
├── images/     # Изображения
├── labels/     # Разметка YOLO
├── classes.txt # Список классов
└── notes.json  # Метаданные (опционально)

Конфигурация

Поместите файл data.yaml в корневую директорию (это структура файла):
path: C:/temp/apple/data
train: train/images
val: val/images
nc: 4
names: ["apple", "defect", "orange", "pear"]

Запуск
python razvernut_traning.py

📂 Структура проекта
C:/temp/apple/
├── data.yaml          # Конфиг обучения
├── data/              # Автогенерируемая
│   ├── train/         # Тренировочные данные
│   └── val/           # Валидационные данные
└── runs/              # Результаты обучения



# trainingyolo
Automated pipeline for YOLOv8 model training with user data support

## 🚀 Main features

- **Automatic data processing**
- Unpacking data archives
- Automatic separation into train/val
- Data integrity check
- **Resource optimization**
- Device auto-selection (CPU/GPU/MPS)
- Memory usage monitoring
- **Flexible configuration**
- Configuration of paths via config
  - Training parameters in one place
- **Built-in checks**
- Data structure validation
- Class matching verification
  - Data quality control

## ⚙️ Requirements

- Python 3.9+
- NVIDIA GPU (optional)
- Ultralytics YOLO 8.3.82+

```bash
# Installing
pip install ultralytics torch dependencies
