from ultralytics import YOLO
import shutil
from pathlib import Path
import os
import zipfile
import random
import platform
import torch
from typing import Optional, List

# =============================================
# = Г Л О Б А Л Ь Н Ы Е   Н А С Т Р О Й К И =
# =============================================

BASE_DIR = Path("C:/temp/apple")
DATA_DIR = BASE_DIR / "data"
TRAIN_IMAGES = DATA_DIR / "train" / "images"
TRAIN_LABELS = DATA_DIR / "train" / "labels"
VAL_IMAGES = DATA_DIR / "val" / "images"
VAL_LABELS = DATA_DIR / "val" / "labels"

CLEAN_BEFORE_PROCESS = True
VAL_RATIO = 0.2
SEED = 42


# =============================================
# = В С П О М О Г А Т Е Л Ь Н Ы Е   Ф У Н К Ц И И =
# =============================================

def clean_directory(*paths: Path) -> None:
    """Очистка и создание директорий при необходимости"""
    for path in paths:
        if path.exists():
            for item in path.glob('*'):
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item)
            print(f"Очищено: {path}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"Создана директория: {path}")


def find_latest_zip() -> Optional[Path]:
    """Поиск последнего ZIP-архива с данными"""
    zip_files = list(BASE_DIR.glob("project-*-*.zip"))
    return max(zip_files, key=lambda x: x.stat().st_mtime) if zip_files else None


def unzip_and_process(zip_path: Path) -> None:
    """Обработка и распаковка архива"""
    extract_dir = BASE_DIR / zip_path.stem
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Архив распакован в: {extract_dir}")

    # Перенос данных в train
    for data_type in ["images", "labels"]:
        src = extract_dir / data_type
        dest = TRAIN_IMAGES if data_type == "images" else TRAIN_LABELS
        dest.mkdir(parents=True, exist_ok=True)

        moved_files = 0
        for file in src.glob('*'):
            shutil.move(str(file), str(dest))
            moved_files += 1
        print(f"Перенесено {moved_files} файлов в {dest}")

    # Копирование вспомогательных файлов
    for filename in ["classes.txt", "notes.json"]:
        src = extract_dir / filename
        if src.exists():
            shutil.copy(src, DATA_DIR / "train")
            print(f"Скопирован файл: {filename}")


def create_validation_set() -> None:
    """Создание валидационной выборки"""
    random.seed(SEED)
    all_images = list(TRAIN_IMAGES.glob('*'))
    random.shuffle(all_images)

    num_val = int(len(all_images) * VAL_RATIO)
    val_images = all_images[:num_val]

    for img_path in val_images:
        # Перенос изображений
        dest_img = VAL_IMAGES / img_path.name
        shutil.move(str(img_path), str(dest_img))

        # Перенос соответствующих меток
        label_name = img_path.stem + '.txt'
        src_label = TRAIN_LABELS / label_name
        if src_label.exists():
            dest_label = VAL_LABELS / label_name
            shutil.move(str(src_label), str(dest_label))

    print(f"Создан валидационный набор из {num_val} файлов")


def get_device() -> str:
    """Определение доступного устройства для обучения"""
    try:
        if platform.system() == 'Darwin':
            return 'mps' if torch.backends.mps.is_available() else 'cpu'
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception as e:
        print(f"Ошибка определения устройства: {e}")
        return 'cpu'


def validate_paths() -> None:
    """Проверка целостности структуры данных"""
    required_paths = [
        BASE_DIR / "data.yaml",
        TRAIN_IMAGES,
        TRAIN_LABELS,
        VAL_IMAGES,
        VAL_LABELS,
        DATA_DIR / "train" / "classes.txt"
    ]

    # Проверка существования путей
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Отсутствует обязательный элемент: {path}")

    # Проверка наличия данных
    train_images = list(TRAIN_IMAGES.glob("*"))
    val_images = list(VAL_IMAGES.glob("*"))

    if not train_images:
        raise ValueError(f"Нет изображений для обучения в {TRAIN_IMAGES}")
    if not val_images:
        raise ValueError(f"Нет изображений для валидации в {VAL_IMAGES}")

    # Проверка соответствия классов
    with open(DATA_DIR / "train" / "classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    expected_classes = ["apple", "defect", "orange", "pear"]
    if classes != expected_classes:
        raise ValueError(
            f"Классы в classes.txt не соответствуют ожидаемым. Получено: {classes}, Ожидалось: {expected_classes}")


# =============================================
# = О С Н О В Н О Й   Б Л О К   В Ы П О Л Н Е Н И Я =
# =============================================

def main() -> None:
    """Основной рабочий процесс"""
    device = get_device()
    print(f"🚀 Используемое устройство: {device.upper()}")

    # Инициализация директорий
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Очистка данных
    if CLEAN_BEFORE_PROCESS:
        clean_directory(TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS)

    # Обработка архива
    if (latest_zip := find_latest_zip()) is not None:
        unzip_and_process(latest_zip)
    else:
        print("⚠️ Архив с данными не найден!")
        return

    # Создание валидационной выборки
    create_validation_set()

    # Валидация структуры данных
    try:
        validate_paths()
        print("✅ Проверка структуры данных успешно пройдена")
    except Exception as e:
        print(f"❌ Ошибка проверки данных: {e}")
        return

    # Запуск обучения
    print("\nЗапуск процесса обучения...")
    model = YOLO("yolo11n.pt")

    try:
        model.train(
            data=BASE_DIR / "data.yaml",
            epochs=100,
            imgsz=640,
            device=device,
            verbose=True
        )
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")


if __name__ == "__main__":
    main()