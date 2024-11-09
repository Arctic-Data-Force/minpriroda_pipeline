import os
import cv2
import pandas as pd
import csv
from tqdm import tqdm
import numpy as np
import torch
import logging
from easy_ViTPose import VitInference

# Настройки
DEBUG = False  # Установите в True для включения режима отладки
DEBUG_FOLDER = 'debug_images'  # Папка для сохранения отладочных изображений
ANNOTATION_PATH = 'train_data_minprirodi/annotation.csv'  # Путь к аннотации

# Конфигурация логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# {0: 'pig', 1: 'wolf', 2: 'fox', 3: 'buffalo', 4: 'moose', 5: 'bear', 6: 'hare', 7: 'tiger', 8: 'deer', 9: 'bird', 10: 'bobcat', 11: 'leopard', 12: 'goat', 13: 'yellow-throated marten'}
# Определение классов маленьких животных по их индексам
SMALL_ANIMAL_CLASSES = {2, 6, 9, 10, 13}

def get_device():
    """Определяет устройство для вычислений."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(model_path, yolo_path, model_size, dataset, yolo_size, device):
    """Инициализирует модель VitInference."""
    model = VitInference(
        model=model_path,
        yolo=yolo_path,
        model_name=model_size,
        det_class="animals",
        dataset=dataset,
        yolo_size=yolo_size,
        detection_model_type='rtdetr',
        is_video=False
    )
    logging.info(f"YOLO классы: {model.detector.names}")
    return model

def perform_iqa(cropped_img, threshold=100):
    """
    Выполняет оценку качества изображения с использованием дисперсии Лапласиана.

    Args:
        cropped_img (numpy.ndarray): Обрезанное изображение.
        threshold (float): Порог для дисперсии, определяющий качество.

    Returns:
        int: 1 для хорошего качества, 0 для плохого.
    """
    try:
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return 1 if variance > threshold else 0
    except Exception as e:
        logging.error(f"Ошибка при выполнении IQA: {e}")
        return 0

def check_conditions(keypoints, animal_class):
    """
    Проверяет условия для обнаруженных ключевых точек животного.

    Аргументы:
        keypoints (numpy.ndarray): Массив ключевых точек с формой (N, 3).
        animal_class (int): Класс животного.

    Возвращает:
        bool: True, если условия выполнены, иначе False.
    """
    if keypoints is None:
        logging.warning("Ключевые точки не обнаружены.")
        return False

    keypoints = np.array(keypoints)

    # Проверка формы ключевых точек
    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        logging.warning(f"Непредвиденная форма ключевых точек: {keypoints.shape}")
        return False

    # Обработка ключевых точек с координатами (0, 0) как недоступных
    missing_mask = (keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)
    keypoints[missing_mask, 2] = 0  # Устанавливаем confidence в 0 для недоступных точек

    conf_threshold = 0.35  # Порог доверия для проверки

    # Индексы ключевых точек (нумерация с нуля)
    torso_keypoints = [3, 4, 5, 6, 8, 9, 11, 12, 14, 15]
    head_keypoints = [0, 1, 2]
    leg_tail_keypoints = [4, 7, 10, 13, 16]

    # Проверка видимости туловища (не менее 60% ключевых точек туловища видны)
    visible_torso_points = np.sum(keypoints[torso_keypoints, 2] > conf_threshold)
    required_torso_points = max(1, int(len(torso_keypoints) * 0.6))  # Минимум 1 точка
    if visible_torso_points < required_torso_points:
        logging.warning("Туловище скрыто более чем на 40%.")
        return False

    # Проверка видимости головы (хотя бы одна ключевая точка головы видна)
    head_visible = np.any(keypoints[head_keypoints, 2] > conf_threshold)
    if not head_visible:
        logging.warning("Голова не видна.")
        return False

    # Если животное не относится к маленьким, проверяем видимость лап и хвоста
    if animal_class not in SMALL_ANIMAL_CLASSES:
        leg_tail_visible = np.any(keypoints[leg_tail_keypoints, 2] > conf_threshold)
        if not leg_tail_visible:
            logging.warning("Лапы или хвост не видны.")
            return False

    return True

def draw_keypoints(img, keypoints, offset=(0, 0)):
    """
    Отрисовывает ключевые точки на изображении для визуализации.

    Args:
        img (numpy.ndarray): Оригинальное изображение.
        keypoints (numpy.ndarray): Массив ключевых точек.
        offset (tuple): Смещение (x_offset, y_offset) для коррекции позиций ключевых точек.

    Returns:
        numpy.ndarray: Изображение с отрисованными ключевыми точками.
    """
    x_offset, y_offset = offset
    for kp in keypoints:
        x, y, conf = kp
        if conf > 0:
            cv2.circle(img, (int(x + x_offset), int(y + y_offset)), 3, (0, 255, 0), -1)
    return img

def process_image(image_path, model, device, annotations_dict, conf_threshold=0.35, bbox_min_size=128, bbox_conf_threshold=0.5):
    """
    Обрабатывает одно изображение: обнаруживает объекты, выполняет IQA, запускает позу и проверяет условия.

    Args:
        image_path (str): Путь к файлу изображения.
        model (VitInference): Инициализированная модель VitInference.
        device (torch.device): Устройство для вычислений.
        annotations_dict (dict): Словарь аннотаций для сравнения.
        conf_threshold (float): Порог доверия для видимости ключевых точек.
        bbox_min_size (int): Минимальный размер бокса по ширине и высоте.
        bbox_conf_threshold (float): Порог доверия для боксов.

    Returns:
        list: Список записей для сабмишена.
    """
    submission_records = []
    image_file = os.path.basename(image_path)
    img = cv2.imread(image_path)

    if img is None:
        logging.warning(f"Не удалось прочитать изображение {image_file}. Пропуск.")
        return submission_records

    height, width = img.shape[:2]
    results = model.detector.predict(img, device=device)[0]

    if len(results) == 0:
        logging.info(f"Нет детекций в изображении {image_file}.")
        return submission_records

    boxes = results.boxes  # Boxes object для выводов bbox

    # Создаем копию для отрисовки, если DEBUG включен
    if DEBUG:
        img_debug = img.copy()
        # Добавляем информацию о количестве детекций
        cv2.putText(img_debug, f"Detections: {len(boxes)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Получаем аннотации для текущего изображения
    image_annotations = annotations_dict.get(image_file, [])

    # Отрисовка аннотированных bounding box
    if DEBUG and image_annotations:
        for ann in image_annotations:
            ann_bbox = ann['Bbox']
            ann_class = ann['Class']
            # Декодируем координаты bbox
            x_center, y_center, norm_width, norm_height = map(float, ann_bbox.split(','))
            x1_ann = int((x_center - norm_width / 2) * width)
            y1_ann = int((y_center - norm_height / 2) * height)
            x2_ann = int((x_center + norm_width / 2) * width)
            y2_ann = int((y_center + norm_height / 2) * height)
            # Отрисовываем аннотированный bbox синим цветом
            cv2.rectangle(img_debug, (x1_ann, y1_ann), (x2_ann, y2_ann), (255, 0, 0), 2)
            label_ann = f"GT Class: {ann_class}"
            cv2.putText(img_debug, label_ann, (x1_ann, y1_ann - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for idx, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]  # Преобразуем в список float
            conf = box.conf.tolist()[0]  # Уровень доверия
            cls = int(box.cls.tolist()[0])  # Класс

            logging.debug(f"Деталь бокса - x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, conf: {conf}, cls: {cls}")

            # Проверяем, превышает ли уверенность бокса порог
            if conf < bbox_conf_threshold:
                logging.info(f"Уверенность бокса {conf:.2f} ниже порога {bbox_conf_threshold}. Классификация как 0.")
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                # Нормализуем координаты бокса в диапазоне [0,1]
                x_center = (x1 + bbox_width / 2) / width
                y_center = (y1 + bbox_height / 2) / height
                norm_width = bbox_width / width
                norm_height = bbox_height / height
                bbox_string = f"{x_center},{y_center},{norm_width},{norm_height}"

                submission_records.append({
                    'Name': image_file,
                    'Bbox': bbox_string,
                    'Class': 0  # Плохое качество из-за низкой уверенности
                })

                if DEBUG:
                    # Отрисовка бокса красным для класса 0
                    cv2.rectangle(img_debug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    label = f"Class: 0"
                    cv2.putText(img_debug, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    # Добавляем информацию о количестве ключевых точек (0 в данном случае)
                    cv2.putText(img_debug, f"Keypoints: 0", (int(x1), int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                continue  # Пропускаем дальнейшую обработку этого бокса

            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # Нормализуем координаты бокса в диапазоне [0,1]
            x_center = (x1 + bbox_width / 2) / width
            y_center = (y1 + bbox_height / 2) / height
            norm_width = bbox_width / width
            norm_height = bbox_height / height
            bbox_string = f"{x_center},{y_center},{norm_width},{norm_height}"

            bbox_class = 1  # Инициализируем как хорошее качество

            # Проверяем, меньше ли bbox минимального размера
            if bbox_width < bbox_min_size or bbox_height < bbox_min_size:
                bbox_class = 0
                submission_records.append({
                    'Name': image_file,
                    'Bbox': bbox_string,
                    'Class': bbox_class
                })

                if DEBUG:
                    # Отрисовка бокса красным для класса 0
                    cv2.rectangle(img_debug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    label = f"Class: 0"
                    cv2.putText(img_debug, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.putText(img_debug, f"Keypoints: 0", (int(x1), int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                continue

            # Обрезаем изображение внутри bbox
            x1_int, y1_int, x2_int, y2_int = map(int, [x1, y1, x2, y2])
            x1_int = max(0, x1_int)
            y1_int = max(0, y1_int)
            x2_int = min(width, x2_int)
            y2_int = min(height, y2_int)

            cropped_img = img[y1_int:y2_int, x1_int:x2_int]

            # Запускаем инференс VitPose на обрезанном изображении
            keypoints_dict = model.inference(cropped_img)

            keypoints_count = 0

            if not keypoints_dict:
                logging.warning(f"Не обнаружены ключевые точки для {image_file}.")
                bbox_class = 0
                submission_records.append({
                    'Name': image_file,
                    'Bbox': bbox_string,
                    'Class': bbox_class
                })

                if DEBUG:
                    # Отрисовка бокса красным для класса 0
                    cv2.rectangle(img_debug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    label = f"Class: 0"
                    cv2.putText(img_debug, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.putText(img_debug, f"Keypoints: 0", (int(x1), int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                continue

            # Итерация по каждому набору ключевых точек
            for id, keypoints in keypoints_dict.items():
                logging.debug(f"Форма ключевых точек для {image_file}, ID {id}: {keypoints.shape}")

                keypoints_count = keypoints.shape[0]

                # Проверяем условия на ключевых точках, передавая класс
                conditions_met = check_conditions(keypoints, cls)
                if not conditions_met:
                    bbox_class = 0

                submission_records.append({
                    'Name': image_file,
                    'Bbox': bbox_string,
                    'Class': bbox_class
                })

                # Сравнение с аннотациями и логирование расхождений
                if image_file in annotations_dict:
                    for annotation in annotations_dict[image_file]:
                        ann_bbox = annotation['Bbox']
                        ann_class = annotation['Class']
                        # Здесь можно добавить логику сравнения предсказанных и аннотированных бокс
                        if bbox_class != ann_class:
                            logging.warning(f"Расхождение классов для {image_file}: Предсказано {bbox_class}, Аннотация {ann_class}")

                if DEBUG:
                    # Выбираем цвет в зависимости от класса
                    color = (0, 255, 0) if bbox_class == 1 else (0, 0, 255)
                    cv2.rectangle(img_debug, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"Class: {bbox_class}"
                    cv2.putText(img_debug, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Добавляем информацию о количестве ключевых точек
                    cv2.putText(img_debug, f"Keypoints: {keypoints_count}", (int(x1), int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Отрисовка ключевых точек (опционально)
                    # img_debug = draw_keypoints(img_debug, keypoints, offset=(x1_int, y1_int))

            # Если было несколько наборов ключевых точек, выходим из цикла
            break

        except Exception as e:
            logging.error(f"Ошибка при обработке бокса в {image_file}: {e}")
            submission_records.append({
                'Name': image_file,
                'Bbox': bbox_string if 'bbox_string' in locals() else "",
                'Class': 0
            })

            if DEBUG:
                # Отрисовка бокса красным для класса 0
                cv2.rectangle(img_debug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                label = f"Class: 0"
                cv2.putText(img_debug, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.putText(img_debug, f"Keypoints: 0", (int(x1), int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # Сохраняем отладочное изображение, если режим отладки включен
    if DEBUG:
        os.makedirs(DEBUG_FOLDER, exist_ok=True)
        debug_image_path = os.path.join(DEBUG_FOLDER, image_file)
        cv2.imwrite(debug_image_path, img_debug)
        logging.info(f"Отладочное изображение сохранено в '{debug_image_path}'.")

    return submission_records

def save_submission(submission_records, output_path='submission.csv'):
    """
    Сохраняет записи сабмишена в CSV файл.

    Args:
        submission_records (list): Список словарей с данными для сабмишена.
        output_path (str): Путь для сохранения CSV файла.
    """
    if not submission_records:
        logging.warning("Нет записей для сохранения.")
        return

    submission_df = pd.DataFrame(submission_records, columns=['Name', 'Bbox', 'Class'])
    submission_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    logging.info(f"Сабмишн сохранен в '{output_path}'.")

def load_annotations(annotation_path):
    """
    Загружает аннотации из CSV файла.

    Args:
        annotation_path (str): Путь к файлу аннотации.

    Returns:
        dict: Словарь аннотаций, где ключ - имя изображения, значение - список аннотаций.
    """
    if not os.path.exists(annotation_path):
        logging.error(f"Файл аннотации не найден по пути '{annotation_path}'.")
        return {}

    annotations_df = pd.read_csv(annotation_path)
    annotations_dict = {}

    for _, row in annotations_df.iterrows():
        name = row['Name']
        bbox = row['Bbox']
        cls = int(row['Class'])

        if name not in annotations_dict:
            annotations_dict[name] = []

        annotations_dict[name].append({
            'Bbox': bbox,
            'Class': cls
        })

    logging.info(f"Загружено аннотаций для {len(annotations_dict)} изображений.")
    return annotations_dict

def main():
    """Главная функция для организации процесса обработки."""
    device = get_device()
    logging.info(f"Используемое устройство: {device}")

    # Параметры конфигурации
    MODEL_TYPE = "torch"
    YOLO_TYPE = "torch"
    MODEL_SIZE = 'h'  # ['s', 'b', 'l', 'h']
    YOLO_SIZE = 640  # Чаще всего YOLO использует числовые размеры, например 640
    DATASET = 'apt36k'  # ['coco_25', 'coco', 'wholebody', 'mpii', 'aic', 'ap10k', 'apt36k']
    model_path = 'models/vitpose-h-apt36k.pth'
    yolo_path = 'models/rtdetr_l_e10_bs8_640.pt'
    images_folder = 'train_data_minprirodi/images'
    output_csv = 'submission.csv'
    BBOX_CONF_THRESHOLD = 0.8  # Порог доверия для боксов

    # Инициализируем модель
    model = initialize_model(
        model_path=model_path,
        yolo_path=yolo_path,
        model_size=MODEL_SIZE,
        dataset=DATASET,
        yolo_size=YOLO_SIZE,
        device=device
    )

    # Проверяем существование папки с изображениями
    if not os.path.exists(images_folder):
        logging.error(f"Папка с изображениями не найдена по пути '{images_folder}'. Проверьте путь.")
        raise FileNotFoundError(f"Папка с изображениями не найдена по пути '{images_folder}'.")

    # Загружаем аннотации
    annotations_dict = load_annotations(ANNOTATION_PATH)

    # Готовим список для накопления записей сабмишена
    all_submission_records = []

    # Получаем список файлов изображений
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Обрабатываем каждое изображение с прогресс-баром
    for image_file in tqdm(image_files, desc="Обработка изображений"):
        image_path = os.path.join(images_folder, image_file)
        records = process_image(
            image_path,
            model,
            device,
            annotations_dict,
            conf_threshold=0.1,
            bbox_min_size=128,
            bbox_conf_threshold=BBOX_CONF_THRESHOLD
        )
        all_submission_records.extend(records)

    # Сохраняем файл сабмишена
    save_submission(all_submission_records, output_csv)

    # Опционально: Просмотр DataFrame сабмишена
    if all_submission_records:
        submission_df = pd.DataFrame(all_submission_records)
        print(submission_df.head())
        print(submission_df['Class'].value_counts())
    else:
        logging.info("Нет записей для отображения.")

if __name__ == "__main__":
    main()
