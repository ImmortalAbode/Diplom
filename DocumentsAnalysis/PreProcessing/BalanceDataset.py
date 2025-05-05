import json
import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import pdf2image as p2i

import ConsoleOutput.PrintFunctions as pf


"""=====================================================================================================> 
| Класс, отвечающий за балансировку количества оригинальных и поддельных изображений.                   |
| Класс производит агументацию поддельных изображений до числа оригинальных путем различных             |
| манипуляций с ними. Удаляет чисто белые патчи из оригинальных для улучшения обучения.                 |
<====================================================================================================="""
class DatasetBalancer:
    def __init__(self, pdf_dir: str,  annotations_dir: str, output_dir: str, patch_size: int, max_patches: int):
        # Запоминание необходимых путей.
        self.pdf_dir = pdf_dir                      # путь к .pdf выборке документов
        self.annotations_dir = annotations_dir      # путь к аннотации .pdf документа
        self.output_dir = output_dir                # путь к папке сохранения патчей

        self.patch_size = patch_size    # размер патча
        self.max_patches = max_patches  # максимальное число патчей

        # Создание нужных для выходного датасета папок, а также их очистка в случае наличия в них файлов.
        os.makedirs(output_dir, exist_ok=True)
        self._clear_dataset_subfolders(output_dir)
        
    # Метод, который выполняет очистку папки патчей от существующих файлов.
    def _clear_dataset_subfolders(self, output_dir: str):
        subfolders = ['genuine', 'forged']
        # Спуск вниз к папкам подпапок.
        for subfolder in subfolders:
           subfolder_path = os.path.join(output_dir, subfolder)
           os.makedirs(subfolder_path, exist_ok=True)

           # Удаляем файлы в папке.
           for filename in os.listdir(subfolder_path):
               file_path = os.path.join(subfolder_path, filename)
               if os.path.isfile(file_path):
                   os.remove(file_path)

    # Генерация одного патча с цепочкой из нескольких аугментаций для уменьшения числа повторяющихся патчей.
    def _random_augment_chain(self, patch: Image.Image, depth: int):
        ops = self._augment_ops()
        img = patch.copy()
        for op in random.sample(ops, k=depth):
            img = op(img)
        return img

    # Метод, возвращающий список доступных трансформаций патча в процессе аугментации.
    def _augment_ops(self):
        ops = [
            # --- Геометрия ---
            lambda x: x.rotate((random.choice([-180, -90, 90, 180]))),                  # Случайный поворот изображения на +-90 и +-180 градусов

            # --- Освещение ---
            lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.8, 1.2)),     # случайное изменение яркости (от 80% до 120% от исходной)
            lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.8, 1.2)),       # случайное изменение контрастности (от 80% до 120%)
            lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.9, 1.1)),          # случайное изменение насыщенности

            # --- Шум ---
            lambda x: ( lambda img: Image.fromarray( np.clip( np.array(img).astype(np.int16) + np.random.randint(-20, 20, size=np.array(img).shape), 0, 255).astype(np.uint8)))(x),
        ]
        return ops

    # Метод, который возвращает список изображений документа, приведенных к виду, по которому получались патчи в процессе аннотирования.
    def _load_pdf_images(self, pdf_file: str, used_dpi: int):
        if used_dpi is None:
            pdf_images = p2i.convert_from_path(pdf_file)
        else:
            pdf_images = p2i.convert_from_path(pdf_file, dpi=used_dpi)
        padded_images = []
        for img in pdf_images:
            img = img.convert("RGB")                                        # Получаем изображение страницы в формате RGB.

            # Предобработка изображения, чтобы не было черных полей при выходе за пределы исходного изображения.
            img_np = np.array(img)                                          # представление изображение в виде нормальных пикселей
            # Рассчитываем, сколько дополнительных пикселей нужно добавить.
            h, w, _ = img_np.shape                                          # представление (H, W, C), где С - каналы, H - высота, W - ширина
            new_h = ((h // self.patch_size) + 1) * self.patch_size          # определение новой высоты изображения
            new_w = ((w // self.patch_size) + 1) * self.patch_size          # определение новой ширины изображения

            # Дополняем изображение белыми пикселями, если оно не кратно размеру патча.
            padded_img = np.full((new_h, new_w, 3), 255, dtype=np.uint8)    # белый фон
            padded_img[:h, :w] = img_np

            # Превращаем обратно в изображение PIL и добавляем в список.
            padded_pil = Image.fromarray(padded_img)
            padded_images.append(padded_pil)

        return padded_images

    # Метод, принимающий путь к файлу, путь к соответствующей аннотации и метку патчей. Возвращает соответствующие метки патчи этого документа в виде .json без чисто белых патчей.
    def _process_pdf(self, pdf_file: str, annotation_path: str, label: int):
        # Получение названия аннотации соответствующего файла.
        doc_name = os.path.splitext(os.path.basename(annotation_path))[0]
        # Загрузка его аннотации.
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)

        # Получение изображений соответсвующего .pdf документа.
        used_dpi = annotation['dpi']
        pdf_images = self._load_pdf_images(pdf_file, used_dpi)
        result_patches = [] # список для хранения нужных патчей в формате .json

        # Проходимся по всем страницам аннотированного документа.
        for page in annotation["pages"]:
            # Достаем соответствующую патчам страницу документа.
            page_id = page["id"]
            page_image = pdf_images[page_id]

            # Для каждого патча внутри страницы выполняем выборку по метке.
            for patch in page["patches"]:
                if patch["label"] != label:
                    continue

                x, y, width, height = patch["x"], patch["y"], patch["width"], patch["height"]
                patch_img = page_image.crop((x, y, x + width, y + height))

                # Возращение списка.
                result_patches.append({
                    "img": patch_img,
                    "page_id": page_id,
                    "patch": patch,
                    "doc_name": doc_name
                })

        return result_patches

    # Определяет, является ли патч почти полностью белым по заданному порогу (98%).
    def _is_mostly_white_patch(self, patch_img: Image.Image):
        img_np = np.array(patch_img.convert("L"))  # преобразуем в оттенки серого
        white_pixels = np.sum(img_np > 250)        # считаем почти белые пиксели
        total_pixels = img_np.size
        return white_pixels / total_pixels > 0.98

    # Метод, выполняющий сохранение патча в нужный класс выборки.
    def _save_patches(self, patches: list[dict[str, Image.Image | int | dict[str, int] | str]], outputs_path: str, label: int, counter: dict[int, int]):
        # Сохраняем патчи в папку с оригинальными патчами в формате {документ}_страница{...}_патч{...}.jpg
        for i, item in enumerate(patches):
            if counter[label] >= self.max_patches:
                break
            save_path = os.path.join(outputs_path, f"{item['doc_name']}_page{item['page_id']}_patch{i}.jpg")
            # Пропуск чисто белых патчей.
            if not self._is_mostly_white_patch(item["img"]) or label==1:
                item["img"].save(save_path)
            else:
                continue
            
            # Вывод прогресса выполнения.
            counter[label] += 1
            percent = counter[label] / self.max_patches * 100
            print(f"\rСохранение {'оригинальных' if label == 0 else 'поддельных'} патчей...: "
                  f"{percent:.2f}% ({counter[label]}/{self.max_patches})", end="")

    # Метод, отвечающий за обработку оригинальных и поддельных патчей.
    def balance(self):
        classes = ["genuine", "forged"]
        counter = {0: 0, 1: 0}  # 0 — genuine, 1 — forged - счетчик
        forged_patches = []     # список поддельных патчей для аугментации

        print_wait = False  # для вывода, что сбор поддельных патчей идет только 1 раз
        for class_name in classes:
            pdfs_path = os.path.join(self.pdf_dir, class_name)                  # путь к выборке .pdf файлов
            annotations_path = os.path.join(self.annotations_dir, class_name)   # путь к выборке аннотированных файлов

            # Проходимся по всем аннотированным файлам.
            for fname in os.listdir(annotations_path):
                # Обрабатываем только файлы с расширением .json.
                if not fname.endswith(".json"):
                    continue

                annotation_path = os.path.join(annotations_path, fname)         # формируем путь к аннотированному файлу
                doc_name = os.path.splitext(fname)[0]                           # формируем название аннотированного файла

                # Загружаем аннотацию рассматриваемого файла.
                with open(annotation_path, "r", encoding="utf-8") as f:
                    annotation = json.load(f)

                class_id = annotation["class"]                                  # получаем класс, к которому относится аннотированный файл
                pdf_file = os.path.join(pdfs_path, f"{doc_name}.pdf")           # получаем оригинальный документ .pdf

                # Если достигнут лимит патчей для обоих классов — выход.
                if all(counter[l] >= self.max_patches for l in [0, 1]):
                    return

                # Обработка оригинальных патчей.
                if counter[0] < self.max_patches:
                    genuine_patches = self._process_pdf(pdf_file, annotation_path, 0)     # получаем все оригинальные патчи
                    outputs_path = os.path.join(self.output_dir, "genuine")               # путь к датасету выборки оригинальных файлов
                    self._save_patches(genuine_patches, outputs_path, 0, counter)         # сохраняем патчи
                else:
                    if not print_wait:
                        print()
                        print_wait = True
                    print("\r\u001b[38;5;78mСбор поддельных патчей, пожалуйста подождите...\u001b[0m", end="")

                # Обработка поддельных патчей.
                if class_id == 1 and counter[1] < self.max_patches:                    
                    # Собираем поддельные патчи и добавляем их в список.
                    forged = self._process_pdf(pdf_file, annotation_path, 1)
                    for item in forged:
                        forged_patches.append(item)

        # Для каждого прямого поддельного патча в списке после прохождения всего каталога файлов выполняем сохранение.
        print()
        outputs_path = os.path.join(self.output_dir, "forged")          # путь к датасету выборки
        self._save_patches(forged_patches, outputs_path, 1, counter)    # сохранение

        if self.output_dir.endswith("training_set"):
            # Аугментация поддельных патчей !!!обучающей выборки!!!, если их не хватает.
            print()
            if counter[1] < self.max_patches:
                # Сравнение с оригинальными патчами, потому что их может быть меньше.
                genuine_dir = os.path.join(self.output_dir, "genuine")
                num_jpg_files = len([
                    f for f in os.listdir(genuine_dir)
                    if os.path.isfile(os.path.join(genuine_dir, f)) and f.lower().endswith('.jpg')
                ])
                if self.max_patches > num_jpg_files:
                    pf.print_info_information(f"Было удалено оригинальных патчей: {self.max_patches - num_jpg_files}. Число поддельных патчей будет также уменьшено.")
                    self.max_patches = num_jpg_files

                needed = self.max_patches - counter[1]    # сколько еще нужно поддельных патчей
                base_aug = needed // len(forged_patches)  # минимум аугментаций на каждый патч
                extra = needed % len(forged_patches)      # остаток — распределяем по первым патчам
                pf.print_info_information(f" Оригинальные патчи: {self.max_patches}.\n\tНеобходимое число аугментаций подделок: {needed}.")
                for i, item in enumerate(forged_patches):
                    count_aug = 0     # счетчик для имен аугментированных поддельных патчей
                    num_aug = base_aug + (1 if i < extra else 0) # определяем, сколько аугментаций нужно на текущий патч

                    for _ in range(num_aug):
                        if counter[1] >= self.max_patches:
                            break

                        # Генерация цепочки аугментаций изображения и его сохранение.
                        depth = random.randint(1, 3)  # случайная длина цепочки
                        img_aug = self._random_augment_chain(item["img"], depth)
                        save_path = os.path.join(outputs_path, f"{item['doc_name']}_page{item['page_id']}_patch{i}_aug{count_aug}.jpg")
                        img_aug.save(save_path)
                        # Вывод прогрессса выполненеия.
                        counter[1] += 1
                        count_aug += 1
                        percent = counter[1] / self.max_patches * 100
                        print(f"\rАугментация поддельных патчей... Общая статистика: {percent:.2f}% ({counter[1]}/{self.max_patches})", end="")
                print()

        print()
        pf.print_success_information(f"Было сохранено {counter[0]} оригинальных и {counter[1]} поддельных патчей.", end='\n\n')
        pf.print_line()

    

