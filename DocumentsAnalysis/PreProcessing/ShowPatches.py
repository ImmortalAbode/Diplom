import os
import json
import matplotlib
matplotlib.use('TkAgg') # родной интерфейс tkinter
import matplotlib.pyplot as plt
import pdf2image as p2i
import numpy as np

import ConsoleOutput.PrintFunctions as pf

"""=========================================================================================> 
|  Класс, показывавающий конкретные патчи (в диапазоне, одного документа, порциями).        |
|  Необходим для ручной отладки разметки документов, корректировки аннотирования            |
<========================================================================================="""
class PatchVisualizer:
    def __init__(self, pdf_path: str, annotation_path: str, patch_size: int):
        self.annotation_path = annotation_path  # путь к .json файлу
        self.pdf_path = pdf_path                  # путь к .pdf файлу
        self.patch_size = patch_size              # размер патча

    # Метод для визуализации всех патчей, соответствующих одному конкретному PDF-документу.
    def show_patches_interactive(self, ncols: int=4, nrows: int=4):
        # Загружаем аннотацию из полного пути.
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        pages_annotation = annotation['pages']                      # список страниц из аннотации
        used_dpi = annotation['dpi']
        # загружаем страницы PDF как изображения
        if used_dpi==None:
            pdf_images = p2i.convert_from_path(self.pdf_path)
        else:
            pdf_images = p2i.convert_from_path(self.pdf_path, dpi=used_dpi)

        # Для каждой страницы документа выводим патчи по отдельности.
        for page in pages_annotation :
            page_id = page['id']
            patches = page['patches']
            page_image = pdf_images[page_id].convert("RGB") # Получаем изображение страницы в формате RGB.

            # Предобработка изображения, чтобы не было черных полей при выходе за пределы исходного изображения.
            img_np = np.array(page_image)                                   # представление изображение в виде нормальных пикселей
            # Рассчитываем, сколько дополнительных пикселей нужно добавить.
            h, w, _ = img_np.shape                                          # представление (H, W, C), где С - каналы, H - высота, W - ширина
            new_h = ((h // self.patch_size) + 1) * self.patch_size          # определение новой высоты изображения
            new_w = ((w // self.patch_size) + 1) * self.patch_size          # определение новой ширины изображения
            # Дополняем изображение белыми пикселями, если оно не кратно размеру патча.
            padded_img = np.full((new_h, new_w, 3), 255, dtype=np.uint8)    # белый фон
            padded_img[:h, :w] = img_np           
            
            # Разделяем патчи на группы по 16.
            patches_per_page = ncols * nrows
            num_pages = (len(patches) + patches_per_page - 1) // patches_per_page  # вычисление количества страниц полотна для отображения патчей

            # Запрос у пользователя о необходимости показа страницы документа по патчам.
            pf.print_info_information(f"Страница {page_id}, патчей: {len(patches)}.")
            user_input = ""
            while True:
                user_input = input("Показать текущую страницу (y/n) или прекратить показ вовсе (skip): ").strip().lower()
                if user_input in ("y", "n", "skip"):
                    break
                else:
                    pf.print_warning_information("Ошибка ввода. Введите 'y', 'n' или 'skip'!")
            if user_input == "n":
                pf.print_result_information(f"Страница {page_id} будет пропущена.", end='\n\n')
                continue
            elif user_input == "skip":
                pf.print_result_information("\nПрерывание показа страниц по патчам.", end='\n\n')
                break
            print()

            # Для каждой группы патчей (по 16) создаем новый рисунок.
            for page_num in range(num_pages):
                plt.figure(figsize=(ncols * 3, nrows * 3))  # создание полотна для визуализации, * 3 - масштаб визуализации в дюймах

                # Добавляем название страницы документа, чьи патчи изображены (например, page_1, page_2, и т.д.).
                pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]  # имя документа без расширения
                plt.suptitle(f"{pdf_name} page {page_id}", fontsize=16)

                # Берем только часть патчей для текущей страницы.
                start_idx = page_num * patches_per_page                         # начальный индекс
                end_idx = min((page_num + 1) * patches_per_page, len(patches))  # конечный индекс
                patch_subset = patches[start_idx:end_idx]                       # выборка патчей

                # Отображаем патчи на текущем полотне.
                for i, patch_info in enumerate(patch_subset):
                    # Получаем информацию о патче.
                    x, y = patch_info['x'], patch_info['y']
                    width, height = patch_info['width'], patch_info['height']
                    label = patch_info['label']

                    # Вырезаем патч из изображения.
                    patch = padded_img[y:y + height, x:x + width]

                    # Отображаем патч в сетке.
                    plt.subplot(nrows, ncols, i + 1)                            # Настройка сетки 4x4
                    plt.imshow(patch)
                    plt.title(f"Label: {label}\nx: {x}, y: {y}", fontsize=10)

                plt.tight_layout()                                              # выравнивание элементов полотна
                plt.subplots_adjust(top=0.9, hspace=0.6, wspace=0.4)            # оставляем место для названия, корректировка расстояний
                plt.show()                                                      # выводим полотно
