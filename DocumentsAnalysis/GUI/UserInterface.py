import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QFileDialog, QMessageBox, QScrollArea, QAction
)
from PyQt5.QtCore import Qt, QThreadPool, QRunnable, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage, QResizeEvent

from PIL import Image, ImageDraw
from pdf2image import convert_from_path
import numpy as np

import torch
import torch.nn as nn
import torchvision as tv
from ConsoleOutput.PrintFunctions import print_line, print_info_information

"""================================================================================================>
| Класс, который отвечает за асинхронную загрузку PDF-файлов в виде изображений в отдельном потоке. |
| Имеет два сигнала: окончание успешной загрузки и ошибка при сбое загрузки.                        |
<================================================================================================="""
class PDFLoaderSignals(QObject):
    finished = pyqtSignal(list, str)
    error = pyqtSignal(str)

"""================================================================================================>
| Класс, который представляет собой асинхронный рабочий объект при загрузке PDF.                    |
<================================================================================================="""
class PDFLoader(QRunnable):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.signals = PDFLoaderSignals()

    def run(self):
        dpi_options = [None, 300, 150, 72]
        images = []
        try:
            for dpi in dpi_options:
                try:
                    if dpi is None:
                        images = convert_from_path(self.file_path)
                    else:
                        images = convert_from_path(self.file_path, dpi=dpi)
                    if not images:
                        continue
                    w, _ = images[0].size
                    if w <= 2500:
                        break
                except Exception:
                    continue
            if not images:
                self.signals.error.emit("Не удалось загрузить PDF: файл повреждён или не содержит изображений.")
            else:
                self.signals.finished.emit(images, self.file_path)
        except Exception as e:
            self.signals.error.emit(f"Ошибка при загрузке PDF: {str(e)}")

"""================================================================================================>
| Класс, который отвечает за асинхронный анализ PDF-файлов в виде изображений в отдельном потоке.   |
| Имеет два сигнала: окончание успешного анализа и ошибка при проведении анализа.                   |
<================================================================================================="""
class PDFAnalyzerSignals(QObject):
    finished = pyqtSignal(Image.Image)
    error = pyqtSignal(str)

"""================================================================================================>
| Класс, который представляет собой асинхронный рабочий объект при анализе PDF.                     |
<================================================================================================="""
class PDFAnalyzer(QRunnable):
    def __init__(self, images, patches, model, device, current_page, patch_size):
        super().__init__()
        self.images = images
        self.patches = patches
        self.model = model
        self.device = device
        self.current_page = current_page
        self.patch_size = patch_size
        self.signals = PDFAnalyzerSignals()

    # Подготовка патча для модели.
    def _prepare_patch(self, patch: np.ndarray):
        transform = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        pil_patch = Image.fromarray(patch).convert('RGB')
        t_img = transform(pil_patch)
        return t_img

    # Определяет, является ли патч почти полностью белым по заданному порогу (98%).
    def _is_mostly_white_patch(self, patch: np.ndarray):
        patch_img = Image.fromarray(patch)         # преобразуем np.ndarray в Image
        img_np = np.array(patch_img.convert("L"))  # преобразуем в оттенки серого
        white_pixels = np.sum(img_np > 250)        # считаем почти белые пиксели
        total_pixels = img_np.size
        return white_pixels / total_pixels > 0.98

    def run(self):
        try:
            # Если патчей нет, то выход.
            if not self.patches:
                self.signals.error.emit("Нет патчей для анализа.")
                return

            print_info_information("Обработка изображения по патчам...", end='\n\n')

            # Получение исходного изображения (страницы), с белыми отступами (как при отображении страницы документа сделано).
            page_image = self.images[self.current_page].convert("RGB")        
            img_np = np.array(page_image)
            h, w, _ = img_np.shape
            new_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
            new_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
            padded_img = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
            padded_img[:h, :w] = img_np

            # Создаем изображение PIL и слой для рисования.
            pil_image = Image.fromarray(padded_img).convert("RGBA")             # преобразование в картинку
            overlay = Image.new("RGBA", pil_image.size, (255, 255, 255, 0))     # создание слоя-подложки размером с исходное изображение
            draw = ImageDraw.Draw(overlay, "RGBA")                              # создание объекта для рисования draw

            # Цикл по патчам и анализ модели.
            for idx, patch in enumerate(self.patches):
                # Координаты патча.
                y = (idx // (new_w // self.patch_size)) * self.patch_size
                x = (idx % (new_w // self.patch_size)) * self.patch_size
                # Пропускаем белые патчи.
                if self._is_mostly_white_patch(patch):
                    # Рисуем полупрозрачный зеленый прямоугольник.
                    draw.rectangle([x, y, x + self.patch_size, y + self.patch_size], fill=(0, 255, 0, 100))
                    print(f"Патч: {idx + 1} - белый (оригинал)")
                    continue
                # Анализируем.
                patch_tensor = self._prepare_patch(patch).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(patch_tensor)
                    probs = nn.functional.softmax(output, dim=1)    # вероятность классов
                    prob_class_0 = probs[0][0].item()               # принадлежность классу оригиналов
                    prob_class_1 = probs[0][1].item()               # принадлежность классу подделок
                    print(f"Патч: {idx + 1}, Класс 0: {prob_class_0:.4f}, Класс 1: {prob_class_1:.4f}")

                # Если патч считается с уверенность 85%, то выделяем область.
                if prob_class_1 >= 0.85:
                    # Рисуем полупрозрачный красный прямоугольник.
                    draw.rectangle([x, y, x + self.patch_size, y + self.patch_size], fill=(255, 0, 0, 100))
                if prob_class_0 >= 0.85:
                    # Рисуем полупрозрачный зеленый прямоугольник.
                    draw.rectangle([x, y, x + self.patch_size, y + self.patch_size], fill=(0, 255, 0, 100))
                # Модель неуверена в классе патча.
                if prob_class_0 < 0.85 and prob_class_1 < 0.85:
                    # Рисуем полупрозрачный желтый прямоугольник.
                    draw.rectangle([x, y, x + self.patch_size, y + self.patch_size], fill=(255, 255, 0, 100))

            print()
            print_info_information("Обработка изображения завершена...", end='\n\n')
            print_line()
            # Объединяем оригинал с оверлеем.
            combined = Image.alpha_composite(pil_image, overlay).convert("RGB")

            self.signals.finished.emit(combined)

        except Exception as e:
            self.signals.error.emit(f"Ошибка при анализе PDF: {str(e)}")

"""================================================================================================>
| Класс, предостовляющий пользователю "дружелюьный" интерфейс в виде GUI, который поможет ему       |
| использовать обученную CNN, а также наглядно видеть результат работы.                             |
| Путь к весам жестко задается внутри класса.                                                       |
<================================================================================================="""
class PDFViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Patch Viewer (Documnets Analysis By CNN)") # заголовок окна
        self.resize(1000, 800)                                              # размеры окна

        # --- Переменные состояния ---
        self.images = []            # список изображений страниц документа
        self.current_page = 0       # текущая страница
        self.total_pages = 0        # общее количество страниц документа
        self.patch_size = 224       # размер патча изображения
        self.patches = []           # список патчей страницы для анализа
        self.scale_factor = 1.0     # изначально масштаб 1 (реальный размер)

        self.threadpool = QThreadPool.globalInstance() # запуск пула потоков

        # --- Подготовка модели ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = tv.models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 2)
        )
        weights_path = "./runs/2025-05-04_16-48-27/model_checkpoint_12_epochs.pth"
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.init_ui()

    def init_ui(self):
        # --- Основной виджет и компоновка ---
        self.central_widget = QWidget()                                 # создание виджета
        self.setCentralWidget(self.central_widget)                      # установка виджета как центрального
        self.layout = QVBoxLayout()                                     # создание вертикальной компановки
        self.central_widget.setLayout(self.layout)                      # установка компановки на центр. виджет

        # Установка элементов на центральный виджет.
        # --- Метка для имени файла ---
        self.file_label = QLabel("Файл не выбран")                      # создание метки
        self.file_label.setAlignment(Qt.AlignCenter)                    # центрирование метки
        self.layout.addWidget(self.file_label)                          # установка метки на центр. виджет

        # --- Метка для отображения страницы ---
        self.scroll_area = QScrollArea()                                # создание виджета прокрутки для метки
        self.scroll_area.setWidgetResizable(False)                      # автоматическое изменение размера от содержимого
        self.page_label = QLabel()                                      # создание метки
        self.page_label.setAlignment(Qt.AlignCenter)                    # центрирование метки
        self.page_label.setScaledContents(False)                        # НЕ масштабируем внутри QLabel
        self.scroll_area.setWidget(self.page_label)                     # помещение метки внутрь контейнера
        self.layout.addWidget(self.scroll_area, stretch=1)              # установка метки на центр. виджет с приоритетом
        
        # --- Контейнер для кнопок ---
        self.button_layout = QHBoxLayout()                              # создание горизонтальной компановки

        self.open_button = QPushButton("Загрузить PDF")                 # создание кнопки "Загрузить PDF"
        self.open_button.clicked.connect(self.open_pdf)                 # прикрепление кнопки к функции open_pdf
        self.button_layout.addWidget(self.open_button)                  # добавление кнопки на гор. компановку

        self.analyze_button = QPushButton("Проанализировать PDF")       # создание кнопки "Проанализировать PDF"
        self.analyze_button.clicked.connect(self.analyze_pdf)           # прикрепление кнопки к функции analyze_pdf
        self.analyze_button.setEnabled(False)                           # инактивна вначале перед загрузкой
        self.button_layout.addWidget(self.analyze_button)               # добавление кнопки на гор. компановку

        self.prev_button = QPushButton("Предыдущая страница")           # создание кнопки "Предыдущая страница"
        self.prev_button.clicked.connect(self.prev_page)                # прикрепление кнопки к функции prev_page
        self.prev_button.setEnabled(False)                              # инактивна вначале перед загрузкой
        self.button_layout.addWidget(self.prev_button)                  # добавление кнопки на гор. компановку

        self.next_button = QPushButton("Следующая страница")            # создание кнопки "Следующая страница"
        self.next_button.clicked.connect(self.next_page)                # прикрепление кнопки к функции next_page
        self.next_button.setEnabled(False)                              # инактивна вначале перед загрузкой
        self.button_layout.addWidget(self.next_button)                  # добавление кнопки на гор. компановку

        self.zoom_out_button = QPushButton("Уменьшить масштаб")         # создание кнопки "Уменьшить масштаб"
        self.zoom_out_button.clicked.connect(self.zoom_out)             # прикрепление кнопки к функции zoom_out
        self.zoom_out_button.setEnabled(False)                          # инактивна вначале перед загрузкой
        self.button_layout.addWidget(self.zoom_out_button)              # добавление кнопки на гор. компановку

        self.zoom_in_button = QPushButton("Увеличить масштаб")          # создание кнопки "Увеличить масштаб"
        self.zoom_in_button.clicked.connect(self.zoom_in)               # прикрепление кнопки к функции zoom_in
        self.zoom_in_button.setEnabled(False)                           # инактивна вначале перед загрузкой
        self.button_layout.addWidget(self.zoom_in_button)               # добавление кнопки на гор. компановку

        self.layout.addLayout(self.button_layout)                       # добавление гор. компановки на центр. виджет

        # --- Меню "Помощь" со справкой ---
        help_action = QAction("Справка", self)
        help_action.triggered.connect(self.show_help_dialog)  # подключаем слот

        menubar = self.menuBar()  # главное меню окна
        help_menu = menubar.addMenu("Помощь")
        help_menu.addAction(help_action)

    # Метод отображения справки.
    def show_help_dialog(self):
        QMessageBox.information(
            self,
            "Справка",
            "Приложение является вспомогательным к анализу и не уверяет на 100%, что оно самое умное :C\n\n"
            "При нажатии кнопки 'Проанализировать PDF' пройдёт некоторое время, пока модель анализирует изображение.\n"
            "Цвета: \n"
            " * Зелёный - оригинальный фрагмент изображения, можно доверять (>=90% оригинал);\n"
            " * Красный - поддельный фрагмент изображения, нельзя доверять (>=90% подделка);\n"
            " * Желтый - модель не уверена, что она увидела, может подделка, может оригинал. Можно не обращать внимания на данные фрагменты, лучше проглядеть глазами.\n\n"
            "При загрузке PDF документа также может пройти некоторое время, особенно если он в высоком dpi. Приложение адаптирует документы так, чтобы рассматриваемые "
            "фрагменты (патчи 224х224) моделью содержали достаточное количество информации.\n\n"
            "Все остальные кнопки интуитивно понятны и не требуют рассмотрения."
        )

    # Метод открывает .pdf файл в нормальном разрешении для отображения и по страницам.
    def open_pdf(self):
        # Окно, чтобы выбрать .pdf файл.
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите PDF-файл",
            "",
            "PDF Files (*.pdf)",
            options=options
        )
        # Пользователь отменил выбор.
        if not file_path:
            return 

        # Операция по загрузке постраничных изображений PDF.
        self.open_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        loader = PDFLoader(file_path)
        loader.signals.finished.connect(self.handle_pdf_loaded)
        loader.signals.error.connect(self.handle_pdf_error)
        self.threadpool.start(loader)

    # Метод обновления состояния переменных.
    def handle_pdf_loaded(self, images: list, file_path: str):
        self.images = images
        self.total_pages = len(images)
        self.current_page = 0
        self.file_label.setText(f"Файл: {file_path.split('/')[-1]} ({self.current_page + 1}/{self.total_pages} стр.)")
        self.display_page()
        self.open_button.setEnabled(True)
        self.analyze_button.setEnabled(True)

    # Метод вывода ошибки.
    def handle_pdf_error(self, message: str):
        QMessageBox.critical(self, "Ошибка", message)
        self.open_button.setEnabled(True)
        self.analyze_button.setEnabled(True)

    # Метод обновления пиксмапа внутри метки отображения страниц.
    def update_pixmap_to_label(self):
        # Масштабирование страницы.
        if hasattr(self, 'original_pixmap'):
            scaled_pixmap = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.page_label.setPixmap(scaled_pixmap)
            self.page_label.adjustSize() # размер QLabel теперь будет как у pixmap

    # Изменение масштаба изображения.
    def zoom_change(self, old_width, old_height):
        # Получаем текущие параметры.
        h_bar = self.scroll_area.horizontalScrollBar()
        v_bar = self.scroll_area.verticalScrollBar()

        # Центр текущей области просмотра.
        center_x = h_bar.value() + self.scroll_area.viewport().width() // 2
        center_y = v_bar.value() + self.scroll_area.viewport().height() // 2

        # Центр в относительных координатах от QLabel (до масштабирования).
        rel_x = center_x / old_width if old_width > 0 else 0.5
        rel_y = center_y / old_height if old_height > 0 else 0.5

        # Новая величина изображения.
        new_width = self.page_label.width()
        new_height = self.page_label.height()

        # Восстановление центра после масштабирования.
        new_center_x = int(new_width * rel_x)
        new_center_y = int(new_height * rel_y)

        h_bar.setValue(new_center_x - self.scroll_area.viewport().width() // 2)
        v_bar.setValue(new_center_y - self.scroll_area.viewport().height() // 2)

    def zoom_in(self):
        old_width = self.page_label.width()
        old_height = self.page_label.height()
        self.scale_factor *= 1.1
        if self.scale_factor > 5:
            self.scale_factor = 5
        self.update_pixmap_to_label()
        self.zoom_change(old_width, old_height)

    def zoom_out(self):
        old_width = self.page_label.width()
        old_height = self.page_label.height()
        self.scale_factor *= 0.9
        if self.scale_factor < 0.1:
            self.scale_factor = 0.1
        self.update_pixmap_to_label()
        self.zoom_change(old_width, old_height)

    # Метод, который отображает страницу на экране.
    def display_page(self):
        # Если список изображений страниц пуст - выход.
        if not self.images:
            return

        page_image = self.images[self.current_page].convert("RGB")
        # Предобработка изображения, чтобы не было черных полей при выходе за пределы исходного изображения.
        img_np = np.array(page_image)                                   # представление изображение в виде нормальных пикселей
        # Рассчитываем, сколько дополнительных пикселей нужно добавить.
        h, w, _ = img_np.shape                                          # представление (H, W, C), где С - каналы, H - высота, W - ширина
        new_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size # определение новой высоты изображения
        new_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size # определение новой ширины изображения      
        # Дополняем изображение белыми пикселями, если оно не кратно размеру патча.
        padded_img = np.full((new_h, new_w, 3), 255, dtype=np.uint8)    # белый фон
        padded_img[:h, :w] = img_np         
        
        # Масштабирование под размер QLabel.
        qimage = QImage(padded_img.data, padded_img.shape[1], padded_img.shape[0], padded_img.strides[0], QImage.Format_RGB888) # создание объекта QImage из numpy массива
        pixmap = QPixmap.fromImage(qimage)                                                                                      # преобразование QImage в QPixmap, который можно отобразить в QLabel
        self.original_pixmap = pixmap   # сохраняем оригинал для масштабирования
        self.update_pixmap_to_label()

        # Формирование списка патчей.
        self.patches = []
        for y in range(0, new_h, self.patch_size):
            for x in range(0, new_w, self.patch_size):
                patch = padded_img[y:y+self.patch_size, x:x+self.patch_size]
                self.patches.append(patch)

        # Обновление состояния кнопок.
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(self.current_page < self.total_pages - 1)
        self.analyze_button.setEnabled(bool(self.patches))
        self.zoom_in_button.setEnabled(True)
        self.zoom_out_button.setEnabled(True)
    
    # Сохранение логики и пропорций окна при изменении размера.
    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.update_pixmap_to_label()

    # Метод перехода к следующей странице.
    def next_page(self):
        if self.images and self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.file_label.setText(
                f"{self.file_label.text().split(' (')[0]} ({self.current_page + 1}/{self.total_pages} стр.)"
            )
            self.display_page()

    # Метод перехода к предыдущей странице.
    def prev_page(self):
        if self.images and self.current_page > 0:
            self.current_page -= 1
            self.file_label.setText(
                f"{self.file_label.text().split(' (')[0]} ({self.current_page + 1}/{self.total_pages} стр.)"
            )
            self.display_page()

    # Прогон изображения через модель и отрисовка поддельных областей.
    def analyze_pdf(self):
        self.analyze_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        analyzer = PDFAnalyzer(
            images=self.images,                     # набор изображений исходного документа (не дополненный до кратного патчу размера)
            patches=self.patches,                   # набор патчей
            model=self.model,                       # CNN для анализа
            device=self.device,                     # устройство вычислений
            current_page=self.current_page,         # номер текущей страницы
            patch_size=self.patch_size              # размер патча
        )
        analyzer.signals.finished.connect(self.show_analyzed_image)
        analyzer.signals.error.connect(self.show_error_message)
        self.threadpool.start(analyzer)

    # Метод отображения проанализированной страницы.
    def show_analyzed_image(self, image: Image.Image):
        # Конвертируем в QImage и показываем через QLabel (операции аналогичные отрисовки страницы).
        image_data = image.convert("RGB").tobytes()
        qimage = QImage(image_data, image.width, image.height, image.width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.original_pixmap = pixmap
        self.update_pixmap_to_label()
        self.analyze_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.next_button.setEnabled(True)
        self.prev_button.setEnabled(True)

    # Метод вывода ошибки.
    def show_error_message(self, msg):
        QMessageBox.critical(self, "Ошибка анализа", msg)
        self.analyze_button.setEnabled(True)
        self.open_button.setEnabled(True)

# Функция запуска интерфейса пользователя.
def run_gui():
    # ============================= Внутренняя настройка окружения Qt =============================
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    plugin_path = os.path.join(project_root, 'mymlenv', 'lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    # ==============================================================================================
    app = QApplication(sys.argv)    # создание объекта QApplication — это основной объект PyQt5-программы, который управляет событиями окна, взаимодействием с пользователем и всем GUI
    viewer = PDFViewer()            # создание экземпляра нашего интерфейса
    viewer.show()                   # запуск GUI для отображения
    app.exec_()                     # запуск главного цикла событий GUI и закрытие окна при выходе