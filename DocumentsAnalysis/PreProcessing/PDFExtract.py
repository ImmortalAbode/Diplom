import os
import numpy as np
import pdf2image as p2i
import json
import ConsoleOutput.PrintFunctions as pf
"""===========================================================================================> 
| Класс - генератор аннотаций. Превращает PDF-документ в набор патчей фиксированного размера, |
| а также сохраняет разметку документа в JSON формате.                                        |
| Является препроцессором PDF-документов для обучения.                                        |
<==========================================================================================="""
class PDFPatchExtractor:
    def __init__(self, patch_size: int, annotation_path: str):
        self.patch_size = patch_size            # размер патча
        self.annotation_path = annotation_path    # название папки
        # Создание нужных для аннотирования папок, а также их очистка в случае наличия в них файлов.
        os.makedirs(annotation_path, exist_ok=True)
        self._clear_dataset_subfolders(self.annotation_path)

    # Метод, который выполняет очистку папки аннотаций от существующих файлов.
    def _clear_dataset_subfolders(self, annotation_path: str):
        required_sets = ['training_set', 'validation_set', 'test_set'] 
        subfolders = ['genuine', 'forged']                              

        # Спуск вниз к подпапкам.
        for dataset_name in required_sets:
            dataset_path = os.path.join(annotation_path, dataset_name)
            os.makedirs(dataset_path, exist_ok=True)

            # Спуск вниз к папкам подпапок.
            for subfolder in subfolders:
                subfolder_path = os.path.join(dataset_path, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)

                # Удаляем файлы в папке.
                for filename in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

    # Пытаемся конвертировать PDF с разным DPI, чтобы добиться ширины <= 2500 пикселей - равномерность информации патча для обучения. Ширина - просто показатель, так проще.
    def _convert_with_dpi(self, pdf_path):
        dpi_list=[None, 300, 150, 72]
        for dpi in dpi_list:
            # Отлавливает файлы, которые повреждены и не подлежат открытию, тем более конвертации.
            try:
                if dpi is None:
                    pages = p2i.convert_from_path(pdf_path)
                    current_dpi = None
                else:
                    pages = p2i.convert_from_path(pdf_path, dpi=dpi)
                    current_dpi = dpi
            except Exception as e:
                print('\n')
                pf.print_line()
                error_lines = str(e).splitlines()  # разбиение текста ошибки на строки
                indented_errors = "\n".join(f"\t{line}" for line in error_lines)
                pf.print_critical_information(f"\n\tФайл повреждён и не будет обработан: {pdf_path}\n\tПричины:\n{indented_errors}", end='\n\n')
                try:
                    os.remove(pdf_path)
                    pf.print_info_information(f"\tФайл удалён: {pdf_path}.", end='\n\n')
                except Exception as delete_error:
                    pf.print_warning_information(f"Не удалось удалить файл: {delete_error}.")
                pf.print_line()
                return None, None

            # Проверяем размер первой страницы.
            w, _ = pages[0].size
            if w <= 2500:
                return pages, current_dpi

        # Если ни один dpi не подошел, возвращаем последние страницы
        return pages, current_dpi

    # Метод, создающий необходимую структуру JSON формата и возвращающий сформированную аннотацию.
    def _generate_annotation(self, pdf_path: str, class_id: int):
        pages, used_dpi = self._convert_with_dpi(pdf_path) # конвертируем .pdf в набор изображений по страницам в виде списка page_image
        if pages==None and used_dpi==None:
            return None

        # Структура аннотации документа.
        annotation = {
            "document": os.path.basename(pdf_path),
            "class": class_id,
            "dpi": used_dpi,
            "pages": []
        }

        # Проходимся по всем страницам документа.
        for page_id, page_image in enumerate(pages):
            # Предобработка изображения.
            page_image = page_image.convert("RGB")                          # открытие в RGB
            img_np = np.array(page_image)                                   # представление изображение в виде нормальных пикселей

            # Рассчитываем, сколько дополнительных пикселей нужно добавить.
            h, w, _ = img_np.shape                                          # представление (H, W, C), где С - каналы, H - высота, W - ширина
            new_h = ((h // self.patch_size) + 1) * self.patch_size          # определение новой высоты изображения
            new_w = ((w // self.patch_size) + 1) * self.patch_size          # определение новой ширины изображения

            # Структура аннотации страницы.
            page_annotation = {
                "id": page_id, 
                "patches": []
            }
            
            # Проходимся по всему новому изображению по патчам (224х224).
            for i in range(0, new_h, self.patch_size):
                for j in range(0, new_w, self.patch_size):
                    # Структура аннотации патча страницы.
                    patch_info = {
                        "x": j,
                        "y": i,
                        "width": self.patch_size,
                        "height": self.patch_size,
                        "label": 0                      # по умолчанию считаем, что любой патч - оригинал, затем сами уже исправляем
                    }
                    # Формируем аннотацию страницы.
                    page_annotation["patches"].append(patch_info)

            # Формируем аннотацию документа.
            annotation["pages"].append(page_annotation)

        return annotation

    # Метод, который создает файл базовой разметки. Возвращает сформированную разметку.
    def extract(self, pdf_path: str, class_id: int):
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]              # формируем имя файла разметки документа без расширения

        # Получаем два уровня вверх от файла — например: training_set/genuine
        two_levels_up = os.path.join(
            os.path.basename(os.path.dirname(os.path.dirname(pdf_path))),  # training_set
            os.path.basename(os.path.dirname(pdf_path))                    # genuine
        )

        json_path = os.path.join(self.annotation_path, two_levels_up, f"{doc_name}.json")       # формируем полный путь к файлу .json

        # Создаем сам файл разметки и сохраняем по пути json_path.
        annotation = self._generate_annotation(pdf_path, class_id)
        # Если аннотация не была создана (например, PDF повреждён)
        if annotation is None:
            return None
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=4)  # параметры - что записать, куда записать, отступы для улучшения читаемости

        return annotation


"""
*----------------------------*
| ПРИМЕР АННОТАЦИИ ДОКУМЕНТА |
*----------------------------*

{
    "document": "doc1.pdf",
    "class": 0,
    "dpi": 150,
    "pages": [
        {
            "id": 0,
            "patches": [
                {
                    "x": 0,
                    "y": 0,
                    "width": 224,
                    "height": 224,
                    "label": 0
                },
                {
                    "x": 224,
                    "y": 0,
                    "width": 224,
                    "height": 224,
                    "label": 0
                }
                ...
            ]
        }
        ...
    ]
}

"""