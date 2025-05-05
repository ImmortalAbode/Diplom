import os

from .PDFExtract import PDFPatchExtractor
import ConsoleOutput.PrintFunctions as pf

"""=====================================================================================================> 
| Класс, формирующий каталог с аннотированными данными исходных .pdf документов по патчам для           |
| дальнейшего их редактирования и балансировки. Конструктор выполняет очистку имеющегося каталога с     |
| данными, а метод выполняет сохранение файлов с необходимой разметкой.                                 |
<====================================================================================================="""
class PatchDataset:
    def __init__(self, annotation_path: str, patch_size: int):
        # Создание папки с файлами разметки, настройка размера батча и названия папки.
        self.annotation_path = annotation_path  # путь к папке сохранения аннотаций
        self.patch_size = patch_size    # размер патча
        self.extractor = PDFPatchExtractor(patch_size, annotation_path)

        self.subsets = ["training_set", "validation_set", "test_set"]    # подкаталоги датасета с данными
        self.classes = [("genuine", 0), ("forged", 1)]                   # подкаталоги подкаталогов датасета с файлами по классам
        
    # Метод выполняющий разметку и сохранение аннотированных файлов по всем подкаталогам.
    def get_annotation(self, dataset_path: str):
        pf.print_result_information("Производится аннотирование всех подкаталогов...")
        # Для каждого подкаталога датасета.
        for subset in self.subsets:
            # Для каждого подкаталога подкаталогов датасета.
            for class_name, class_id in self.classes:
                folder_path = os.path.join(dataset_path, subset, class_name)    # формируем полный путь до файлов
                ann_dir = os.path.join(self.annotation_path, subset, class_name)     # формируем полный путь до аннотированных файлов

                os.makedirs(ann_dir, exist_ok=True) # создаем папку в директории проекта с аннотациями

                pdf_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".pdf")]  # список файлов соответствующей группы

                # Для каждого найденного файла.
                percent = 0
                for idx, filename in enumerate(pdf_files):
                    pdf_path = os.path.join(folder_path, filename)      # формируем полный путь к файлу
                    ann = self.extractor.extract(pdf_path, class_id)    # получаем разметку конкретного документа
                    # Пропуск поврежденных файлов.
                    if ann is None:
                        continue

                    percent = int((idx + 1) / len(pdf_files) * 100)
                    print(f"\r[{subset}/{class_name}] Аннотирование: {percent}% ({idx + 1}/{len(pdf_files)})", end="", flush=True)
                if percent == 100:
                    print()