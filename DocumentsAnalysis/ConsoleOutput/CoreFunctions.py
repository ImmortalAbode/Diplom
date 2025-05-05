from tabnanny import check
from urllib import request
import ConsoleOutput.PrintFunctions as pf
import PreProcessing as pp
from ModelLearning.ModelTrainer import CNNTrainer
from ModelLearning.config import config as CONFIG 

import os
import json

"""===============================================================================*
| В данном модуле находится набор функций для корректного взаимодействия с        |
| программой, а также разичные проверки возможных ошибок.                         |
*==============================================================================="""

"""ФУНКЦИИ ПРОВЕРОК"""
# Функция проверки пути к каталогу.
def get_valid_path(prompt: str):
    while True:
        path = input(prompt).strip()
        if os.path.exists(path) or path == "exit":
            return path
        else:
            pf.print_warning_information("Путь не существует. Попробуйте снова.")

# Функция проверки корректной структуры датасета.
def is_dataset_structure_valid(dataset_path: str):
    required_subdirs = [
        os.path.join(dataset_path, "training_set", "genuine"),
        os.path.join(dataset_path, "training_set", "forged"),
        os.path.join(dataset_path, "validation_set", "genuine"),
        os.path.join(dataset_path, "validation_set", "forged"),
        os.path.join(dataset_path, "test_set", "genuine"),
        os.path.join(dataset_path, "test_set", "forged"),
    ]

    for path in required_subdirs:
        if not os.path.isdir(path):
            pf.print_warning_information(f"Не найдена необходимая папка: {path}")
            return False
    pf.print_result_information(f"Структура папки {dataset_path} корректна!")
    return True

"""ПЕРВАЯ ОПЦИЯ"""
# Первая опция разработчика - аннотирование документов.
def dev_opt1_execute():
    # Информация об опции.
    pf.print_dev_opt1_info()
    pf.print_line(end='\n')
    print("\u001b[38;5;105m=== \u001b[38;5;104mРежим разработчика: Опция 1 \u001b[38;5;105m===\u001b[0m".center(125))
    pf.print_line()
    
    # Получение необходимых путей.
    while True:
        dataset_path = get_valid_path("Введите путь к вашему датасету .pdf файлов разбитых на выборки ([exit] - выход): ")
        if dataset_path == "exit":
            pf.print_info_information("Выход из опции.", end='\n\n')
            return
        if is_dataset_structure_valid(dataset_path):
            break
    pf.print_info_information("Папка с аннотациями будет автоматически создана внутри проекта с именем './annotations/'", end='\n\n')
    annotations_path = "./annotations"

    # Обработка попытки перезаписать уже существующий файл разметки.
    if os.path.exists(annotations_path):
        pf.print_line()
        pf.print_critical_information(f"\n\tПапка аннотаций уже существует: {annotations_path}"
                                     "\n\tЕсли в этой папке есть ручная разметка — она может быть перезаписана!", end='\n\n')
        
        while True:
            confirm = input("Продолжить и перезаписать существующую папку? Это может привести к потере данных. [y/n]: ").strip().lower()
            if confirm in ("y", "n"):
                break
            else:
                pf.print_warning_information("Неверный ввод. Введите 'y' (да) или 'n' (нет).", end='\n\n')

        if confirm == "n":
            print()
            pf.print_critical_information("Операция прервана пользователем. Данные не будут изменены.", end='\n\n')
            return

    # Получение размера патча.
    pf.print_info_information("Размер патча будет выбран автоматически с размером 224х224 (требование ResNet50).", end='\n\n')
    patch_size = 224
    
    # Вывод всей полученной информации.
    pf.print_line()
    print("Ввод завершён. Полученные значения:")
    print(f"Датасет данных:     \u001b[38;5;46m{dataset_path}\u001b[0m")
    print(f"Папка аннотаций:    \u001b[38;5;46m{annotations_path}\u001b[0m")
    print(f"Размер патча:       \u001b[38;5;46m{patch_size}\u001b[0m")
    print()
    pf.print_line()

    # Аннотирование .pdf документов. При этом будет вызван PDFPatchExtractor и созданы аннотации в папке annotations_folder.
    pp.PatchDataset(annotations_path, patch_size).get_annotation(dataset_path)
    pf.print_result_information("Аннотирование завершено.")

    pf.print_success_information("Операция выполнена!", end='\n\n')

"""ВТОРАЯ ОПЦИЯ"""
# Вторая опция разработчика - редактирование аннотированных документов.
def dev_opt2_execute():
    # Информация об опции.
    pf.print_dev_opt2_info()
    pf.print_line(end='\n')
    print("\u001b[38;5;105m=== \u001b[38;5;104mРежим разработчика: Опция 2 \u001b[38;5;105m===\u001b[0m".center(125))
    pf.print_line()
    
    # Получение необходимых путей.
    pdf_path, annotation_path = "", ""
    while True:
           # Получение полного пути к PDF-файлу.
           pdf_path = get_valid_path("Введите полный путь к PDF-файлу, который хотите анализировать ([exit] - выход): ").strip()
           if pdf_path == "exit":
               pf.print_info_information("Выход из опции.", end='\n\n')
               return
           # Проверка, что это действительно PDF файл.
           if not pdf_path.lower().endswith('.pdf'):
               pf.print_warning_information(f"{pdf_path} не является PDF-файлом.")
               continue

           # Формируем имя файла без расширения для поиска аннотации.
           doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
           # Формируем путь к аннотации, основываясь на структуре каталогов - например: training_set/genuine.
           two_levels_up = os.path.join(
               os.path.basename(os.path.dirname(os.path.dirname(pdf_path))),  # например, 'training_set'
               os.path.basename(os.path.dirname(pdf_path))                    # например, 'genuine'
           )

           annotation_folder = "annotations"
           annotation_path = os.path.join(annotation_folder, two_levels_up, f"{doc_name}.json")

           # Проверка наличия PDF-файла и аннотации.
           if not os.path.isfile(pdf_path):
               pf.print_critical_information(f"PDF-файл '{pdf_path}' не найден.")
           elif not os.path.isfile(annotation_path):
               pf.print_critical_information(f"Аннотация '{doc_name}.json' не найдена в папке './{annotation_folder}/'.")
           else:
               # Если все в порядке, можно продолжить с файлом и аннотацией.
               print()
               pf.print_line()
               pf.print_result_information(f"Аннотация для '{doc_name}.pdf' найдена: {annotation_path}")
               break 

    # Получение размера патча.
    pf.print_info_information("Размер патча будет выбран автоматически с размером 224х224 (требование ResNet50).", end='\n\n')
    patch_size = 224
    # Вывод всей полученной информации.
    pf.print_line()
    print("Ввод завершён. Полученные значения:")
    print(f"Анализируемый файл:     \u001b[38;5;46m{pdf_path}\u001b[0m")
    print(f"Файл аннотаций:         \u001b[38;5;46m./{annotation_path}\u001b[0m")
    print(f"Размер патча:           \u001b[38;5;46m{patch_size}\u001b[0m")
    print()
    pf.print_line()
    # Создание визуализатора и запуск отображения.
    
    visualizer = pp.PatchVisualizer(pdf_path, annotation_path, patch_size)
    visualizer.show_patches_interactive()
    pf.print_info_information("Визуализация патчей завершена.")
    pf.print_info_information("Рекомендуется сохранить изменения, чтобы они вступили в силу.")

    pf.print_success_information("Операция выполнена!", end='\n\n')

"""ТРЕТЬЯ ОПЦИЯ"""
# Получение списка имен файлов соответствующей директории.
def get_base_names(directory: str, extension: str):
    return {os.path.splitext(f)[0] for f in os.listdir(directory) if f.lower().endswith(extension)}

# Проверка, что все .pdf в папке имеют аннотацию .json в папке.
def validate_pdf_annotation(pdf_path: str, annotations_path: str):
    pdf_basenames = get_base_names(pdf_path, ".pdf")                    # список имен .pdf файлов
    annotation_basenames = get_base_names(annotations_path, ".json")    # список имен .json файлов

    missing = pdf_basenames - annotation_basenames  # получение неаннотированных имен файлов
    # Сообщение информации пользователю.
    if not missing:
        return True
    pf.print_critical_information(f"\n\tДля следующих файлов отсутствуют аннотации:")
    for name in sorted(missing):
        print(f"\n\t  - {name}.pdf")
    return False

# Подсчет количества оригинальных патчей.
def count_genuine_patches(annotations_dir: str):
    classes = ["genuine", "forged"] # классы документов
    genuine_count = 0   # число оригинальных патчей

    # Подсчет соответствующих патчей во всех документах training_set.
    for class_name in classes:
        annotation_path = os.path.join(annotations_dir, class_name)
        for filename in os.listdir(annotation_path):
            if filename.endswith(".json"):
                filepath = os.path.join(annotation_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for page in data["pages"]:
                        for patch in page["patches"]:
                            label = patch["label"]
                            if label == 0:
                                genuine_count += 1
    return genuine_count

# Получения числа максимального количества патчей в каждом классе.
def get_max_patch_counts(annotations_dir: str):
    # Получение фактического количества оригинальных патчей.
    genuine_total = count_genuine_patches(annotations_dir)
    pf.print_info_information(f"Доступно оригинальных патчей: {genuine_total} в классе датасета {annotations_dir}.", end='\n')

    while True:
        user_input = input(f"Введите максимальное количество патчей (макс. {genuine_total}): ")
        if not user_input.isdigit():
            pf.print_warning_information("Введите неотрицательное целое число.")
            continue
        max_patches = int(user_input)
        if max_patches > genuine_total:
            pf.print_warning_information(f"Заданное число превышает доступное количество ({genuine_total}).")
        else:
            return max_patches

# Проверка состояния файлов в папке на наличие не .pdf файлов.
def only_pdfs_in_folder(folder_path):
    classes = ["forged", "genuine"]
    for class_name in classes:
        class_path = os.path.join(folder_path, class_name)
        for filename in os.listdir(class_path):
            if not filename.lower().endswith('.pdf'):
                pf.print_warning_information(f"Найден не PDF: {filename}. Операция прервана.\n")
                return False
    return True

# Третья опция разработчика - формирование и балансировка датасета.
def dev_opt3_execute():
    # Информация об опции.
    pf.print_dev_opt3_info()
    pf.print_line(end='\n')
    print("\u001b[38;5;105m=== \u001b[38;5;104mРежим разработчика: Опция 3 \u001b[38;5;105m===\u001b[0m".center(125))
    pf.print_line()
    
    # Получение необходимых путей.
    dataset_path = ""
    while True:
        dataset_path = get_valid_path("Введите путь к вашему датасету .pdf файлов разбитых на выборки ([exit] - выход): ")
        if dataset_path == "exit":
            pf.print_info_information("Выход из опции.", end='\n\n')
            return
        if is_dataset_structure_valid(dataset_path):
            break
    annotations_path = "./annotations"
    if not is_dataset_structure_valid(annotations_path):
        pf.print_warning_information("Проверьте корректность директории аннотаций в проекте и повторите опцию.", end='\n\n')
        return
    # Получение размера патча.
    pf.print_info_information("Размер патча будет выбран автоматически с размером 224х224 (требование ResNet50).", end='\n\n')
    patch_size = 224
    pf.print_line()

    subsets = ["training_set", "validation_set", "test_set"]    # подпапки итогового датасета

    # Проверка, что в папке dataset_path корректные файлы.
    for subset in subsets:
        dataset_subset = os.path.join(dataset_path, subset) # подкаталог папки dataset_path
        if not only_pdfs_in_folder(dataset_subset):
            return

    # Для каждой выборки датасета.
    output_path = "./dataset_patches"   # папка итогового датасета
    os.makedirs(output_path, exist_ok=True)
    for subset in subsets:
        # Путь к датасету патчей соответствующей выборки.
        output_subset_path = os.path.join(output_path, subset)
        os.makedirs(output_subset_path, exist_ok=True)
        # Путь к датасету .pdf документов соответствующей выборки.
        dataset_subset_path = os.path.join(dataset_path, subset)
        # Путь к аннотации .pdf документов соответствующей выборки.
        annotation_subset_path = os.path.join(annotations_path, subset)
        # Получаем максимальное число оригинальных патчей для каждой выборки.
        max_patches = get_max_patch_counts(annotation_subset_path)
        
        pf.print_info_information(f"Обработка {dataset_subset_path}...")
        # Запуск и баланстровка датасета патчей.
        balancer = pp.DatasetBalancer(dataset_subset_path, annotation_subset_path, output_subset_path, patch_size, max_patches)
        balancer.balance()

    pf.print_result_information("Формирование и балансировка датасета завершена.")

    pf.print_success_information("Операция выполнена!", end='\n\n')

"""ЧЕТВЕРТАЯ ОПЦИЯ"""
# Проверка наличия обученных запусков модели.
def check_runs_dir(runs_path: str):
    if not os.path.exists(runs_path):
        return []

    found_runs = {}  # словарь с подпапками и последними эпохами

    # Проходим по всем папкам и файлам в директории
    for root, dirs, _ in os.walk(runs_path):
        need_line = False
        # Для каждого подкаталога в dirs.
        for dir_name in dirs:
            # Путь к подкаталогу.
            dir_path = os.path.join(root, dir_name)
            last_epoch = -1  # переменная для отслеживания последней эпохи в подкаталоге

            # Получаем список файлов в этом подкаталоге.
            for file in os.listdir(dir_path):
                if file.startswith("model_checkpoint_") and file.endswith("_epochs.pth"):
                    try:
                        # Извлекаем число эпох из имени файла.
                        epoch_str = file[len("model_checkpoint_"):-len("_epochs.pth")]
                        epoch = int(epoch_str)
                        
                        # Если найден файл с более высокой эпохой, обновляем last_epoch.
                        if epoch > last_epoch:
                            last_epoch = epoch
                    except ValueError:
                        # В случае, если имя файла не содержит корректное число эпохи.
                        pf.print_warning_information(f"Пропущен файл с некорректным именем {file} в папке {dir_name}.")
                        need_line = True
            if need_line:
                print()
                pf.print_line()
            # Если в подкаталоге были файлы с эпохами, добавляем их в словарь.
            if last_epoch >= 0:
                found_runs[dir_name] = last_epoch

    # Выводим найденные данные.
    if not found_runs:
        pf.print_result_information("Ранних запусков модели не обнаружено! Будет загружена программная конфигурация модели!", end='\n\n')
    else:
        pf.print_info_information(f"Найдено {len(found_runs)} папок с обучением:")
        for dir_path, last_epoch in sorted(found_runs.items()):
            pf.print_info_information(f"  Папка: {dir_path}, Последняя эпоха: {last_epoch}.")
        print()
        pf.print_line()
    return found_runs

# Продолжить ли обучение модели.
def continue_learning_model():
    while True:
        continue_learning = input("Продолжить обучение или начать заново ([y/n]): ").strip().lower()
        if continue_learning in ("y", "n"):
            print()
            pf.print_line()
            if continue_learning == "y":
                return True
            else:
                return False
        else:
            pf.print_warning_information("Неверный ввод. Введите 'y' (да) или 'n' (нет).", end='\n\n')
            pf.print_line()

# Запрос числа эпох у пользователя в случае дообучения запуска модели.
def request_epochs(last_epoch: int):
    while True:
        try:
            epochs_to_train = int(input(f"Введите количество эпох для дообучения (последняя эпоха: {last_epoch}): ").strip())
            return epochs_to_train
        except ValueError:
            pf.print_warning_information("Пожалуйста, введите целое положительное число.")

# Импорт имеющегося конфига.
def import_config(selected_folder: str):
    # Путь к файлу config.json в выбранной подпапке
    config_file_path = os.path.join("./runs", selected_folder, "config.json")
    
    # Проверка, существует ли файл config.json
    if os.path.exists(config_file_path):
        pf.print_info_information(f"Загружаем конфигурацию из {config_file_path}")
        with open(config_file_path, "r") as file:
            try:
                # Чтение и парсинг JSON-файла
                new_config = json.load(file)
                pf.print_result_information("Конфигурация обновлена.", end='\n\n')
                return new_config
            except json.JSONDecodeError:
                pf.print_warning_information(f"Ошибка при чтении файла конфигурации: {config_file_path}")
                pf.print_critical_information("Опция прервана.", end='\n\n')
                return None
    else:
        pf.print_warning_information(f"Файл config.json не найден в {selected_folder}.")
        pf.print_critical_information("Опция прервана.", end='\n\n')
        return None

# Четвертая опция разработчика - обучение/дообучение модели ResNet50.
def dev_opt4_execute():
    # Информация об опции.
    pf.print_dev_opt4_info()
    pf.print_line(end='\n')
    print("\u001b[38;5;105m=== \u001b[38;5;104mРежим разработчика: Опция 4 \u001b[38;5;105m===\u001b[0m".center(125))
    pf.print_line()
    
    # Проверка наличия датасета для обучения.
    dataset_patches_dir = "./dataset_patches"
    if os.path.isdir(dataset_patches_dir):
        if is_dataset_structure_valid(dataset_patches_dir):
            pf.print_success_information("Каталог './dataset_patches' найден, структура данных корректна.")
        else:
            pf.print_warning_information("Проверьте корректность './dataset_patches' или воспользуйтесь опцией 3.")
            pf.print_critical_information("Опция прервана.", end='\n\n')
            return
    else:
        pf.print_warning_information(f"Папка c датасетом [{dataset_patches_dir}] не найдена в директории проекта.")
        pf.print_critical_information("Опция прервана.", end='\n\n')
        return
    print()
    pf.print_line()

    # Получение от пользователя действий по запуску с конфигурационного файла модели.
    found_runs = check_runs_dir("./runs")   # корневая папка логов обучения
    my_config = CONFIG                      # программный конфиг модели
    if found_runs:
        # Если найдено хотя бы одно обучение, спрашиваем, продолжить ли обучение.
        if continue_learning_model():
            while True:
                # Запрос выбора подпапки и количества эпох.
                selected_folder = input("Введите имя подпапки для продолжения обучения: ").strip()
                # Проверка, что выбранная подпапка существует.
                if selected_folder in found_runs:
                    last_epoch = found_runs[selected_folder]
                    pf.print_info_information(f"Последняя эпоха в выбранной подпапке: {last_epoch}.")
                    # Запрос числа эпох дообучения.
                    epochs_to_train = request_epochs(last_epoch)
                    # Логика продолжения обучения на основе выбранных данных.
                    pf.print_info_information(f"Продолжаем обучение модели в папке {selected_folder} с {last_epoch} до {last_epoch + epochs_to_train} эпохи.")
                    # Импорт конфига подпапки.
                    my_config = import_config(selected_folder)
                    if my_config is None:
                        return
                    my_config["model"]["add_epochs"] = epochs_to_train
                    my_config["run_path"] = os.path.join("./runs", selected_folder)
                    break
                else:
                    pf.print_warning_information(f"Выбранная папка '{selected_folder}' не найдена.", end='\n\n')
                    pf.print_line()
        else:
            pf.print_info_information("Обучение будет произведено заново на основе программной конфигурации модели.", end='\n\n')

    # Вывод всей полученной информации.
    pf.print_line()
    pf.print_result_information("Полученный конфиг:")
    # Folder path
    print(f"Папка запуска:          \u001b[38;5;46m{my_config['run_path']}\u001b[0m")
    # Model config
    model = my_config["model"]
    print("Модель:")
    print(f"  Архитектура:          \u001b[38;5;46m{model['architecture']}\u001b[0m")
    print(f"  Предобученные веса:   \u001b[38;5;46m{model['base_weights']}\u001b[0m")
    print(f"  Замороженные слои:    \u001b[38;5;46m{', '.join(model['unfrozen_layers'])}\u001b[0m")
    print(f"  Кол-во классов:       \u001b[38;5;46m{model['num_classes']}\u001b[0m")
    print(f"  Размер входа:         \u001b[38;5;46m{model['input_size']}\u001b[0m")
    print(f"  Устройство:           \u001b[38;5;46m{model['device']}\u001b[0m")
    print(f"  Эпох обучения:        \u001b[38;5;46m{model['prev_epochs']}\u001b[0m")
    print(f"  Эпох дообучения:      \u001b[38;5;46m{model['add_epochs']}\u001b[0m")
    print(f"  Функция потерь:       \u001b[38;5;46m{model['loss_fn']}\u001b[0m")
    # Optimizer
    opt = model["optimizer"]
    print("  Оптимизатор:")
    print(f"    Тип:                \u001b[38;5;46m{opt['type']}\u001b[0m")
    print(f"    lr:                 \u001b[38;5;46m{opt['lr']}\u001b[0m")
    print(f"    betas:              \u001b[38;5;46m{opt['betas']}\u001b[0m")
    print(f"    weight decay:       \u001b[38;5;46m{opt['weight_decay']}\u001b[0m")
    # Scheduler
    sch = model["scheduler"]
    print("  Планировщик:")
    print(f"    Тип:                \u001b[38;5;46m{sch['type']}\u001b[0m")
    print(f"    mode:               \u001b[38;5;46m{sch['mode']}\u001b[0m")
    print(f"    factor:             \u001b[38;5;46m{sch['factor']}\u001b[0m")
    print(f"    patience:           \u001b[38;5;46m{sch['patience']}\u001b[0m")
    print(f"    threshold:          \u001b[38;5;46m{sch['threshold']}\u001b[0m")
    # Training config
    training = my_config["training"]
    print("Обучение:")
    print(f"  Размер батча:         \u001b[38;5;46m{training['batch_size']}\u001b[0m")
    print(f"  Использование AMP:    \u001b[38;5;46m{training['use_amp']}\u001b[0m")
    print(f"  cuDNN:                \u001b[38;5;46m{training['cuDNN']}\u001b[0m")
    # Dataset config
    dataset = my_config["dataset"]
    print("Датасет:")
    print(f"  Train: оригиналы:     \u001b[38;5;46m{dataset['train_genuine_path']}\u001b[0m")
    print(f"         подделки:      \u001b[38;5;46m{dataset['train_forged_path']}\u001b[0m")
    print(f"  Val:   оригиналы:     \u001b[38;5;46m{dataset['validation_genuine_path']}\u001b[0m")
    print(f"         подделки:      \u001b[38;5;46m{dataset['validation_forged_path']}\u001b[0m")
    print(f"  Test:  оригиналы:     \u001b[38;5;46m{dataset['test_genuine_path']}\u001b[0m")
    print(f"         подделки:      \u001b[38;5;46m{dataset['test_forged_path']}\u001b[0m")
    print(f"  Классы:               \u001b[38;5;46m{', '.join(dataset['class_names'])}\u001b[0m")
    print(f"  Аугментация:          \u001b[38;5;46m{dataset['augmentation']}\u001b[0m")
    print(f"  Размер train выборки: \u001b[38;5;46m{dataset['train_size']}\u001b[0m")
    print(f"  Размер val выборки:   \u001b[38;5;46m{dataset['val_size']}\u001b[0m")
    print(f"  Размер test выборки:  \u001b[38;5;46m{dataset['test_size']}\u001b[0m")
    print()
    pf.print_line()

    # Работа с моделью ResNet50.
    trainer = CNNTrainer(my_config) # инициализации основных настроек модели
    trainer.process_learning()      # дообучение модели

    pf.print_result_information("Дообучение модели ResNet50 завершено.", end='\n\n')
    pf.print_line()
    pf.print_result_information("Визуализация и сохранение графиков процесса обучения завершены.", end='\n\n')
    pf.print_line()
    pf.print_success_information("Операция выполнена!", end='\n\n')

"""ЛОГИКА ОПЦИЙ"""
# Функция обрабоки основных опций Разработчика.
def dev_processing():
    while True:
        pf.print_dev_menu()
        mode = input("Выберите опцию: ").strip()
        if mode == '1':
           pf.print_info_information("Вы выбрали аннотирование датасета.", end='\n\n')
           pf.print_line()
           dev_opt1_execute()
           pf.print_line()
        elif mode == '2':
           pf.print_info_information("Вы выбрали редактирование аннотированного датасета.", end='\n\n')
           pf.print_line()
           dev_opt2_execute()
           pf.print_line()
        elif mode == '3':
           pf.print_info_information("Вы выбрали формирование датасета и его балансировку в формате .jpeg.", end='\n\n')
           pf.print_line()
           dev_opt3_execute()
           pf.print_line()
        elif mode == '4':
           pf.print_info_information("Вы выбрали обучение/дообучение модели ResNet50.", end='\n\n')
           pf.print_line()
           dev_opt4_execute()
           pf.print_line()
        elif mode == '5':
           pf.print_result_information("Выход...", end='\n\n')
           pf.print_line()
           break
        else:
           pf.print_warning_information("Неверный ввод. Пожалуйста, попробуйте снова.")
           pf.print_line()