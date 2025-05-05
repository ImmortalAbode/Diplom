import os.path
import torch
import torch.nn as nn
import torchvision as tv

import os, json
from tqdm import tqdm
from .DatasetLoader import Dataset2class
import ConsoleOutput.PrintFunctions as pf

import matplotlib
matplotlib.use('TkAgg') # родной интерфейс tkinter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import csv
from datetime import datetime

"""================================================================================================>
| Класс, выполняющий основную работу с ResNet50.                                                   |
| Класс принимает 3 параметра: путь к набору датасета данных оригиналы/подделки по выборкам,       |
| размер батча и число эпох. Выполняет прогон датасета через модель и подсчет основных метрик и    |
| их визуализацию в виде графиков. Сохранение модели после обучения, а также параметров и логов.   |
<================================================================================================="""
class CNNTrainer:
    # Конструктор: создание и настройка основных параметров и путей для модели.
    def __init__(self, config: dict):
        self.config = config # конфигурационный файл модели

        # Параметры обучения.
        self.batch_size = config["training"]["batch_size"]
        self.prev_epochs = config["model"].get("prev_epochs", 0)
        self.add_epochs = config["model"]["add_epochs"]

        # Параметры для путей к данным.
        ds = config["dataset"]
        self.train_forged_path = ds["train_forged_path"]
        self.train_genuine_path = ds["train_genuine_path"]
        self.test_forged_path = ds["test_forged_path"]
        self.test_genuine_path = ds["test_genuine_path"]
        self.validation_forged_path = ds["validation_forged_path"]
        self.validation_genuine_path = ds["validation_genuine_path"]

        # Модель и устройство.
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config["model"]["device"] = str(self.device)

        # Параметры для обучения.
        self.optimizer = None       # Оптимизатор градиентов (SGD|Adam|AdamW, AdamW лучше).
        self.loss_fn = None         # Функция потерь.
        self.scheduler = None       # Экспоненциальное изменение скорости обучения.
        self.scaler = None          # Использование AMP для оптимизации вычислений на GPU.

        # Оптимизация и обучение.
        # --- Оптимизатор ---
        optimizer_cfg = config["model"]["optimizer"]
        self.lr = optimizer_cfg["lr"]                       # скорость обучения
        self.betas = tuple(optimizer_cfg["betas"])          # variance (насколько долго учитвать разброс), momentum (память о направлении градиента)
        self.weight_decay = optimizer_cfg["weight_decay"]   # параметр регуляризации

        # --- Шедулер ---
        scheduler_cfg = config["model"]["scheduler"]
        self.mode = scheduler_cfg["mode"]               # min - метрика потерь, max - метрика точности
        self.factor = scheduler_cfg["factor"]           # изменение lr на шаге
        self.patience = scheduler_cfg["patience"]       # сколько эпох ждать без улучшения валидации
        self.min_threshold = scheduler_cfg["threshold"] # минимальный порог улучшения
        
        # --- Разморозка слоёв ---
        self.unfreeze = config["model"].get("unfrozen_layers", [])

        # Директории для логов, весов и конфигов.
        # --- Папка текущего запуска ---
        self.run_dir = self.config["run_path"]
        if not self.run_dir:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.run_dir = os.path.join("runs", timestamp)
            os.makedirs(self.run_dir, exist_ok=True)
            self.config["run_path"] = self.run_dir
        # --- Файл логов обучения и валидации ---
        self.log_path = os.path.join(self.run_dir, "train_val_log.csv")
        # --- Файл логов тестирования ---
        self.test_path = os.path.join(self.run_dir, "test_log.csv")
        # --- Файл весов модели ---
        self.weights_path = os.path.join(self.run_dir, "model_checkpoint")
        # --- Конфиг модели ---
        self.config_path = os.path.join(self.run_dir, "config.json")
        # --- График обучения ---
        self.plot_path = os.path.join(self.run_dir, "acc_loss_train_valid.png")
        # --- График скорости обучения ---
        self.plot_lr_path = os.path.join(self.run_dir, "lr_shedule.png")

        # Подготовка модели.
        self._build_model()         # построение модели ResNet50
        self._prepare_training()    # придание основным настройкам модели конкретных значений
        # Загрузка данных.
        self.train_loader, self.test_loader, self.validation_loader = self._load_data()

        # Обновленная конфигурация.
        pf.print_result_information("Обновление конфигурации...")
        print("Модель:")
        print(f"  Устройство:           \u001b[38;5;46m{self.config['model']['device']}\u001b[0m")
        print("Обучение:")
        print(f"  Использование AMP:    \u001b[38;5;46m{self.config['training']['use_amp']}\u001b[0m")
        print(f"  cuDNN:                \u001b[38;5;46m{self.config['training']['cuDNN']}\u001b[0m")
        print("Датасет:")
        print(f"  Размер train выборки: \u001b[38;5;46m{self.config['dataset']['train_size']}\u001b[0m")
        print(f"  Размер val выборки:   \u001b[38;5;46m{self.config['dataset']['val_size']}\u001b[0m")
        print(f"  Размер test выборки:  \u001b[38;5;46m{self.config['dataset']['test_size']}\u001b[0m")
        print()
        pf.print_line()

        # Выводы информации.
        pf.print_info_information(f"Каталог сохранения:                     \u001b[38;5;45m{self.run_dir}\u001b[0m")
        pf.print_info_information(f" Файл обучения:                         \u001b[38;5;45m{self.log_path}\u001b[0m")
        pf.print_info_information(f" Файл параметров CNN:                   \u001b[38;5;45m{self.weights_path}_XX_epochs.pth\u001b[0m")
        pf.print_info_information(f" Конфиг CNN:                            \u001b[38;5;45m{self.config_path}\u001b[0m")
        pf.print_info_information(f" Файл тестирования:                     \u001b[38;5;45m{self.test_path}\u001b[0m")
        pf.print_info_information(f" Файл графиков обучения и валидации:    \u001b[38;5;45m{self.plot_path}\u001b[0m")
        pf.print_info_information(f" Файл графика скорости обучения:        \u001b[38;5;45m{self.plot_lr_path}\u001b[0m")
        pf.print_info_information(f" Папка ошибок валидации по эпохам:      \u001b[38;5;45m./runs\\2025-05-02_19-27-00\\mistake_patches\u001b[0m")  
        print()
        pf.print_line()
        
    # Создаем модель ResNet50 и модифицируем ее для 2 классов.
    def _build_model(self):
        # Устанавливаем кастомный путь для загрузки весов.
        custom_weights_dir = "./base_weights"               # папка, куда сохраняются предобученные веса в директории проекта
        os.environ["TORCH_HOME"] = custom_weights_dir       # перенаправляем кеш данных в проект

        # Полный путь до весов, которые загрузит torchvision (по имени ResNet50)
        filename = "resnet50-0676ba61.pth"
        weights_path = os.path.join(custom_weights_dir, "hub", "checkpoints", filename)
        self.model = tv.models.resnet50(weights=None)

        # Загрузка прям предобученных базовых весов, если никаких предыдущих эпох обучения у модели не было.
        if self.config["model"]["prev_epochs"] == 0:
            # Проверка наличия весов.
            if not os.path.exists(weights_path):
                pf.print_info_information("Загрузка предобученной модели, пожалуйста, подождите...")
                self.model = tv.models.resnet50(weights = self.config["model"]["base_weights"])  # Скачает и автоматически сохранит в custom_weights_dir
                print(f"Веса были скачаны и сохранены по пути {weights_path}")
            else:
                print(f"Предобученная модель обнаружена по пути {weights_path}")
                # Загрузка имеющихся предобученных весов.
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            # Модификация модели для 2 классов.
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
               nn.Dropout(p=0.5),       # Dropout с вероятностью 50% для регуляризации
               nn.Linear(num_ftrs, 2)   # FC для двух классов
            )
            pf.print_info_information("Произведена модификация модели на 2 класса в FC слое с добавлением Dropout.")
        # Если мы дообучаем с какой-то эпохи, то загрузка последних весов обучения.
        else:
            # Загружаем последние сохранённые веса модели (если prev_epochs > 0).
            # Путь к последнему файлу настроек, используя номер эпохи.
            last_weights_path = os.path.join(self.run_dir, f"model_checkpoint_{self.prev_epochs:02d}_epochs.pth")

            if os.path.exists(last_weights_path):
                print(f"Загрузка последних сохранённых весов из {last_weights_path}")
                # Модификация модели для 2 классов.
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                   nn.Dropout(p=0.5),       # Dropout с вероятностью 50% для регуляризации
                   nn.Linear(num_ftrs, 2)   # FC для двух классов
                )
                pf.print_info_information("Произведена модификация модели на 2 класса в FC слое с добавлением Dropout.")
                # Загрузка весов из последней сохранённой эпохи
                state_dict = torch.load(last_weights_path, map_location="cpu")
                self.model.load_state_dict(state_dict['model_state_dict'])

        pf.print_info_information("Предобученная модель была загружена.")
        trainable_params = self._count_parameters(self.model)
        pf.print_info_information(f"Количество обучаемых параметров модели: {trainable_params}.")

        if self.unfreeze:
            # Заморозка всех слоев.
            pf.print_info_information("Заморозка всех обучаемых параметров.")
            for param in self.model.parameters():
                param.requires_grad = False

            # # Размораживаем последний FC слой.
            pf.print_info_information(f"Разморозка параметров {', '.join(self.unfreeze)}.")
            for name, param in self.model.named_parameters():
                if any(name.startswith(layer_name) for layer_name in self.unfreeze):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        trainable_params = self._count_parameters(self.model)
        pf.print_info_information(f"Количество обучаемых параметров модели: {trainable_params}.", end='\n\n')
        for name, param in self.model.named_parameters():
            status = "обучается" if param.requires_grad else "заморожен"
            print(f"\t{status}: {name}")
        print()
        pf.print_line()

        # Перенос модели на устройство (CPU или GPU).
        self.model = self.model.to(self.device)

    # Подготовка функций потерь, оптимизаторов и скейлеров.
    def _prepare_training(self):
        # Функция потерь и оптимизатор, шедулер.
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.lr, 
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.mode,                 
            factor=self.factor,             
            patience=self.patience,         
            threshold=self.min_threshold,   
        )

        # Ускорение на GPU с использованием AMP (смешанной точности).
        self.scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
        if self.scaler:
            self.config["training"]["use_amp"] = "GradScaler"

        # Настройка cuDNN для оптимизации производительности на GPU.
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            self.config["training"]["cuDNN"] = True

        # Загружаем последние сохранённые состояния оптимизатора, шедулера и скейлера (если prev_epochs > 0).
        if self.config["model"]["prev_epochs"] > 0:
            # Путь к последнему файлу настроек, используя номер эпохи.
            last_weights_path = os.path.join(self.run_dir, f"model_checkpoint_{self.prev_epochs:02d}_epochs.pth")
            if os.path.exists(last_weights_path):
                print(f"Загрузка последних сохранённых состояний оптимизатора, шедулера и скейлера из {last_weights_path}")
                state_dict = torch.load(last_weights_path, map_location="cpu")
                self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
                if 'scaler_state_dict' in state_dict and self.scaler:
                    self.scaler.load_state_dict(state_dict['scaler_state_dict'])

    # Метод для подсчета обучаемых параметров модели.
    def _count_parameters(self, model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Метод загрузки и подготовки данных для обучения и тестирования.
    def _load_data(self):
        # Формирование патчей датасета по выборкам.
        train_ds = Dataset2class(self.train_genuine_path, self.train_forged_path)
        test_ds = Dataset2class(self.test_genuine_path, self.test_forged_path)
        validation_ds = Dataset2class(self.validation_genuine_path, self.validation_forged_path)

        # DataLoader.
        # Параметры - датасет, перемешивание, размер батча, число параллельных потоков, обрезка лишних изображений.
        train_loader = torch.utils.data.DataLoader(
            train_ds, shuffle=True, batch_size=self.batch_size, num_workers=1, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, shuffle=False, batch_size=self.batch_size, num_workers=1, drop_last=True
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_ds, shuffle=False, batch_size=self.batch_size, num_workers=1, drop_last=True
        )
        # Обновление конфига.
        self.config["dataset"]["train_size"] = len(train_ds)
        self.config["dataset"]["val_size"] = len(validation_ds)
        self.config["dataset"]["test_size"] = len(test_ds)

        return train_loader, test_loader, validation_loader

    # Метод для подсчета точности модели.
    def _accuracy(self, pred: torch.Tensor, label: torch.Tensor):
        pred_classes = pred.argmax(dim=1)
        if label.dim() > 1:
            label = label.argmax(dim=1)
        return (pred_classes == label).float().mean().item()

    # Метод проверки модели после каждой эпохи на валидационном наборе данных.
    def _validate(self, epoch: int):
        self.model.eval()  # переводим модель в режим оценки
        val_loss = 0
        val_acc = 0

        # Обратная нормализация: x_norm = (x - mean) / std => x = x_norm * std + mean
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        unnormalize = tv.transforms.Compose([
            tv.transforms.Lambda(lambda x: x * torch.tensor(std).view(-1, 1, 1)),  # умножаем на std
            tv.transforms.Lambda(lambda x: x + torch.tensor(mean).view(-1, 1, 1))  # добавляем mean
        ])

        # Папка для сохранения ошибочных патчей/
        mistakes_dir = os.path.join(self.run_dir, "mistake_patches", f"epoch_{epoch+1}")
        os.makedirs(mistakes_dir, exist_ok=True)


        with torch.no_grad():  # отключаем вычисление градиентов
            for batch_idx, sample in enumerate(self.validation_loader):
                img, label = sample['img'].to(self.device), sample['label'].to(self.device)

                # Прогон модели.
                pred = self.model(img)              
                loss = self.loss_fn(pred, label)

                # Считаем точность.
                loss_item = loss.item()
                acc_current = self._accuracy(pred.cpu().float(), label.cpu().float())
                val_loss += loss_item
                val_acc += acc_current

                # Обновление прогресс-бара.
                self.pbar.set_description(f'Валидация модели: epoch {epoch+1}/{self.add_epochs + self.prev_epochs}')
                self.pbar.set_postfix(loss=f'{loss_item:.5f}', accuracy=f'{acc_current:.3f}', refresh=True)
                self.pbar.update(1)

                # Получаем предсказанные и истинные классы.
                pred_classes = pred.argmax(dim=1)
                true_classes = label.argmax(dim=1) if label.dim() > 1 else label

                # Индексы ошибочных предсказаний.
                incorrect_indices = (pred_classes != true_classes).nonzero(as_tuple=True)[0]    # из массива по критерию достаем индексы (выход тензор)

                # Сохраняем ошибочные патчи.
                for idx in incorrect_indices:
                    wrong_img = img[idx].cpu()                # изображение
                    wrong_img = unnormalize(wrong_img)        # денормализация к нормальному виду
                    wrong_img = torch.clamp(wrong_img, 0, 1)  # чтобы избежать выхода за [0,1]
                    true_label = true_classes[idx].item()
                    pred_label = pred_classes[idx].item()
                    save_path = os.path.join(mistakes_dir, f"batch{batch_idx}_idx{idx.item()}_true{true_label}_pred{pred_label}.png")
                    tv.utils.save_image(wrong_img, save_path)

        # Возвращаем среднее значение потерь и точности
        val_loss_avg = val_loss / len(self.validation_loader)
        val_acc_avg = val_acc / len(self.validation_loader)

        return val_loss_avg, val_acc_avg

    # Тестирование модели.
    def _test(self, epoch: int):
        self.model.eval()   # перевод в режим тестирования
        test_loss = 0
        test_acc = 0

        with torch.no_grad():  # отключаем вычисление градиентов
            for sample in self.test_loader:
                img, label = sample['img'].to(self.device), sample['label'].to(self.device)

                # Прогон модели.
                pred = self.model(img)              
                loss = self.loss_fn(pred, label)

                # Считаем точность и потери.
                loss_item = loss.item()
                acc_current = self._accuracy(pred.cpu().float(), label.cpu().float())
                test_loss += loss_item
                test_acc += acc_current

                # Обновление прогресс-бара.
                self.pbar.set_description(f'Тестирование модели: epoch {epoch+1}')
                self.pbar.set_postfix(loss=f'{loss_item:.5f}', accuracy=f'{acc_current:.3f}', refresh=True)
                self.pbar.update(1)

        # Возвращаем среднее значение потерь и точности.
        test_loss_avg = test_loss / len(self.test_loader)
        test_acc_avg = test_acc / len(self.test_loader)

        # Сохраняем результаты теста в CSV.
        header = ["epoch", "test_loss", "test_acc"]
        file_exists = os.path.exists(self.test_path)
        if not file_exists:
            with open(self.test_path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader() # cоздаёт объект writer, который будет записывать строки в формате словаря, где ключи — это имена столбцов (header)
        with open(self.test_path, mode="a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow({
                "epoch": epoch + 1,
                "test_loss": test_loss_avg,
                "test_acc": test_acc_avg
            })

    # Обучение модели.
    def _train(self, epoch: int):
        self.model.train() # перевод в режим обучения
        loss_val = 0
        acc_val = 0

        # Проход по батчам. sample = (img, label), где img - это тензор изображений, а label - тензор меток.
        for sample in self.train_loader:
            img, label = sample['img'].to(self.device), sample['label'].to(self.device)
            self.optimizer.zero_grad()  # обнуление градиентов (особенность PyTorch)

            # Оптимизация с использованием AMP, если доступно.
            if self.scaler:
                with torch.amp.autocast():
                    pred = self.model(img)
                    loss = self.loss_fn(pred, label)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(img)
                loss = self.loss_fn(pred, label)
                loss.backward()
                self.optimizer.step()

            # Считаем потери и точность на текущем батче.
            loss_item = loss.item()
            loss_val += loss_item
            acc_current = self._accuracy(pred.cpu().float(), label.cpu().float())
            acc_val += acc_current

            # Обновление прогресс-бара.
            self.pbar.set_description(f'Обучение модели: epoch {epoch+1}/{self.add_epochs + self.prev_epochs}')
            self.pbar.set_postfix(loss=f'{loss_item:.5f}', accuracy=f'{acc_current:.3f}', refresh=True)
            self.pbar.update(1)

        # Вычисление средней ошибки и точности на эпохе.
        train_loss_avg = loss_val / len(self.train_loader)
        train_acc_avg = acc_val / len(self.train_loader)

        return train_loss_avg, train_acc_avg

    # Координация обучения.
    def process_learning(self):
        pf.print_learning()
        # Заголовки CSV-файла
        header = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"]

        # Проверка наличия файла.
        file_exists = os.path.exists(self.log_path)

        if not file_exists:
            with open(self.log_path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader() # cоздаёт объект writer, который будет записывать строки в формате словаря, где ключи — это имена столбцов (header)

        # Цикл по эпохам.
        train_val_steps = self.add_epochs * (len(self.train_loader) + len(self.validation_loader))
        test_epochs = len(range(self.prev_epochs + 1, self.prev_epochs + self.add_epochs + 1, 5))
        test_steps = test_epochs * len(self.test_loader)
        total_steps = train_val_steps + test_steps
        self.pbar = tqdm(
            total=total_steps, 
            desc=f'Обучение, валидация и тестирование модели: epoch {self.prev_epochs}/{self.add_epochs + self.prev_epochs}',
            ncols=200
            )
        for epoch in range(self.prev_epochs, self.prev_epochs + self.add_epochs):
            # Обучение модели на эпохе.
            train_loss_avg, train_acc_avg = self._train(epoch)
            # Валидация модели после каждой эпохи.
            val_loss_avg, val_acc_avg = self._validate(epoch)
            # Тестирование модели каждые 5 эпох.
            if (epoch + 1) % 5 == 0:
                self._test(epoch)
            # --- Сохраняем текущие метрики в CSV ---
            with open(self.log_path, mode="a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writerow({
                    "epoch": epoch + 1,
                    "train_loss": train_loss_avg,
                    "val_loss": val_loss_avg,
                    "train_acc": train_acc_avg,
                    "val_acc": val_acc_avg,
                    "lr": self.optimizer.param_groups[0]['lr']
                })
            # Обновление метрик после эпохи на основе потерь валидации.
            self.scheduler.step(val_loss_avg)
            # Сохранение обученной модели на каждой эпохе.
            self._save_model(epoch)
            # Обновление конфига.
            self.config["model"]["prev_epochs"] += 1
            self.config["model"]["add_epochs"] -= 1
            # Сохранение конфига.
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
            # Визуализация процесса обучения и тестирования. Сохранение графиков.
            if (epoch + 1) >= 2:
               self.plot_metrics()
        self.pbar.close()
        print()
        pf.print_line()

    # Сохранение модели.
    def _save_model(self, epoch: int):
        try:
            # Формируем путь с сохранением всех настроек модели.
            filename = f"{self.weights_path}_{epoch + 1:02d}_epochs.pth"
            torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            }, filename)
        except Exception as e:
            self.pbar.write(f"\nОшибка при сохранении модели: {e}.", end='\n\n')

    # Сохранение графиков.
    def _save_plot(self, fig: Figure, suffix: str):
        try:
            if suffix == "acc_loss":
                fig.savefig(self.plot_path, dpi=300)
            if suffix == "lr":
                fig.savefig(self.plot_lr_path, dpi=300)
        except Exception as e:
            self.pbar.write(f"\nОшибка при сохранении графиков: {e}.", end='\n\n')

    # Отрисовка графиков. Графики точности и потерь для обучения и валидации строятся по эпохам как среднее по батчу.
    def plot_metrics(self):
        # Списки для хранения данных из CSV
        epochs = []
        loss_train = []
        acc_train = []
        loss_valid = []
        acc_valid = []
        lr = []
        # Загрузка данных из CSV
        with open(self.log_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                epochs.append(int(row["epoch"]))
                loss_train.append(float(row["train_loss"]))
                loss_valid.append(float(row["val_loss"]))
                acc_train.append(float(row["train_acc"]))
                acc_valid.append(float(row["val_acc"]))
                lr.append(float(row["lr"]))

        # Графики для обучения и валидации (по эпохам).
        plt.figure(figsize=(14, 6)) # размер окна в дюймах
    
        # --- Потери ---
        plt.subplot(1, 2, 1)    # под окно для графика train and validation loss
        plt.plot(range(1, len(loss_train) + 1), loss_train, label='Training Loss', color='red')     # график (x, y, текст легенды, цвет)
        plt.plot(range(1, len(loss_valid) + 1), loss_valid, label='Validation Loss', color='blue')    # график (x, y, текст легенды, цвет)
        plt.xlabel('Epoch') # название оси х
        plt.ylabel('Loss')  # название оси у
        plt.title('Training and Validation Loss')   # название графика
        plt.legend()    # отрисовка легенды
        plt.ylim(0, max(max(loss_train), max(loss_valid)) * 1.1) # масштабирование оси y
        plt.xlim(1, len(epochs))    # масштабирование оси x
        plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)  # сетка графика

        # --- Точность ---
        plt.subplot(1, 2, 2)    # под окно для графика train and validation accuracy
        plt.plot(range(1, len(acc_train) + 1), acc_train, label='Training Accuracy', color='red')       # график (x, y, текст легенды, цвет)
        plt.plot(range(1, len(acc_valid) + 1), acc_valid, label='Validation Accuracy', color='blue')    # график (x, y, текст легенды, цвет)
        plt.xlabel('Epoch')     # название оси х
        plt.ylabel('Accuracy')  # название оси у
        plt.title('Training and Validation Accuracy')   # название графика
        plt.legend()    # отрисовка легенды
        plt.ylim(0, max(max(acc_train), max(acc_valid)) * 1.1) # масштабирование оси y
        plt.xlim(1, len(epochs))    # масштабирование оси x
        plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)  # сетка графика
        plt.tight_layout()  # отступы

        fig = plt.gcf()
        self._save_plot(fig, "acc_loss")  # сохраняем график точности и потерь для обучения и валидации
        plt.close(fig)

         # --- График изменения Learning Rate ---
        plt.figure(figsize=(8, 4))  # окно для графика learning rate
        plt.plot(epochs, lr, label='Learning Rate', color='green')   # график (x, y, текст легенды, цвет)
        plt.xlabel('Epoch')         # название оси х
        plt.ylabel('Learning Rate') # название оси y
        plt.title('Learning Rate Schedule') # название графика
        plt.legend()    # отрисовка легенды
        plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)   # сетка графика
        plt.tight_layout()  # отступы

        fig = plt.gcf()
        self._save_plot(fig, "lr")  # сохраняем график скорости обучения
        plt.close(fig)
   
