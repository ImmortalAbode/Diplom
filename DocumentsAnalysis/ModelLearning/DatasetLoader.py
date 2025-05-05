import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

"""==================================================================================================>
| Класс, формирующий кастомный датасет из двух папок (genuine - оригиналы, forged - подделки).       |
| DataLoader - позволяет разбивать на батчи данные, оперировать с ними и перемешивать.               |
| Класс принимает 2 параметра, путь к первому классу (ОРИГИНАЛЫ) и путь ко второму классу (ПОДДЕЛКИ).|
<=================================================================================================="""
class Dataset2class(torch.utils.data.Dataset):
    # Запоминаем пути к обоим классам и сортируем для наличия хотя бы какой-то определенной последовательности файлов.
    def __init__(self, path_dir1:str, path_dir2:str):
        super().__init__()

        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))

        # Так обучалась ResNet50 в своё время!
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # сам преобразует в [0, 1] и (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # Метод, выводящий длину датасета из обоих папок.
    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)

    # Метод, принимающий один параметр - индекс соответствующего элемента датасета.
    """
    * По индексу определяем из какой папки взять элемент. 
    * Если индекс меньше длины первой папки - первый класс. Иначе - второй.
    """
    def __getitem__(self, idx: int):

        if idx < len(self.dir1_list): 
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
        else:
            class_id = 1
            # Вычитание из индекса длины первой папки - нормальный индекс относительно второй папки.
            idx -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[idx])

        # Приведение изображение к нормальному виду для обработкой CNN.
        img = Image.open(img_path).convert('RGB')   # открытие в RGB
        # --- Аналог преобразований библиотеки transforms ---
        t_img = self.transform(img)                             # преобразуем к нужному виду для ResNet50
        t_class_id = torch.tensor(class_id,  dtype=torch.long)  # класс тоже преобразуем в тензор

        return {'img': t_img, 'label': t_class_id}  # возвращаем пару: {изображение, метка} - суть обучения CNN.


"""ПРИМЕР АНАЛОГИЧНЫХ ПРЕОБРАЗОВАНИЙ transforms"""
# img = img.resize((224, 224))                # размер: 224x224
# img = np.array(img)                         # представление изображение в виде нормальных пикселей
# img = img.astype(np.float32)                # преобразование из int к float 32
# img = img/255.0                             # нормализация
# img = img.transpose((2, 0, 1))              # из представления (H, W, C) -> (C, H, W). С - каналы, H - высота, W - ширина
# формула нормализации по пикселям: x_norm = (x - mean) / std
# t_img = torch.from_numpy(img)               # преобразуем в тензор
