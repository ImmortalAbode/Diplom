from ConsoleOutput import PrintFunctions as pf
from ConsoleOutput import CoreFunctions as cf
from GUI import run_gui
import sys

# Идея обучения модели:
# ------------------------------------------------------------------------------------------------------------------|
#                                                                                                                   |
#   Изменение изображений до входного размера в ResNet50 до 224х224 чревато потерей необходимых паттернов фотошопа  |
#   В качестве решения пробуем разбивать пдф на изображения, а затем каждое изображение на патчи.                   |
#   Итог: pdf -> images -> patches * images картинок.                                                               |
#                                                                                                                   |
# ------------------------------------------------------------------------------------------------------------------|

# Основная функция начала программы.
def main():
    all_opt = True
    if all_opt:
        print()
        pf.print_welcome()
        while True:
            mode = input("Введите режим (1 - Разработчик, 2 - Пользователь, 0 - Выход): ").strip()

            if mode == '1':
                pf.print_info_information("Вы выбрали режим 'Разработчик'.", end='\n\n')
                pf.print_line()
                cf.dev_processing()

            elif mode == '2':
                pf.print_info_information("Вы выбрали режим 'Пользователь'.", end='\n\n')
                pf.print_line()
                run_gui()
                if not all_opt:
                    exit()

            elif mode == '0':
                print("Завершение программы...", end='\n\n')
                pf.print_line()
                break

            else:
                pf.print_warning_information("Неверный ввод. Пожалуйста, попробуйте снова.", end='\n\n')
                pf.print_line()
    else:
        run_gui()

    sys.exit()

# Запуск программы.
if __name__ == "__main__":
    main()    
