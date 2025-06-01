"""
Разбирает XML-файл MuJoCo для извлечения геометрии стен лабиринта.
"""
import xml.etree.ElementTree as ET
import numpy as np
from maze_config import MAZE_XML_PATH

class Wall:
    """Представляет прямоугольную стену в 2D пространстве."""
    def __init__(self, center_x, center_y, size_x, size_y, name=None):
        """
        Аргументы:
            center_x (float): X-координата центра стены.
            center_y (float): Y-координата центра стены.
            size_x (float): Половина размера стены вдоль ее локальной оси X (ширина/2).
            size_y (float): Половина размера стены вдоль ее локальной оси Y (глубина/2).
                           Примечание: размеры геометрии box в MuJoCo - это половины длин.
            name (str, опционально): Имя геометрии.
        """
        self.center = np.array([center_x, center_y])
        # Храним половины размеров напрямую, так как они даны в XML MuJoCo для типа 'box'
        self.half_size_x = size_x 
        self.half_size_y = size_y
        self.name = name

        # Вычисляем угловые точки для упрощения проверки столкновений или визуализации
        # Предполагается, что стены выровнены по осям в их локальной системе координат, что верно для неповернутых геометрий верхнего уровня
        self.min_x = center_x - size_x
        self.max_x = center_x + size_x
        self.min_y = center_y - size_y
        self.max_y = center_y + size_y

    def __repr__(self):
        return f"Стена(имя='{self.name}', центр=({self.center[0]:.2f}, {self.center[1]:.2f}), полуразмеры=({self.half_size_x:.2f}, {self.half_size_y:.2f}))"

    def get_patch_params(self):
        """Возвращает параметры для патча Matplotlib Rectangle."""
        return (self.min_x, self.min_y), 2 * self.half_size_x, 2 * self.half_size_y

class MazeParser:
    """Разбирает XML-файл MuJoCo для извлечения информации о стенах лабиринта."""
    def __init__(self, xml_path=MAZE_XML_PATH):
        self.xml_path = xml_path
        self.walls = []
        self.world_min_bounds = np.array([np.inf, np.inf])
        self.world_max_bounds = np.array([-np.inf, -np.inf])
        self._parse_xml()

    def _parse_xml(self):
        """Защищенный метод для разбора XML и заполнения списка стен."""
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
        except FileNotFoundError:
            print(f"Ошибка: XML-файл не найден по пути {self.xml_path}")
            return
        except ET.ParseError as e:
            print(f"Ошибка разбора XML-файла {self.xml_path}: {e}")
            return

        # Ищем геометрии внутри worldbody или вложенных body
        # Для scene_2.xml стены находятся непосредственно под body в worldbody
        for body_element in root.findall('.//worldbody/body'): # Находит все элементы <body> под <worldbody>
            # Получаем смещение позиции body, по умолчанию (0,0,0), если не указано
            body_pos_str = body_element.get('pos', '0 0 0')
            body_pos_parts = list(map(float, body_pos_str.split()))
            body_offset = np.array([body_pos_parts[0], body_pos_parts[1]]) # Только X, Y

            for geom_element in body_element.findall('geom'):
                if geom_element.get('type') == 'box':
                    name = geom_element.get('name', 'безымянная_стена')
                    pos_str = geom_element.get('pos', '0 0 0')
                    size_str = geom_element.get('size', '0 0 0') # половины длин
                    # rgba_str = geom_element.get('rgba') # Для возможной фильтрации при необходимости

                    try:
                        pos_parts = list(map(float, pos_str.split()))
                        size_parts = list(map(float, size_str.split()))
                    except ValueError as e:
                        print(f"Предупреждение: Не удалось разобрать pos/size для геометрии '{name}': {e}. Пропускается.")
                        continue

                    # Нас интересует 2D плоскость (X, Y)
                    # Позиция геометрии относительна ее родительского body
                    center_x = body_offset[0] + pos_parts[0]
                    center_y = body_offset[1] + pos_parts[1]
                    # Для box, size это (половина_ширины, половина_глубины, половина_высоты)
                    # Мы предполагаем, что стены в основном определены в плоскости XY, поэтому relevant half_size_x, half_size_y
                    half_size_x = size_parts[0]
                    half_size_y = size_parts[1]
                    # Мы игнорируем Z-измерение для 2D планирования пути

                    wall = Wall(center_x, center_y, half_size_x, half_size_y, name)
                    self.walls.append(wall)
                    
                    # Обновляем границы мира
                    self.world_min_bounds[0] = min(self.world_min_bounds[0], wall.min_x)
                    self.world_min_bounds[1] = min(self.world_min_bounds[1], wall.min_y)
                    self.world_max_bounds[0] = max(self.world_max_bounds[0], wall.max_x)
                    self.world_max_bounds[1] = max(self.world_max_bounds[1], wall.max_y)

        print(f"Разобрано {len(self.walls)} стен из {self.xml_path}.")
        if not self.walls:
            print("Предупреждение: Стены не были разобраны. Проверьте структуру XML и типы геометрий.")
        else:
            print(f"Границы мира (min_x, min_y): ({self.world_min_bounds[0]:.2f}, {self.world_min_bounds[1]:.2f})")
            print(f"Границы мира (max_x, max_y): ({self.world_max_bounds[0]:.2f}, {self.world_max_bounds[1]:.2f})")

    def get_walls(self):
        """Возвращает список разобранных объектов Wall."""
        return self.walls

    def get_world_bounds(self):
        """Возвращает мин. и макс. XY координаты разобранного мира."""
        return self.world_min_bounds, self.world_max_bounds

if __name__ == '__main__':
    # Пример использования:
    parser = MazeParser()
    walls = parser.get_walls()
    for wall in walls:
        print(wall)
    min_bounds, max_bounds = parser.get_world_bounds()
    # print(f"Мин. границы: {min_bounds}, Макс. границы: {max_bounds}") 