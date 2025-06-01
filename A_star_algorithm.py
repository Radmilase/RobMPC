"""
Реализация алгоритма поиска пути A* для лабиринта.
"""
import heapq
import numpy as np
import time # <--- Добавляем импорт time
from maze_config import ROBOT_START_POS_2D, TARGET_POS_2D, GRID_RESOLUTION, HEURISTIC_WEIGHT
from maze_parser import MazeParser
from trajectory_visualizer import TrajectoryVisualizer # Для тестирования

class Node:
    """Представляет узел в сетке поиска A*."""
    def __init__(self, position, parent=None):
        self.position = np.array(position) # (x, y) координаты
        self.parent = parent
        self.g = 0  # Стоимость от начального до текущего узла
        self.h = 0  # Эвристическая стоимость от текущего узла до цели
        self.f = 0  # Общая стоимость (g + h)

    def __eq__(self, other):
        return np.array_equal(self.position, other.position)

    def __lt__(self, other):
        # heapq - это мин-куча, поэтому меньшее значение f имеет более высокий приоритет
        return self.f < other.f
    
    def __hash__(self):
        # Требуется для добавления узлов в множество
        return hash(tuple(self.position))

class AStar:
    """Реализует алгоритм поиска пути A*."""
    DECIMAL_PRECISION = 5 # Для округления позиций и решения проблем с плавающей точкой

    def _get_position_tuple(self, pos_np):
        """Округляет массив numpy с позицией и конвертирует в хешируемый кортеж."""
        return tuple(np.round(pos_np, self.DECIMAL_PRECISION))

    def __init__(self, walls, world_min_bounds, world_max_bounds, start_pos, goal_pos, resolution, robot_radius):
        """
        Аргументы:
            walls (list): Список объектов Wall.
            world_min_bounds (np.array): Мин (x, y) мира.
            world_max_bounds (np.array): Макс (x, y) мира.
            start_pos (np.array): Начальная позиция (x, y).
            goal_pos (np.array): Целевая позиция (x, y).
            resolution (float): Размер ячейки сетки.
            robot_radius (float): Радиус робота для проверки столкновений (инфляция).
        """
        self.walls = walls
        self.world_min_bounds = world_min_bounds # Сохраняем оригинал для справки или других использований
        self.world_max_bounds = world_max_bounds
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        self.resolution = resolution
        self.robot_radius = robot_radius

        # Эффективные границы для поиска, немного расширенные, чтобы избежать проблем на самом краю
        self.search_min_bounds = self.world_min_bounds - self.resolution / 2.01 # Расширяем чуть меньше чем на половину разрешения для безопасности
        self.search_max_bounds = self.world_max_bounds + self.resolution / 2.01

        self.start_node_pos = self._discretize_point(self.start_pos)
        self.goal_node_pos = self._discretize_point(self.goal_pos)
        self.goal_node_pos_tuple = self._get_position_tuple(self.goal_node_pos) # Сохраняем округленный кортеж для проверки цели
        
        self.open_set = []  # Очередь с приоритетом (мин-куча)
        self.closed_set = set() # Множество посещенных позиций узлов (в виде кортежей для хешируемости)
        # open_set_dict отображает округленные кортежи позиций на объекты Node в open_set
        self.open_set_dict = {}

        self.grid_min_x, self.grid_min_y = self._discretize_point(world_min_bounds) - resolution # Немного расширяем
        self.grid_max_x, self.grid_max_y = self._discretize_point(world_max_bounds) + resolution

    def _discretize_point(self, point):
        """Привязывает непрерывную точку к центру ее ячейки сетки."""
        # Привязка к ближайшей линии сетки, затем смещение на половину разрешения, чтобы получить центр ячейки
        snapped = np.round(point / self.resolution) * self.resolution
        return snapped # Для A* часто работают с индексами сетки или центрами. Будем использовать центры.

    def _heuristic(self, pos1, pos2):
        """Манхэттенская эвристика расстояния (можно изменить на Евклидову)."""
        # return np.sum(np.abs(pos1 - pos2)) * HEURISTIC_WEIGHT # Манхэттенское
        return np.linalg.norm(pos1 - pos2) * HEURISTIC_WEIGHT # Евклидово

    def _is_valid_node_pos(self, node_pos):
        """Проверяет, находится ли позиция узла в границах и не сталкивается ли со стеной."""
        # Проверка эффективных границ поиска
        if not (self.search_min_bounds[0] <= node_pos[0] <= self.search_max_bounds[0] and \
                self.search_min_bounds[1] <= node_pos[1] <= self.search_max_bounds[1]):
            # print(f"Узел {node_pos} находится вне эффективных границ поиска: Мин({self.search_min_bounds}), Макс({self.search_max_bounds})") # ОТЛАДКА
            return False

        # Проверка столкновений со стенами (учитывая радиус робота)
        for wall in self.walls:
            inflated_min_x = wall.min_x - self.robot_radius
            inflated_max_x = wall.max_x + self.robot_radius
            inflated_min_y = wall.min_y - self.robot_radius
            inflated_max_y = wall.max_y + self.robot_radius

            if (node_pos[0] >= inflated_min_x and node_pos[0] <= inflated_max_x and
                node_pos[1] >= inflated_min_y and node_pos[1] <= inflated_max_y):
                # Если центр узла находится внутри или на границе расширенной стены, это столкновение.
                return False # Столкновение
        return True

    def _get_neighbors(self, current_node_pos):
        """Генерирует действительные позиции соседних узлов (8-связность)."""
        neighbors = []
        for dx in [-self.resolution, 0, self.resolution]:
            for dy in [-self.resolution, 0, self.resolution]:
                if dx == 0 and dy == 0:
                    continue # Пропускаем текущий узел
                
                neighbor_pos = current_node_pos + np.array([dx, dy])
                # Дискретизируем, чтобы убедиться, что он выровнен по сетке, хотя добавление разрешения должно сохранять выравнивание
                # neighbor_pos = self._discretize_point(neighbor_pos) 
                
                if self._is_valid_node_pos(neighbor_pos):
                    neighbors.append(neighbor_pos)
        return neighbors

    def _reconstruct_path(self, current_node):
        """Восстанавливает путь от целевого узла обратно к начальному."""
        path = []
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        return path[::-1] # Разворачиваем, чтобы получить путь от старта к цели

    def find_path(self):
        """Выполняет алгоритм поиска A*."""
        start_time = time.time() # <--- Засекаем время начала

        print(f"A* Инициализация. Старт: {self.start_pos}, Цель: {self.goal_pos}")
        print(f"A* Дискретизированный начальный узел: {self.start_node_pos}, Дискретизированный целевой узел: {self.goal_node_pos}")
        # print(f"A* Минимальные/Максимальные границы мира (сырые): {self.world_min_bounds} / {self.world_max_bounds}") # Опционально: сохранить, если полезно
        # print(f"A* Эффективные минимальные/максимальные границы поиска: {self.search_min_bounds} / {self.search_max_bounds}") # Опционально: сохранить, если полезно

        if not self._is_valid_node_pos(self.start_node_pos):
            print(f"A* КРИТИЧЕСКАЯ ОШИБКА: Начальный узел {self.start_node_pos} недействителен!")
            if not (self.search_min_bounds[0] <= self.start_node_pos[0] <= self.search_max_bounds[0] and \
                    self.search_min_bounds[1] <= self.start_node_pos[1] <= self.search_max_bounds[1]):
                print(f"A* Начальный узел {self.start_node_pos} находится вне эффективных границ поиска.")
            return None

        if not self._is_valid_node_pos(self.goal_node_pos):
            print(f"A* КРИТИЧЕСКАЯ ОШИБКА: Целевой узел {self.goal_node_pos} недействителен!")
            if not (self.search_min_bounds[0] <= self.goal_node_pos[0] <= self.search_max_bounds[0] and \
                    self.search_min_bounds[1] <= self.goal_node_pos[1] <= self.search_max_bounds[1]):
                print(f"A* Целевой узел {self.goal_node_pos} находится вне эффективных границ поиска.")
            return None
        
        # --- Отладка: Проверка соседей целевого узла (Закомментировано) ---
        # print(f"A* ОТЛАДКА: Проверка соседей целевого узла {self.goal_node_pos}")
        # goal_neighbors = self._get_neighbors(self.goal_node_pos)
        # valid_goal_neighbors_count = 0
        # if not goal_neighbors:
        #     print(f"A* ОТЛАДКА: Целевой узел {self.goal_node_pos} НЕ имеет соседей, сгенерированных _get_neighbors.")
        # else:
        #     for gn_idx, gn_pos in enumerate(goal_neighbors):
        #         is_gn_valid = self._is_valid_node_pos(gn_pos)
        #         if is_gn_valid:
        #             valid_goal_neighbors_count += 1
        #         print(f"A* ОТЛАДКА: Сосед_Цели {gn_idx + 1} в {gn_pos} (Кортеж: {self._get_position_tuple(gn_pos)}): Действителен = {is_gn_valid}")
        #     if valid_goal_neighbors_count == 0:
        #         print(f"A* КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Целевой узел {self.goal_node_pos} ДЕЙСТВИТЕЛЕН, но ВСЕ его потенциальные соседи НЕДЕЙСТВИТЕЛЬНЫ. Цель изолирована!")
        #     else:
        #         print(f"A* ОТЛАДКА: Целевой узел {self.goal_node_pos} имеет {valid_goal_neighbors_count} действительных соседей из {len(goal_neighbors)} потенциальных соседей.")
        # --- Конец Отладки ---

        start_node = Node(self.start_node_pos)
        start_node.g = 0
        start_node.h = self._heuristic(self.start_node_pos, self.goal_node_pos)
        start_node.f = start_node.g + start_node.h
        
        heapq.heappush(self.open_set, start_node)
        self.open_set_dict[self._get_position_tuple(start_node.position)] = start_node

        processed_nodes_count = 0 # Счетчик для отладки

        while self.open_set:
            current_node = heapq.heappop(self.open_set) # Извлекаем узел с наименьшим f
            current_pos_tuple = self._get_position_tuple(current_node.position)

            # 1. Если узел уже в closed_set, пропускаем его
            if current_pos_tuple in self.closed_set:
                continue
            
            # 2. Добавляем текущий узел в closed_set, так как он сейчас будет обработан
            self.closed_set.add(current_pos_tuple)

            # 3. Корректное удаление из open_set_dict
            #    Удаляем только если это был тот самый экземпляр узла, который отслеживался для данной позиции.
            #    Это важно, если для одной позиции в кучу могли попасть разные узлы (старый и новый пути).
            node_instance_in_dict = self.open_set_dict.get(current_pos_tuple)
            if node_instance_in_dict is current_node: # Сравниваем именно экземпляры объектов
                del self.open_set_dict[current_pos_tuple]
            # Если current_node - это устаревшая версия из кучи, а в словаре уже более новый (лучший)
            # узел для этой позиции, то запись в словаре для нового узла останется.

            processed_nodes_count += 1 # Увеличиваем счетчик обработанных (развернутых) узлов
            if processed_nodes_count % 500 == 0:
                print(f"A*: Обработано {processed_nodes_count} узлов. Размер открытого множества: {len(self.open_set)}")

            # 4. Проверка достижения цели
            if current_pos_tuple == self.goal_node_pos_tuple:
                end_time = time.time() # <--- Засекаем время окончания (путь найден)
                print(f"A*: Цель достигнута! Обработано {processed_nodes_count} узлов.")
                print(f"A*: Время поиска пути: {end_time - start_time:.4f} секунд.") # <--- Выводим время
                return self._reconstruct_path(current_node)

            # 5. Генерация и обработка соседей (остальная часть цикла остается без изменений)
            for neighbor_pos_np in self._get_neighbors(current_node.position):
                neighbor_pos_tuple = self._get_position_tuple(neighbor_pos_np)
                if neighbor_pos_tuple in self.closed_set:
                    continue

                tentative_g_score = current_node.g + np.linalg.norm(neighbor_pos_np - current_node.position)

                # Проверяем, есть ли этот сосед уже в open_set_dict
                existing_neighbor_node = self.open_set_dict.get(neighbor_pos_tuple)

                if existing_neighbor_node is not None and tentative_g_score >= existing_neighbor_node.g:
                    continue # Уже есть лучший или такой же путь к этому соседу
                
                # Если путь лучше или соседа нет в open_set, создаем/обновляем и добавляем в кучу
                neighbor_node = Node(neighbor_pos_np, parent=current_node)
                neighbor_node.g = tentative_g_score
                neighbor_node.h = self._heuristic(neighbor_pos_np, self.goal_node_pos)
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                
                heapq.heappush(self.open_set, neighbor_node)
                self.open_set_dict[neighbor_pos_tuple] = neighbor_node # Добавляем или обновляем в словаре
        
        end_time = time.time() # <--- Засекаем время окончания (путь НЕ найден)
        print(f"A*: Не удалось найти путь. Обработано {processed_nodes_count} узлов.")
        print(f"A*: Время поиска пути: {end_time - start_time:.4f} секунд.") # <--- Выводим время
        return None

if __name__ == '__main__':
    print("Запуск примера алгоритма A*...")
    parser = MazeParser() # Использует MAZE_XML_PATH из конфига
    walls = parser.get_walls()
    min_bounds, max_bounds = parser.get_world_bounds()

    if not walls:
        print("Стены не распознаны, запуск примера A* невозможен.")
    else:
        print(f"A* Старт: {ROBOT_START_POS_2D}, Цель: {TARGET_POS_2D}, Разрешение: {GRID_RESOLUTION}")
        astar_planner = AStar(walls, min_bounds, max_bounds, 
                                ROBOT_START_POS_2D, TARGET_POS_2D, 
                                GRID_RESOLUTION, robot_radius=0.15 # Убедимся, что здесь 0.1
                                )
        
        path = astar_planner.find_path()

        visualizer = TrajectoryVisualizer(walls, min_bounds, max_bounds, 
                                          astar_planner.start_node_pos, # Используем дискретизированные старт/цель для виз
                                          astar_planner.goal_node_pos)
        if path:
            print(f"A*: Путь найден, {len(path)} точек.")
            visualizer.draw_path(path, label='A* Path')
        else:
            print("A*: Путь не найден.")
        
        # Опционально отрисовать сетку (может быть очень плотной)
        # visualizer.draw_grid(astar_planner.grid_min_x, astar_planner.grid_min_y, 
        #                     astar_planner.grid_max_x, astar_planner.grid_max_y, GRID_RESOLUTION)
        visualizer.show()
    print("Пример алгоритма A* завершен.") 