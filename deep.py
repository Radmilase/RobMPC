"""
Реализация алгоритма Поиска в Глубину (DFS) для решения лабиринта.
"""
import numpy as np
import time
from maze_config import ROBOT_START_POS_2D, TARGET_POS_2D, GRID_RESOLUTION, DFS_GRID_RESOLUTION
from maze_parser import MazeParser
from trajectory_visualizer import TrajectoryVisualizer

class DFSNode:
    """Представляет узел в поиске DFS."""
    def __init__(self, position, parent=None):
        self.position = np.array(position)  # (x, y) координаты
        self.parent = parent

    def __eq__(self, other):
        return np.array_equal(self.position, other.position)

    def __hash__(self):
        return hash(tuple(self.position))

class DepthFirstSearch:
    """Реализует алгоритм поиска пути DFS."""
    DECIMAL_PRECISION = 5 # Для округления позиций

    def _get_position_tuple(self, pos_np):
        """Округляет массив numpy с позицией и конвертирует в хешируемый кортеж."""
        return tuple(np.round(pos_np, self.DECIMAL_PRECISION))

    def __init__(self, walls, world_min_bounds, world_max_bounds, start_pos, goal_pos, resolution, robot_radius=0.1):
        self.walls = walls
        self.world_min_bounds = world_min_bounds
        self.world_max_bounds = world_max_bounds
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        self.resolution = resolution
        self.robot_radius = robot_radius

        self.search_min_bounds = self.world_min_bounds - self.resolution / 2.01
        self.search_max_bounds = self.world_max_bounds + self.resolution / 2.01

        self.start_node_pos = self._discretize_point(self.start_pos)
        self.goal_node_pos = self._discretize_point(self.goal_pos)
        self.goal_node_pos_tuple = self._get_position_tuple(self.goal_node_pos)
        
        self.visited = set() # Множество посещенных позиций узлов (в виде кортежей для хешируемости)

    def _discretize_point(self, point):
        """Привязывает непрерывную точку к центру ее ячейки сетки."""
        snapped = np.round(point / self.resolution) * self.resolution
        return snapped

    def _is_valid_node_pos(self, node_pos):
        """Проверяет, находится ли позиция узла в границах и не сталкивается ли со стеной."""
        if not (self.search_min_bounds[0] <= node_pos[0] <= self.search_max_bounds[0] and \
                self.search_min_bounds[1] <= node_pos[1] <= self.search_max_bounds[1]):
            return False

        for wall in self.walls:
            inflated_min_x = wall.min_x - self.robot_radius
            inflated_max_x = wall.max_x + self.robot_radius
            inflated_min_y = wall.min_y - self.robot_radius
            inflated_max_y = wall.max_y + self.robot_radius

            if (node_pos[0] >= inflated_min_x and node_pos[0] <= inflated_max_x and
                node_pos[1] >= inflated_min_y and node_pos[1] <= inflated_max_y):
                return False
        return True

    def _get_neighbors(self, current_node_pos):
        """Генерирует действительные позиции соседних узлов (8-связность)."""
        neighbors = []
        # Порядок соседей может влиять на путь DFS. Попробуем последовательный порядок.
        # (Вверх, Вниз, Влево, Вправо, а затем диагонали)
        # Этот конкретный порядок может быть не оптимальным для всех лабиринтов, но обеспечивает согласованность.
        moves = [
            (0, self.resolution), (0, -self.resolution), (-self.resolution, 0), (self.resolution, 0), # Кардинальные
            (-self.resolution, self.resolution), (self.resolution, self.resolution), # Диагонали
            (-self.resolution, -self.resolution), (self.resolution, -self.resolution) 
        ]
        for dx, dy in moves:
            neighbor_pos = current_node_pos + np.array([dx, dy])
            if self._is_valid_node_pos(neighbor_pos):
                neighbors.append(neighbor_pos)
        return neighbors

    def _reconstruct_path(self, current_node):
        """Восстанавливает путь от целевого узла обратно к начальному."""
        path = []
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        return path[::-1]

    def find_path(self):
        """Выполняет алгоритм поиска DFS."""
        start_time = time.time()
        print(f"DFS Инициализация. Старт: {self.start_pos}, Цель: {self.goal_pos}")
        print(f"DFS Дискретизированный начальный узел: {self.start_node_pos}, Дискретизированный целевой узел: {self.goal_node_pos}")

        if not self._is_valid_node_pos(self.start_node_pos):
            print(f"DFS КРИТИЧЕСКАЯ ОШИБКА: Начальный узел {self.start_node_pos} недействителен!")
            return None
        if not self._is_valid_node_pos(self.goal_node_pos):
            print(f"DFS КРИТИЧЕСКАЯ ОШИБКА: Целевой узел {self.goal_node_pos} недействителен!")
            return None

        start_node = DFSNode(self.start_node_pos)
        
        # Стек для DFS
        stack = [start_node]
        # Отслеживаем узлы, находящиеся в данный момент в стеке рекурсии (или его итеративном эквиваленте)
        # для предотвращения циклов при реконструкции пути.
        # Это отличается от self.visited, который помечает узлы, чьи поддеревья полностью исследованы.
        path_so_far_nodes = {self._get_position_tuple(start_node.position): start_node}


        processed_nodes_count = 0

        while stack:
            current_node = stack.pop()
            current_pos_tuple = self._get_position_tuple(current_node.position)

            # Если мы уже полностью исследовали поддерево этого узла, пропускаем.
            if current_pos_tuple in self.visited:
                continue
            
            self.visited.add(current_pos_tuple)
            processed_nodes_count += 1
            
            if processed_nodes_count % 500 == 0:
                 print(f"DFS: Обработано {processed_nodes_count} узлов. Размер стека: {len(stack)}")

            if current_pos_tuple == self.goal_node_pos_tuple:
                end_time = time.time()
                print(f"DFS: Цель достигнута! Обработано {processed_nodes_count} узлов.")
                print(f"DFS: Время поиска пути: {end_time - start_time:.4f} секунд.")
                return self._reconstruct_path(current_node)

            # Исследуем соседей
            for neighbor_pos_np in self._get_neighbors(current_node.position):
                neighbor_pos_tuple = self._get_position_tuple(neighbor_pos_np)
                
                # Если сосед не посещен и не находится в данный момент на пути, строящемся к этой точке
                if neighbor_pos_tuple not in self.visited :
                    neighbor_node = DFSNode(neighbor_pos_np, parent=current_node)
                    stack.append(neighbor_node)
        
        end_time = time.time()
        print(f"DFS: Путь не найден. Обработано {processed_nodes_count} узлов.")
        print(f"DFS: Время поиска: {end_time - start_time:.4f} секунд.")
        return None

if __name__ == '__main__':
    print("Запуск примера алгоритма DFS...")
    parser = MazeParser() 
    walls = parser.get_walls()
    min_bounds, max_bounds = parser.get_world_bounds()

    if not walls:
        print("Стены не распознаны, запуск примера DFS невозможен.")
    else:
        print(f"DFS Старт: {ROBOT_START_POS_2D}, Цель: {TARGET_POS_2D}, Разрешение: {DFS_GRID_RESOLUTION}")
        dfs_planner = DepthFirstSearch(walls, min_bounds, max_bounds,
                                 ROBOT_START_POS_2D, TARGET_POS_2D,
                                 DFS_GRID_RESOLUTION, robot_radius=0.1)
        
        path = dfs_planner.find_path()

        # Используем дискретизированные старт/цель для визуализации, аналогично примеру A*
        visualizer = TrajectoryVisualizer(walls, min_bounds, max_bounds,
                                          dfs_planner.start_node_pos, 
                                          dfs_planner.goal_node_pos)
        if path:
            print(f"DFS: Путь найден, {len(path)} точек.")
            visualizer.draw_path(path, label='DFS Path', color='blue') # Другой цвет для DFS
        else:
            print("DFS: Путь не найден.")
        
        visualizer.show()
    print("Пример алгоритма DFS завершен.") 