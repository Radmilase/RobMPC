"""
Implementation of the A* pathfinding algorithm for the maze.
"""
import heapq
import numpy as np
import time # <--- Добавляем импорт time
from maze_config import ROBOT_START_POS_2D, TARGET_POS_2D, GRID_RESOLUTION, HEURISTIC_WEIGHT
from maze_parser import MazeParser
from trajectory_visualizer import TrajectoryVisualizer # For testing

class Node:
    """Represents a node in the A* search grid."""
    def __init__(self, position, parent=None):
        self.position = np.array(position) # (x, y) coordinates
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost from current node to goal
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return np.array_equal(self.position, other.position)

    def __lt__(self, other):
        # heapq is a min-heap, so lower f-value is higher priority
        return self.f < other.f
    
    def __hash__(self):
        # Required for adding nodes to a set
        return hash(tuple(self.position))

class AStar:
    """Implements the A* pathfinding algorithm."""
    DECIMAL_PRECISION = 5 # For rounding positions to handle floating point issues

    def _get_position_tuple(self, pos_np):
        """Rounds a numpy position array and converts to a hashable tuple."""
        return tuple(np.round(pos_np, self.DECIMAL_PRECISION))

    def __init__(self, walls, world_min_bounds, world_max_bounds, start_pos, goal_pos, resolution, robot_radius=1.0):
        """
        Args:
            walls (list): List of Wall objects.
            world_min_bounds (np.array): Min (x, y) of the world.
            world_max_bounds (np.array): Max (x, y) of the world.
            start_pos (np.array): Start position (x, y).
            goal_pos (np.array): Goal position (x, y).
            resolution (float): Grid cell size.
            robot_radius (float): Radius of the robot for collision checking (inflation).
        """
        self.walls = walls
        self.world_min_bounds = world_min_bounds # Keep original for reference or other uses
        self.world_max_bounds = world_max_bounds
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        self.resolution = resolution
        self.robot_radius = robot_radius

        # Effective bounds for search, slightly expanded to avoid issues at the very edge
        self.search_min_bounds = self.world_min_bounds - self.resolution / 2.01 # Expand by slightly less than half res to be safe
        self.search_max_bounds = self.world_max_bounds + self.resolution / 2.01

        self.start_node_pos = self._discretize_point(self.start_pos)
        self.goal_node_pos = self._discretize_point(self.goal_pos)
        self.goal_node_pos_tuple = self._get_position_tuple(self.goal_node_pos) # Store rounded tuple for goal check
        
        self.open_set = []  # Priority queue (min-heap)
        self.closed_set = set() # Set of visited node positions (as tuples for hashability)
        # open_set_dict maps rounded position tuples to Node objects in the open_set
        self.open_set_dict = {}

        self.grid_min_x, self.grid_min_y = self._discretize_point(world_min_bounds) - resolution # Expand slightly
        self.grid_max_x, self.grid_max_y = self._discretize_point(world_max_bounds) + resolution

    def _discretize_point(self, point):
        """Snaps a continuous point to the center of its grid cell."""
        # Snap to nearest grid line, then offset by half resolution to get cell center
        snapped = np.round(point / self.resolution) * self.resolution
        return snapped # For A*, often work with grid indices or centers. Let's use centers.

    def _heuristic(self, pos1, pos2):
        """Manhattan distance heuristic (can be changed to Euclidean)."""
        # return np.sum(np.abs(pos1 - pos2)) * HEURISTIC_WEIGHT # Manhattan
        return np.linalg.norm(pos1 - pos2) * HEURISTIC_WEIGHT # Euclidean

    def _is_valid_node_pos(self, node_pos):
        """Checks if a node position is within bounds and not colliding with a wall."""
        # Check effective search bounds
        if not (self.search_min_bounds[0] <= node_pos[0] <= self.search_max_bounds[0] and \
                self.search_min_bounds[1] <= node_pos[1] <= self.search_max_bounds[1]):
            # print(f"Node {node_pos} is outside effective search bounds: Min({self.search_min_bounds}), Max({self.search_max_bounds})") # DEBUG
            return False

        # Check for collisions with walls (considering robot radius)
        for wall in self.walls:
            inflated_min_x = wall.min_x - self.robot_radius
            inflated_max_x = wall.max_x + self.robot_radius
            inflated_min_y = wall.min_y - self.robot_radius
            inflated_max_y = wall.max_y + self.robot_radius

            if (node_pos[0] >= inflated_min_x and node_pos[0] <= inflated_max_x and
                node_pos[1] >= inflated_min_y and node_pos[1] <= inflated_max_y):
                # If the node's center is within or on the boundary of the inflated wall, it's a collision.
                return False # Collision
        return True

    def _get_neighbors(self, current_node_pos):
        """Generates valid neighbor node positions (8-connectivity)."""
        neighbors = []
        for dx in [-self.resolution, 0, self.resolution]:
            for dy in [-self.resolution, 0, self.resolution]:
                if dx == 0 and dy == 0:
                    continue # Skip current node itself
                
                neighbor_pos = current_node_pos + np.array([dx, dy])
                # Discretize to be sure it aligns with grid, though adding resolution should keep it aligned
                # neighbor_pos = self._discretize_point(neighbor_pos) 
                
                if self._is_valid_node_pos(neighbor_pos):
                    neighbors.append(neighbor_pos)
        return neighbors

    def _reconstruct_path(self, current_node):
        """Reconstructs the path from the goal node back to the start."""
        path = []
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        return path[::-1] # Reverse to get path from start to goal

    def find_path(self):
        """Executes the A* search algorithm."""
        start_time = time.time() # <--- Засекаем время начала

        print(f"A* Initializing. Start: {self.start_pos}, Goal: {self.goal_pos}")
        print(f"A* Discretized Start Node: {self.start_node_pos}, Discretized Goal Node: {self.goal_node_pos}")
        # print(f"A* World Min/Max Bounds (raw): {self.world_min_bounds} / {self.world_max_bounds}") # Optional: Keep if useful
        # print(f"A* Effective Search Min/Max Bounds: {self.search_min_bounds} / {self.search_max_bounds}") # Optional: Keep if useful

        if not self._is_valid_node_pos(self.start_node_pos):
            print(f"A* CRITICAL: Start node {self.start_node_pos} is not valid!")
            if not (self.search_min_bounds[0] <= self.start_node_pos[0] <= self.search_max_bounds[0] and \
                    self.search_min_bounds[1] <= self.start_node_pos[1] <= self.search_max_bounds[1]):
                print(f"A* Start node {self.start_node_pos} is outside effective search bounds.")
            return None

        if not self._is_valid_node_pos(self.goal_node_pos):
            print(f"A* CRITICAL: Goal node {self.goal_node_pos} is not valid!")
            if not (self.search_min_bounds[0] <= self.goal_node_pos[0] <= self.search_max_bounds[0] and \
                    self.search_min_bounds[1] <= self.goal_node_pos[1] <= self.search_max_bounds[1]):
                print(f"A* Goal node {self.goal_node_pos} is outside effective search bounds.")
            return None
        
        # --- Debug: Check neighbors of the goal node (Commented out) ---
        # print(f"A* DEBUG: Checking neighbors of goal node {self.goal_node_pos}")
        # goal_neighbors = self._get_neighbors(self.goal_node_pos)
        # valid_goal_neighbors_count = 0
        # if not goal_neighbors:
        #     print(f"A* DEBUG: Goal node {self.goal_node_pos} has NO neighbors generated by _get_neighbors.")
        # else:
        #     for gn_idx, gn_pos in enumerate(goal_neighbors):
        #         is_gn_valid = self._is_valid_node_pos(gn_pos)
        #         if is_gn_valid:
        #             valid_goal_neighbors_count += 1
        #         print(f"A* DEBUG: Goal_Neighbor {gn_idx + 1} at {gn_pos} (Tuple: {self._get_position_tuple(gn_pos)}): Valid = {is_gn_valid}")
        #     if valid_goal_neighbors_count == 0:
        #         print(f"A* CRITICAL WARNING: Goal node {self.goal_node_pos} is VALID, but ALL its potential neighbors are INVALID. The goal is isolated!")
        #     else:
        #         print(f"A* DEBUG: Goal node {self.goal_node_pos} has {valid_goal_neighbors_count} valid neighbors out of {len(goal_neighbors)} potential neighbors.")
        # --- End Debug ---

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
                print(f"A*: Processed {processed_nodes_count} nodes. Open set size: {len(self.open_set)}")

            # 4. Проверка достижения цели
            if current_pos_tuple == self.goal_node_pos_tuple:
                end_time = time.time() # <--- Засекаем время окончания (путь найден)
                print(f"A*: Goal reached! Processed {processed_nodes_count} nodes.")
                print(f"A*: Pathfinding time: {end_time - start_time:.4f} seconds.") # <--- Выводим время
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
        print(f"A*: Could not find a path. Processed {processed_nodes_count} nodes.")
        print(f"A*: Pathfinding time: {end_time - start_time:.4f} seconds.") # <--- Выводим время
        return None

if __name__ == '__main__':
    print("Running A* Algorithm example...")
    parser = MazeParser() # Uses MAZE_XML_PATH from config
    walls = parser.get_walls()
    min_bounds, max_bounds = parser.get_world_bounds()

    if not walls:
        print("No walls parsed, cannot run A* example.")
    else:
        print(f"A* Start: {ROBOT_START_POS_2D}, Goal: {TARGET_POS_2D}, Resolution: {GRID_RESOLUTION}")
        astar_planner = AStar(walls, min_bounds, max_bounds, 
                                ROBOT_START_POS_2D, TARGET_POS_2D, 
                                GRID_RESOLUTION, robot_radius=0.1 # Убедимся, что здесь 0.1
                                )
        
        path = astar_planner.find_path()

        visualizer = TrajectoryVisualizer(walls, min_bounds, max_bounds, 
                                          astar_planner.start_node_pos, # Use discretized start/goal for viz
                                          astar_planner.goal_node_pos)
        if path:
            print(f"A*: Path found with {len(path)} points.")
            visualizer.draw_path(path, label='A* Path')
        else:
            print("A*: No path found.")
        
        # Optionally draw the grid (can be very dense)
        # visualizer.draw_grid(astar_planner.grid_min_x, astar_planner.grid_min_y, 
        #                     astar_planner.grid_max_x, astar_planner.grid_max_y, GRID_RESOLUTION)
        visualizer.show()
        print("A* Algorithm example finished.") 