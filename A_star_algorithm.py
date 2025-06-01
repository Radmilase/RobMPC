"""
Implementation of the A* pathfinding algorithm for the maze.
"""
import heapq
import numpy as np
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
    def __init__(self, walls, world_min_bounds, world_max_bounds, start_pos, goal_pos, resolution, robot_radius=0.0):
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
        self.world_min_bounds = world_min_bounds
        self.world_max_bounds = world_max_bounds
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        self.resolution = resolution
        self.robot_radius = robot_radius # For inflating walls or checking node validity

        # Discretize start and goal positions to grid cell centers
        self.start_node_pos = self._discretize_point(self.start_pos)
        self.goal_node_pos = self._discretize_point(self.goal_pos)
        
        self.open_set = []  # Priority queue (min-heap)
        self.closed_set = set() # Set of visited node positions (as tuples for hashability)

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
        # Check world bounds (using continuous world for this check is fine)
        # if not (self.world_min_bounds[0] <= node_pos[0] <= self.world_max_bounds[0] and \
        #         self.world_min_bounds[1] <= node_pos[1] <= self.world_max_bounds[1]):
        #     return False # Not strictly needed if neighbors are generated carefully

        # Check for collisions with walls (considering robot radius)
        for wall in self.walls:
            # Check if the square cell (node_pos +- resolution/2) overlaps the wall (inflated by robot_radius)
            # Or, simpler: check if point node_pos is inside an inflated wall
            # Wall corners are (min_x, min_y) and (max_x, max_y)
            # Inflate wall by robot_radius for collision checking point
            if (wall.min_x - self.robot_radius < node_pos[0] < wall.max_x + self.robot_radius and
                wall.min_y - self.robot_radius < node_pos[1] < wall.max_y + self.robot_radius):
                # More precise check for point in rectangle (for axis-aligned boxes)
                # Check if the center of the cell is within the wall boundaries
                if (node_pos[0] >= wall.min_x - self.robot_radius and node_pos[0] <= wall.max_x + self.robot_radius and
                    node_pos[1] >= wall.min_y - self.robot_radius and node_pos[1] <= wall.max_y + self.robot_radius):
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
        start_node = Node(self.start_node_pos)
        start_node.g = 0
        start_node.h = self._heuristic(self.start_node_pos, self.goal_node_pos)
        start_node.f = start_node.g + start_node.h
        
        heapq.heappush(self.open_set, start_node)
        # open_set_dict для быстрого доступа к узлам в куче по их позиции
        open_set_dict = {tuple(start_node.position): start_node}

        processed_nodes_count = 0 # Счетчик для отладки

        while self.open_set:
            current_node = heapq.heappop(self.open_set)
            # Удаляем из словаря, так как он извлечен из кучи
            if tuple(current_node.position) in open_set_dict: # Может быть уже удален, если были дубликаты с разными f в куче
                del open_set_dict[tuple(current_node.position)]

            processed_nodes_count += 1
            if processed_nodes_count % 500 == 0:
                print(f"A*: Processed {processed_nodes_count} nodes. Open set size: {len(self.open_set)}")

            if np.array_equal(current_node.position, self.goal_node_pos):
                print(f"A*: Goal reached! Processed {processed_nodes_count} nodes.")
                return self._reconstruct_path(current_node)

            self.closed_set.add(tuple(current_node.position))

            for neighbor_pos_np in self._get_neighbors(current_node.position):
                neighbor_pos_tuple = tuple(neighbor_pos_np)
                if neighbor_pos_tuple in self.closed_set:
                    continue

                tentative_g_score = current_node.g + np.linalg.norm(neighbor_pos_np - current_node.position)

                # Проверяем, есть ли этот сосед уже в open_set_dict
                existing_neighbor_node = open_set_dict.get(neighbor_pos_tuple)

                if existing_neighbor_node is not None and tentative_g_score >= existing_neighbor_node.g:
                    continue # Уже есть лучший или такой же путь к этому соседу
                
                # Если путь лучше или соседа нет в open_set, создаем/обновляем и добавляем в кучу
                neighbor_node = Node(neighbor_pos_np, parent=current_node)
                neighbor_node.g = tentative_g_score
                neighbor_node.h = self._heuristic(neighbor_pos_np, self.goal_node_pos)
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                
                heapq.heappush(self.open_set, neighbor_node)
                open_set_dict[neighbor_pos_tuple] = neighbor_node # Добавляем или обновляем в словаре
        
        print(f"A*: Could not find a path. Processed {processed_nodes_count} nodes.")
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
                                GRID_RESOLUTION, robot_radius=0.05 # VIS_ROBOT_RADIUS from config can be used
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