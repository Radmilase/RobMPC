"""
Visualizes the maze and the calculated trajectory using Matplotlib.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from maze_config import (
    VIS_PATH_COLOR, VIS_WALL_COLOR, VIS_GRID_COLOR,
    VIS_START_MARKER_COLOR, VIS_GOAL_MARKER_COLOR,
    VIS_FIGURE_SIZE, GRID_RESOLUTION
)

class TrajectoryVisualizer:
    """Handles plotting the maze walls, grid, and path."""

    def __init__(self, walls, world_min_bounds, world_max_bounds, start_pos=None, goal_pos=None):
        """
        Args:
            walls (list): List of Wall objects from MazeParser.
            world_min_bounds (np.array): Min (x, y) of the world.
            world_max_bounds (np.array): Max (x, y) of the world.
            start_pos (np.array, optional): Start position (x, y).
            goal_pos (np.array, optional): Goal position (x, y).
        """
        self.walls = walls
        self.world_min_bounds = world_min_bounds
        self.world_max_bounds = world_max_bounds
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        
        self.fig, self.ax = plt.subplots(figsize=VIS_FIGURE_SIZE)
        self._setup_plot()

    def _setup_plot(self):
        """Sets up the basic plot properties, axes, and draws walls."""
        self.ax.set_aspect('equal', adjustable='box')
        # Add a small margin to the world bounds for better visualization
        margin = GRID_RESOLUTION * 5 
        self.ax.set_xlim(self.world_min_bounds[0] - margin, self.world_max_bounds[0] + margin)
        self.ax.set_ylim(self.world_min_bounds[1] - margin, self.world_max_bounds[1] + margin)
        self.ax.set_xlabel("X coordinate")
        self.ax.set_ylabel("Y coordinate")
        self.ax.set_title("Maze Trajectory Visualization")
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=VIS_GRID_COLOR)
        
        # Draw walls
        for wall in self.walls:
            (x, y), width, height = wall.get_patch_params()
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=VIS_WALL_COLOR, facecolor=VIS_WALL_COLOR, alpha=0.7)
            self.ax.add_patch(rect)

        # Draw Start and Goal markers
        if self.start_pos is not None:
            self.ax.plot(self.start_pos[0], self.start_pos[1], 'o', markersize=10, color=VIS_START_MARKER_COLOR, label='Start')
        if self.goal_pos is not None:
            self.ax.plot(self.goal_pos[0], self.goal_pos[1], 'x', markersize=10, markeredgewidth=3, color=VIS_GOAL_MARKER_COLOR, label='Goal')
        
        if self.start_pos is not None or self.goal_pos is not None:
            self.ax.legend()

    def draw_grid(self, grid_min_x, grid_min_y, grid_max_x, grid_max_y, resolution):
        """Draws the A* search grid (optional)."""
        # Vertical lines
        for x_coord in np.arange(grid_min_x, grid_max_x + resolution, resolution):
            self.ax.axvline(x_coord, color=VIS_GRID_COLOR, linestyle=':', linewidth=0.3)
        # Horizontal lines
        for y_coord in np.arange(grid_min_y, grid_max_y + resolution, resolution):
            self.ax.axhline(y_coord, color=VIS_GRID_COLOR, linestyle=':', linewidth=0.3)

    def draw_path(self, path_points, color=VIS_PATH_COLOR, label='Path'):
        """Draws a given path (list of (x, y) tuples or np.array)."""
        if not path_points:
            print("Visualizer: No path points to draw.")
            return
        
        path_np = np.array(path_points)
        self.ax.plot(path_np[:, 0], path_np[:, 1], color=color, linewidth=2, marker='.', markersize=4, label=label)
        self.ax.legend() # Update legend if new items are added

    def show(self):
        """Displays the plot."""
        plt.show()

if __name__ == '__main__':
    # Example Usage (requires maze_parser.py and maze_config.py)
    from maze_parser import MazeParser
    from maze_config import ROBOT_START_POS_2D, TARGET_POS_2D

    print("Running TrajectoryVisualizer example...")
    parser = MazeParser() # Uses MAZE_XML_PATH from config
    walls = parser.get_walls()
    min_bounds, max_bounds = parser.get_world_bounds()

    if not walls:
        print("No walls parsed, cannot run visualizer example.")
    else:
        visualizer = TrajectoryVisualizer(walls, min_bounds, max_bounds, ROBOT_START_POS_2D, TARGET_POS_2D)
        
        # Example path (replace with actual A* or Potential Field path)
        example_path = [
            ROBOT_START_POS_2D,
            np.array([ROBOT_START_POS_2D[0] + 0.5, ROBOT_START_POS_2D[1] - 0.2]),
            np.array([TARGET_POS_2D[0] - 0.3, TARGET_POS_2D[1] + 0.5]),
            TARGET_POS_2D
        ]
        visualizer.draw_path(example_path, color='green', label='Example Path')
        
        # visualizer.draw_grid(min_bounds[0], min_bounds[1], max_bounds[0], max_bounds[1], GRID_RESOLUTION)
        visualizer.show()
        print("TrajectoryVisualizer example finished.") 