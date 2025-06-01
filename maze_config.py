"""
Configuration file for the maze navigation project.
Stores paths, coordinates, and other shared parameters.
"""
import numpy as np

# --- Maze Configuration ---
MAZE_XML_PATH = "model/scene_2.xml" # Path to the MuJoCo XML file describing the maze

# --- Robot and Goal Configuration (2D coordinates) ---
# These are approximate and might need fine-tuning based on the visualizer
ROBOT_START_POS_2D = np.array([-1.9, 1.3])
# TARGET_POS_2D = np.array([0.75, -1.75]) # A point in a more central chamber, slightly shifted
# Let's pick a more accessible target for initial A* testing, within the upper part of the maze
# Based on scene_2.xml, a point like (-0.5, 0.0) seems reachable from start.
# Or for a longer path:
TARGET_POS_2D = np.array([0.8, -1.8]) # The desired final target

# --- A* Algorithm Configuration ---
GRID_RESOLUTION = 0.01  # Resolution of the grid for A* (e.g., 0.1 units per cell)
HEURISTIC_WEIGHT = 1.0 # Weight for the heuristic in A*

# --- Potential Field Algorithm Configuration ---
# (To be added later)

# --- Visualization Configuration ---
VIS_PATH_COLOR = 'red'
VIS_WALL_COLOR = 'blue'
VIS_GRID_COLOR = 'grey'
VIS_START_MARKER_COLOR = 'green'
VIS_GOAL_MARKER_COLOR = 'magenta'
VIS_FIGURE_SIZE = (10, 10) # Inches
VIS_ROBOT_RADIUS = 0.05 # Approximate radius of the robot for collision checking (if needed) 