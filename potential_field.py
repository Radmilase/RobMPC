"""
Implementation of the Potential Field path planning algorithm.
"""
import numpy as np
import matplotlib.pyplot as plt # For direct visualization if needed
from maze_config import (
    ROBOT_START_POS_2D, TARGET_POS_2D, MAZE_XML_PATH,
    VIS_PATH_COLOR, VIS_WALL_COLOR, VIS_START_MARKER_COLOR, VIS_GOAL_MARKER_COLOR,
    VIS_FIGURE_SIZE, VIS_ROBOT_RADIUS
)
from maze_parser import MazeParser
from trajectory_visualizer import TrajectoryVisualizer # For standardized visualization

class PotentialFieldPlanner:
    """
    Calculates a path using the potential field method.
    An attractive potential pulls towards the goal, and repulsive potentials push away from obstacles.
    """
    def __init__(self, walls, start_pos, goal_pos, 
                 k_att=1.0, k_rep=10.0, rep_dist_influence=0.5, 
                 step_size=0.05, max_steps=1000, goal_threshold=0.1, robot_radius=0.0):
        """
        Args:
            walls (list): List of Wall objects from MazeParser.
            start_pos (np.array): (x,y) starting position.
            goal_pos (np.array): (x,y) goal position.
            k_att (float): Gain for the attractive potential.
            k_rep (float): Gain for the repulsive potential.
            rep_dist_influence (float): Distance up to which obstacles exert repulsive force.
            step_size (float): Size of each step taken along the gradient.
            max_steps (int): Maximum number of steps to take.
            goal_threshold (float): Distance to goal to consider it reached.
            robot_radius (float): Radius of the robot for collision checking/inflation.
        """
        self.walls = walls
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        
        self.k_att = k_att
        self.k_rep = k_rep
        self.dist_influence_sq = rep_dist_influence**2 # Store squared for efficiency
        self.rep_dist_influence = rep_dist_influence
        
        self.step_size = step_size
        self.max_steps = max_steps
        self.goal_threshold_sq = goal_threshold**2
        self.robot_radius = robot_radius
        
        self.path = [self.start_pos.copy()]

    def _attractive_force(self, current_pos):
        """Calculates the attractive force towards the goal."""
        return self.k_att * (self.goal_pos - current_pos)

    def _repulsive_force(self, current_pos):
        """Calculates the total repulsive force from all relevant obstacles."""
        total_rep_force = np.zeros(2)
        for wall in self.walls:
            # Find the closest point on the (inflated) wall to current_pos
            # For a simple AABB wall, this can be complex. 
            # A simpler approach is to find the distance to the wall's boundary segments
            # or just the distance to the center of the wall if the point is near it.
            # More robust: distance to each of the 4 (inflated) line segments of the wall box.
            
            # Simplified: distance to wall center and check if within influence radius of the wall's extent
            # This isn't perfect as it treats wall as a point for direction, but often works for grid worlds.
            # A better way for rectangular obstacles involves gradients perpendicular to surfaces.

            # For each wall, calculate the closest point (cx, cy) on its boundary to current_pos
            # Inflate wall by robot_radius for force calculation
            min_x, max_x = wall.min_x - self.robot_radius, wall.max_x + self.robot_radius
            min_y, max_y = wall.min_y - self.robot_radius, wall.max_y + self.robot_radius

            closest_point_on_wall = np.array([
                np.clip(current_pos[0], min_x, max_x),
                np.clip(current_pos[1], min_y, max_y)
            ])
            
            vec_to_current = current_pos - closest_point_on_wall
            dist_sq = np.sum(vec_to_current**2)

            if dist_sq < self.dist_influence_sq and dist_sq > 1e-6: # If within influence and not on the point
                dist = np.sqrt(dist_sq)
                # Repulsive force magnitude is inversely proportional to distance
                # The gradient direction is away from the closest point on the obstacle
                force_magnitude = self.k_rep * ((1.0 / dist) - (1.0 / self.rep_dist_influence)) * (1.0 / dist**2)
                force_direction = vec_to_current / dist
                total_rep_force += force_magnitude * force_direction
                
        return total_rep_force

    def find_path(self):
        """Generates the path using gradient descent on the potential field."""
        current_pos = self.start_pos.copy()
        
        for step_num in range(self.max_steps):
            if np.sum((current_pos - self.goal_pos)**2) < self.goal_threshold_sq:
                print(f"PotentialField: Goal reached in {step_num} steps.")
                self.path.append(self.goal_pos.copy()) # Ensure goal is the last point
                return self.path
            
            f_att = self._attractive_force(current_pos)
            f_rep = self._repulsive_force(current_pos)
            
            total_force = f_att + f_rep
            
            # Normalize force to get direction, then step
            if np.linalg.norm(total_force) > 1e-6:
                move_direction = total_force / np.linalg.norm(total_force)
                current_pos += self.step_size * move_direction
                self.path.append(current_pos.copy())
            else:
                # Stuck in local minimum or no force
                print(f"PotentialField: Stuck at step {step_num} (zero total force). Pos: {current_pos}")
                # Add a small random perturbation to try to escape (optional, can lead to oscillations)
                # current_pos += np.random.rand(2) * self.step_size * 0.1 - (self.step_size * 0.05)
                # For now, just break if stuck
                break 

        print(f"PotentialField: Max steps ({self.max_steps}) reached or got stuck. Path may be incomplete.")
        return self.path

if __name__ == '__main__':
    print("Running Potential Field Algorithm example...")
    parser = MazeParser(MAZE_XML_PATH)
    walls = parser.get_walls()
    min_b, max_b = parser.get_world_bounds()

    if not walls:
        print("No walls parsed, cannot run Potential Field example.")
    else:
        # Parameters for Potential Field (these may need significant tuning)
        K_ATT = 1.0
        K_REP = 2.0  # Repulsive force gain
        REP_DIST_INFLUENCE = 0.7 # How far obstacles 'push'
        STEP_SIZE = 0.05
        MAX_STEPS = 2000
        GOAL_THRESHOLD = 0.1
        # Use VIS_ROBOT_RADIUS from config if available, otherwise a default
        robot_radius_pf = VIS_ROBOT_RADIUS if 'VIS_ROBOT_RADIUS' in globals() else 0.05 

        print(f"PF Start: {ROBOT_START_POS_2D}, Goal: {TARGET_POS_2D}")
        print(f"Params: k_att={K_ATT}, k_rep={K_REP}, rep_dist={REP_DIST_INFLUENCE}, step={STEP_SIZE}")

        pf_planner = PotentialFieldPlanner(walls, ROBOT_START_POS_2D, TARGET_POS_2D,
                                           k_att=K_ATT, k_rep=K_REP, 
                                           rep_dist_influence=REP_DIST_INFLUENCE,
                                           step_size=STEP_SIZE, max_steps=MAX_STEPS, 
                                           goal_threshold=GOAL_THRESHOLD,
                                           robot_radius=robot_radius_pf)
        
        path = pf_planner.find_path()

        visualizer = TrajectoryVisualizer(walls, min_b, max_b, ROBOT_START_POS_2D, TARGET_POS_2D)
        if path and len(path) > 1:
            print(f"PotentialField: Path found with {len(path)} points.")
            visualizer.draw_path(path, label='Potential Field Path')
        elif path and len(path) == 1:
            print("PotentialField: Path consists of only the start point.")
            visualizer.draw_path(path, label='Potential Field Path (Start Only)')
        else:
            print("PotentialField: No path found or path is empty.")
        
        visualizer.show()
        print("Potential Field Algorithm example finished.") 