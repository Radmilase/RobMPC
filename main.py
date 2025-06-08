import mujoco
from mujoco import viewer
import numpy as np

from A_star_algorithm import AStar
from deep import DepthFirstSearch
from maze_config import ROBOT_START_POS_2D, TARGET_POS_2D, GRID_RESOLUTION, DFS_GRID_RESOLUTION
from maze_parser import MazeParser

def get_trajectory():
    parser = MazeParser() # Uses MAZE_XML_PATH from config
    walls = parser.get_walls()
    min_bounds, max_bounds = parser.get_world_bounds()

    # astar_planner = AStar(walls, min_bounds, max_bounds, 
    #                     ROBOT_START_POS_2D, TARGET_POS_2D, 
    #                     GRID_RESOLUTION, robot_radius=0.15
    #                     )

    # trajectory = astar_planner.find_path()

    dfs_planner = DepthFirstSearch(walls, min_bounds, max_bounds,
                                ROBOT_START_POS_2D, TARGET_POS_2D,
                                DFS_GRID_RESOLUTION, robot_radius=0.15)
    
    trajectory = dfs_planner.find_path()

    return trajectory


def get_actual_pos_theta(data):
    x, y = data.qpos[0], data.qpos[1]
    w, xq, yq, zq = data.qpos[3:7]
    theta = np.arctan2(2 * (w * xq + yq * zq), 1 - 2 * (xq**2 + yq**2))

    return x, y, theta

def get_target_pos_theta(trajectory, x, y):
    target_point = trajectory[0]

    dx = target_point[0] - x
    dy = target_point[1] - y
    dist_to_target = np.hypot(dx, dy)

    # 0.1 for deep search and 0.01 for A-star
    if dist_to_target < 0.1:  # Если близко к точке — переходим к следующей
        trajectory.pop(0)
        if not trajectory:
            return (0, 0), trajectory
        target_point = trajectory[0]
        dx = target_point[0] - x
        dy = target_point[1] - y
        dist_to_target = np.hypot(dx, dy)

    angle_target = np.arctan2(dy, dx)

    return dist_to_target, angle_target

class PDController:
    def __init__(self, Kp_lin, Kd_lin, Kp_ang, Kd_ang):
        self.Kp_lin = Kp_lin
        self.Kd_lin = Kd_lin
        self.Kp_ang = Kp_ang
        self.Kd_ang = Kd_ang

        self.prev_dist_err = 0
        self.prev_angle_err = 0

    def pd_reg(self, actual_dist, actual_angle, target_dist, target_angle, dt):
        v = Kp_lin * (target_dist - actual_dist) + Kd_lin * ((target_dist - actual_dist) - self.prev_dist_err) / dt
        delta = Kp_ang * (target_angle - actual_angle) + Kd_ang * ((target_angle - actual_angle) - self.prev_angle_err) / dt

        self.prev_dist_err = target_dist - actual_dist
        self.prev_angle_err = target_angle - actual_angle

        return v, delta

def control_func(model, data):
    # Получаем текущие положение и ориентацию
    x, y, theta = get_actual_pos_theta(data)

    # Получаем требуемые положение и ориентацию
    dist_to_target, angle_target = get_target_pos_theta(trajectory, x, y)

    # print(dist_to_target, angle_target)

    data.ctrl = controller.pd_reg(dist_to_target, angle_target, np.hypot(x, y), theta, model.opt.timestep)

if __name__ == '__main__':
    trajectory = get_trajectory()

    # A-Star params
    # Kp_lin, Kd_lin = 0.2, 0.2
    # Kp_ang, Kd_ang = -0.05, -0.13

    # Deep params
    Kp_lin, Kd_lin = 0.2, 0.2
    Kp_ang, Kd_ang = -0.01, -0.30
    controller = PDController(Kp_lin, Kd_lin, Kp_ang, Kd_ang)

    # Загрузка модели
    model = mujoco.MjModel.from_xml_path("/Users/polinakuranova/uni/ptur/RobMPC/model_2/scene_2.xml")
    data = mujoco.MjData(model)

    mujoco.set_mjcb_control(control_func)

    # Запуск интерактивного окна
    with viewer.launch(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
