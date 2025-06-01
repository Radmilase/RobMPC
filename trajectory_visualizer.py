"""
Визуализирует лабиринт и вычисленную траекторию с использованием Matplotlib.
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
    """Обрабатывает отрисовку стен лабиринта, сетки и пути."""

    def __init__(self, walls, world_min_bounds, world_max_bounds, start_pos=None, goal_pos=None):
        """
        Аргументы:
            walls (list): Список объектов Wall из MazeParser.
            world_min_bounds (np.array): Мин (x, y) мира.
            world_max_bounds (np.array): Макс (x, y) мира.
            start_pos (np.array, опционально): Начальная позиция (x, y).
            goal_pos (np.array, опционально): Целевая позиция (x, y).
        """
        self.walls = walls
        self.world_min_bounds = world_min_bounds
        self.world_max_bounds = world_max_bounds
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        
        self.fig, self.ax = plt.subplots(figsize=VIS_FIGURE_SIZE)
        self._setup_plot()

    def _setup_plot(self):
        """Настраивает основные свойства графика, оси и рисует стены."""
        self.ax.set_aspect('equal', adjustable='box')
        # Добавляем небольшой отступ к границам мира для лучшей визуализации
        margin = GRID_RESOLUTION * 5 
        self.ax.set_xlim(self.world_min_bounds[0] - margin, self.world_max_bounds[0] + margin)
        self.ax.set_ylim(self.world_min_bounds[1] - margin, self.world_max_bounds[1] + margin)
        self.ax.set_xlabel("Координата X")
        self.ax.set_ylabel("Координата Y")
        self.ax.set_title("Визуализация траектории в лабиринте")
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, color=VIS_GRID_COLOR)
        
        # Рисуем стены
        for wall in self.walls:
            (x, y), width, height = wall.get_patch_params()
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=VIS_WALL_COLOR, facecolor=VIS_WALL_COLOR, alpha=0.7)
            self.ax.add_patch(rect)

        # Рисуем маркеры Начала и Цели
        if self.start_pos is not None:
            self.ax.plot(self.start_pos[0], self.start_pos[1], 'o', markersize=10, color=VIS_START_MARKER_COLOR, label='Старт')
        if self.goal_pos is not None:
            self.ax.plot(self.goal_pos[0], self.goal_pos[1], 'x', markersize=10, markeredgewidth=3, color=VIS_GOAL_MARKER_COLOR, label='Цель')
        
        if self.start_pos is not None or self.goal_pos is not None:
            self.ax.legend()

    def draw_potential_field(self, planner, world_min_bounds, world_max_bounds, resolution=0.1, num_levels=20, cmap='viridis'):
        """Отрисовывает поле потенциалов с использованием контурного графика."""
        x_coords = np.arange(world_min_bounds[0], world_max_bounds[0] + resolution, resolution)
        y_coords = np.arange(world_min_bounds[1], world_max_bounds[1] + resolution, resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pos = np.array([X[i, j], Y[i, j]])
                if planner.is_point_inside_any_wall(pos):
                    Z[i, j] = np.nan # Не рисовать потенциал внутри стен
                else:
                    Z[i, j] = planner.get_total_potential(pos)
        
        # Обрезаем экстремальные значения для лучшей визуализации, если потенциалы уходят в бесконечность
        # Например, если Z содержит inf или очень большие числа, contourf может испытывать трудности.
        # Мы можем ограничить его, например, 99-м процентилем или фиксированным большим числом, если это необходимо.
        # Пока попробуем без обрезки, но будем иметь в виду.
        # mask_inf = np.isinf(Z)
        # Z[mask_inf] = np.nan # Или большое число, plt.contourf обрабатывает nan, не отрисовывая их
        
        # Нормализуем значения потенциала для лучшего отображения цветов, если они слишком велики
        # Это часто помогает показать вариации в областях с низким потенциалом
        # Однако мы также хотим видеть высокие потенциалы вблизи препятствий.
        # Давайте используем логарифмическую шкалу для цветов или обеспечим широкий диапазон уровней.
        
        # Если Z имеет очень большие вариации (например, от почти нуля до очень высоких значений), логарифмическая шкала может быть хороша.
        # Однако contourf может быть чувствителен к отрицательным или нулевым значениям с LogNorm.
        # Сначала попробуем с линейной шкалой по умолчанию.
        # Рассмотрите использование уровней для контроля количества контурных линий и их значений.
        
        # Фильтруем NaN перед вычислением процентилей и min/max для уровней
        Z_finite = Z[np.isfinite(Z)]
        if Z_finite.size == 0: # Все точки NaN или Inf (например, полностью внутри стен)
            print("Предупреждение: Нет допустимых значений потенциала для отрисовки поля.")
            return # Или обработайте этот случай соответствующим образом

        max_potential_display = np.percentile(Z_finite, 99) 
        min_potential_display = np.min(Z_finite)
        
        # Убедимся, что min и max различны, чтобы избежать проблем с linspace
        if min_potential_display >= max_potential_display:
            max_potential_display = min_potential_display + 1 # Добавляем небольшую дельту

        levels = np.linspace(min_potential_display, max_potential_display, num_levels)

        contour = self.ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.6, extend='max') # extend='max' для значений > max_level
        # self.fig.colorbar(contour, ax=self.ax, label='Значение потенциала') # Опционально: добавить цветовую шкалу

    def draw_grid(self, grid_min_x, grid_min_y, grid_max_x, grid_max_y, resolution):
        """Отрисовывает сетку поиска A* (опционально)."""
        # Вертикальные линии
        for x_coord in np.arange(grid_min_x, grid_max_x + resolution, resolution):
            self.ax.axvline(x_coord, color=VIS_GRID_COLOR, linestyle=':', linewidth=0.3)
        # Горизонтальные линии
        for y_coord in np.arange(grid_min_y, grid_max_y + resolution, resolution):
            self.ax.axhline(y_coord, color=VIS_GRID_COLOR, linestyle=':', linewidth=0.3)

    def draw_path(self, path_points, color=VIS_PATH_COLOR, label='Путь'):
        """Отрисовывает заданный путь (список кортежей (x, y) или np.array)."""
        if not path_points:
            print("Визуализатор: Нет точек пути для отрисовки.")
            return
        
        path_np = np.array(path_points)
        self.ax.plot(path_np[:, 0], path_np[:, 1], color=color, linewidth=2, marker='.', markersize=4, label=label)
        self.ax.legend() # Обновляем легенду, если добавлены новые элементы

    def show(self):
        """Отображает график."""
        plt.show()

if __name__ == '__main__':
    # Пример использования (требует maze_parser.py и maze_config.py)
    from maze_parser import MazeParser
    from maze_config import ROBOT_START_POS_2D, TARGET_POS_2D

    print("Запуск примера TrajectoryVisualizer...")
    parser = MazeParser() # Использует MAZE_XML_PATH из конфига
    walls = parser.get_walls()
    min_bounds, max_bounds = parser.get_world_bounds()

    if not walls:
        print("Стены не распознаны, запуск примера визуализатора невозможен.")
    else:
        visualizer = TrajectoryVisualizer(walls, min_bounds, max_bounds, ROBOT_START_POS_2D, TARGET_POS_2D)
        
        # Пример пути (замените на фактический путь A* или Потенциального Поля)
        example_path = [
            ROBOT_START_POS_2D,
            np.array([ROBOT_START_POS_2D[0] + 0.5, ROBOT_START_POS_2D[1] - 0.2]),
            np.array([TARGET_POS_2D[0] - 0.3, TARGET_POS_2D[1] + 0.5]),
            TARGET_POS_2D
        ]
        visualizer.draw_path(example_path, color='green', label='Пример Пути')
        
        # visualizer.draw_grid(min_bounds[0], min_bounds[1], max_bounds[0], max_bounds[1], GRID_RESOLUTION)
        visualizer.show()
        print("Пример TrajectoryVisualizer завершен.") 