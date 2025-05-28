import mujoco
import numpy as np
import cv2
import time
from collections import deque
import os
import glfw

class SimpleSLAM:
    def __init__(self):
        self.map = np.zeros((500, 500), dtype=np.float32)
        self.resolution = 0.01  # meters per pixel
        self.robot_pose = np.array([250, 250, 0])  # x, y, theta (pixels)
        self.path = deque(maxlen=1000)
        
    def update(self, img, velocity, dt):
        # Update robot pose (odometry)
        self.robot_pose[0] += velocity[0] * np.cos(self.robot_pose[2]) * dt / self.resolution
        self.robot_pose[1] += velocity[0] * np.sin(self.robot_pose[2]) * dt / self.resolution
        self.robot_pose[2] += velocity[1] * dt
        self.path.append(self.robot_pose.copy())
        
        # Process image for mapping
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, edges = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Add obstacles to map
        h, w = edges.shape
        center_x, center_y = w//2, h//2
        for y in range(h):
            for x in range(w):
                if edges[y, x] < 127:  # Dark pixels are obstacles
                    global_x = int(self.robot_pose[0] + (x - center_x))
                    global_y = int(self.robot_pose[1] - (y - center_y))
                    if 0 <= global_x < 500 and 0 <= global_y < 500:
                        self.map[global_y, global_x] = min(1.0, self.map[global_y, global_x] + 0.1)

def load_model():
    try:
        # Проверяем несколько возможных путей
        possible_paths = [
            "C:/Users/rad/ptur/RobMPC/model/scene_2.xml",
            os.path.join(os.path.dirname(__file__), "model/scene_2.xml"),
            "scene_2.xml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model = mujoco.MjModel.from_xml_path(path)
                data = mujoco.MjData(model)
                print(f"Модель успешно загружена из {path}")
                print("Актуаторы:", [model.actuator(i).name for i in range(model.nu)])
                print("Сенсоры:", [model.sensor(i).name for i in range(model.nsensor)])
                return model, data
                
        raise FileNotFoundError("Не удалось найти файл scene_2.xml")
        
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None, None

def main():
    # Инициализация GLFW для обработки окон
    if not glfw.init():
        print("Ошибка инициализации GLFW")
        return

    model, data = load_model()
    if model is None:
        glfw.terminate()
        return

    # Создаем рендерер с обработкой ошибок
    try:
        renderer = mujoco.Renderer(model, width=640, height=480)
    except Exception as e:
        print(f"Ошибка создания рендерера: {e}")
        glfw.terminate()
        return

    slam = SimpleSLAM()
    BASE_SPEED = 2.0
    TURN_GAIN = 1.2
    last_time = time.time()
    
    try:
        while not glfw.window_should_close(glfw.get_current_context()):
            # Расчет временного шага
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Обработка событий
            glfw.poll_events()
            
            # Шаг симуляции
            mujoco.mj_step(model, data)
            
            # Получаем изображение с камеры
            try:
                # Пробуем разные камеры
                cameras = ["front_cam", "robot_cam", "global_view"]
                for cam in cameras:
                    try:
                        renderer.update_scene(data, camera=cam)
                        img = renderer.render()
                        break
                    except:
                        continue
                else:
                    raise RuntimeError("Не удалось получить изображение ни с одной камеры")
            except Exception as e:
                print(f"Ошибка рендеринга: {e}")
                break
            
            # Получаем данные с датчиков
            try:
                sensor_data = {}
                for i in range(model.nsensor):
                    name = model.sensor(i).name
                    sensor_data[name] = data.sensor(i).data[0]
                
                right_dist = sensor_data.get("range_right", 1.0)
                front_dist = sensor_data.get("range_front", 1.0)
                left_dist = sensor_data.get("range_left", 1.0)
            except Exception as e:
                print(f"Ошибка чтения датчиков: {e}")
                break
            
            # Логика управления
            if front_dist < 0.4:
                # Обнаружено препятствие - движение назад и поворот
                speed_cmd = -BASE_SPEED * 0.5
                turn_cmd = TURN_GAIN if left_dist > right_dist else -TURN_GAIN
            else:
                # Следование вдоль стены
                error = 0.3 - right_dist
                turn_cmd = error * 3.0
                speed_cmd = BASE_SPEED * min(1.0, front_dist/0.5)
            
            # Применяем управление
            try:
                data.actuator("4x4").ctrl[0] = np.clip(speed_cmd, -BASE_SPEED, BASE_SPEED)
                data.actuator("turn").ctrl[0] = np.clip(turn_cmd, -2.0, 2.0)
            except Exception as e:
                print(f"Ошибка управления: {e}")
                break
            
            # Обновляем SLAM
            slam.update(img, [speed_cmd, turn_cmd*0.1], dt)
            
            # Визуализация
            try:
                # Рендерим сцену
                renderer.update_scene(data)
                sim_img = renderer.render()
                
                # Создаем карту SLAM
                map_img = cv2.cvtColor((slam.map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                
                # Рисуем путь робота
                if len(slam.path) > 1:
                    pts = np.array([(int(p[0]), int(p[1])) for p in slam.path], dtype=np.int32)
                    cv2.polylines(map_img, [pts], False, (0, 0, 255), 1)
                
                # Добавляем информацию о датчиках
                info_img = np.zeros((150, 400, 3), dtype=np.uint8)
                cv2.putText(info_img, f"Front: {front_dist:.2f}m", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                cv2.putText(info_img, f"Left: {left_dist:.2f}m", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                cv2.putText(info_img, f"Right: {right_dist:.2f}m", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                cv2.putText(info_img, f"Speed: {speed_cmd:.2f}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                cv2.putText(info_img, f"Turn: {turn_cmd:.2f}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                
                # Показываем изображения
                cv2.imshow("Simulation", sim_img)
                cv2.imshow("SLAM Map", map_img)
                cv2.imshow("Sensor Info", info_img)
                
                if cv2.waitKey(1) == 27:  # ESC для выхода
                    break
                    
            except Exception as e:
                print(f"Ошибка визуализации: {e}")
                break
                
    finally:
        cv2.destroyAllWindows()
        glfw.terminate()
        print("Симуляция завершена")

if __name__ == "__main__":
    main()