import mujoco
import cv2
import numpy as np
import time

def main():
    try:
        # Загрузка модели
        model = mujoco.MjModel.from_xml_path("C:/Users/rad/ptur/RobMPC/model/scene_2.xml")
        data = mujoco.MjData(model)
        
        # Создание рендерера
        renderer = mujoco.Renderer(model, width=640, height=480)
        
        # Основной цикл симуляции
        while True:
            start_time = time.time()
            
            # Шаг симуляции
            mujoco.mj_step(model, data)
            
            # Обновление сцены с камеры робота
            renderer.update_scene(data, camera="front_cam")
            
            # Рендеринг
            img = renderer.render()
            
            # Конвертация цветового пространства (MuJoCo RGB → OpenCV BGR)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Отображение
            cv2.imshow("Robot Camera View", img_bgr)
            
            # Управление (выход по ESC)
            if cv2.waitKey(1) == 27:
                break
                
            # Контроль частоты кадров
            elapsed = time.time() - start_time
            if elapsed < 0.01:  # timestep 0.01s
                time.sleep(0.01 - elapsed)
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        if 'renderer' in locals():
            renderer.close()
        print("Simulation ended")

if __name__ == "__main__":
    main()