import mujoco
import cv2
import time

# Загрузка модели
model = mujoco.MjModel.from_xml_path("C:/Users/rad/ptur/RobMPC/model/scene_2.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 320, 240)  # Разрешение совпадает с камерой

# Основной цикл симуляции
duration = 100 # Длительность симуляции в секундах
start_time = time.time()

while (time.time() - start_time) < duration:
    # Шаг симуляции
    mujoco.mj_step(model, data)
    
    # Обновление и рендеринг камеры
    renderer.update_scene(data, camera="front_cam")
    img = renderer.render()
    
    # Отображение изображения
    cv2.imshow("Camera Output", img)
    
    # Выход по ESC или по истечении времени
    if cv2.waitKey(1) == 27:  # Код клавиши ESC
        break

cv2.destroyAllWindows()