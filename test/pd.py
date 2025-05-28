import mujoco
import mujoco.viewer
import numpy as np
import time

# PID контроллер
class PIDController:
    def __init__(self, Kp, Kd, setpoint=0.0):
        self.Kp = Kp
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, measurement, dt):
        error = self.setpoint - measurement
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return self.Kp * error + self.Kd * derivative

# Загрузка модели
model_path = r"C:\Users\rad\ptur\RobMPC\model\scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Идентификаторы
joint_id = model.joint('back_left').id
actuator_id = model.actuator('4x4').id

# Настройка PID
pid = PIDController(Kp=2.0, Kd=0.05)
pid.setpoint = 5.0  # Целевая скорость (можно менять на ходу)

# Запуск визуализации
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        # Измеряем текущую скорость по вращению колеса
        vel = data.qvel[joint_id]

        # Обновляем управление PID
        dt = model.opt.timestep
        control = pid.update(vel, dt)

        # Применяем управление
        data.ctrl[actuator_id] = control

        # Передаем управление отрисовке
        viewer.sync()
        time.sleep(0.15)
        print(f"vel={vel:.2f}, control={control:.2f}")
