import mujoco
from mujoco import viewer

# Загрузка модели
model = mujoco.MjModel.from_xml_path("C:/Users/rad/ptur/RobMPC/model/scene_2.xml")
data = mujoco.MjData(model)

# Запуск интерактивного окна
with viewer.launch_passive(model, data) as v:
    while True:
        mujoco.mj_step(model, data)
        v.sync()
