import mujoco
import cv2
import time
import mujoco.viewer


def main():
    try:
        model = mujoco.MjModel.from_xml_path("C:/Users/rad/ptur/RobMPC/model/scene_2.xml")
        data = mujoco.MjData(model)
        
        renderer = mujoco.Renderer(model, width=640, height=480)
        viewer = mujoco.viewer.launch_passive(model, data)  # создаём viewer из mujoco-viewer
        
        while True:
            start_time = time.time()
            
            mujoco.mj_step(model, data)
            
            renderer.update_scene(data, camera="front_cam")
            img = renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.imshow("Robot Camera View", img_bgr)
            
            # Обновляем окно viewer (интерактивное 3D)
            viewer.render()
            
            if cv2.waitKey(1) == 27:
                break
            
            elapsed = time.time() - start_time
            if elapsed < 0.01:
                time.sleep(0.01 - elapsed)
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        if 'renderer' in locals():
            renderer.close()
        if 'viewer' in locals():
            viewer.close()
        print("Simulation ended")

if __name__ == "__main__":
    main()
