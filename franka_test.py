import mujoco
import mujoco_viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
data = mujoco.MjData(model)
# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
pre = [0.0] * 9
for _ in range(10000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
        data.ctrl = [0, 2, 0, -1.57079, 0, 1.57079, -0.7853, 255]
        # data.ctrl = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255]
        """ qpos = [0.0]*9
        qpos[5] = np.pi / 2
        qpos[6] = np.pi / 2
        qpos[7] = 0
        qpos[8] = 0.04
        data.qpos = qpos """
        print(data.qpos[4], data.qpos[8])
        """ print(pre - data.qpos*100000)
        pre = data.qpos * 100000 """
    else:
        break

# close
viewer.close()