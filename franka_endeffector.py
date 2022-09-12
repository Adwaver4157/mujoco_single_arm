import mujoco
import mujoco_viewer
import numpy as np


def get_link_length(s, e):
    return np.linalg.norm(e - s)

def print_link_length(links):
    print(f'base: {links[0]}, link0: {links[1]}, link1: {links[2]}')
    print(f'link2: {links[3]}, link3: {links[4]}, link4: {links[5]}')
    print(f'link5: {links[6]}, link6: {links[7]}, link7: {links[8]}')
    print(f'link8: {links[9]}, hand: {links[10]}, left: {links[11]}')
    

def get_endeffector_pos(data):
    return data.xpos[11]

if __name__=='__main__':
    print("Test the forward kinematics of franka robot.")

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
            # data.ctrl = [0, 0, 0, 0, 1.57079, 0, 0, 0]
            data.ctrl = [np.pi/2, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0]
            links = []
            for i in range(len(data.xpos)-1):
                links.append(get_link_length(data.xpos[i], data.xpos[i+1]))
            print_link_length(links)
            # print(data.xpos[11]) # print(data.xpos[12])
            # print(data.qpos[4], data.qpos[8])
        else:
            break

    # close
    viewer.close()