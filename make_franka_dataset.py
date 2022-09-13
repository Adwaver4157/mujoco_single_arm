import numpy as np
import os
import pickle
import mujoco
import mujoco_viewer
import argparse

def get_endeffector_pos(data):
    pos = (data.xpos[11] + data.xpos[12]) /  2
    pos = np.concatenate([pos, data.xquat[10]])
    return  pos

def scale_minmax(value, min=0., max=1.):
    return value * (max - min) + min

def scale_ctrl_value(ctrl):
    ctrl[0] = scale_minmax(ctrl[0], min=0, max=2*np.pi)
    ctrl[1] = scale_minmax(ctrl[1], min=-1.7628, max=1.7628)
    ctrl[2] = scale_minmax(ctrl[2], min=0, max=2*np.pi)
    ctrl[3] = scale_minmax(ctrl[3], min=-3.0718, max=-0.0698)
    ctrl[4] = scale_minmax(ctrl[4], min=0, max=2*np.pi)
    ctrl[5] = scale_minmax(ctrl[5], min=-0.0175, max=3.7525)
    ctrl[6] = scale_minmax(ctrl[6], min=0, max=2*np.pi)
    return ctrl.tolist()

def random_action(dim=7):
    ctrl = np.random.rand(dim)
    return scale_ctrl_value(ctrl)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Make franka dataset')
    parser.add_argument('--scale', type=int, default=1000)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--num-steps', type=int, default=10000)
    parser.add_argument('--max-steps', type=int, default=700)
    parser.add_argument('--dir-name', type=str, default='dataset')
    parser.add_argument('--file-name', type=str, default='franka.pkl')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    # Setup MuJoCo franka env

    model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)
    # create the viewer object
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # Rollout random actions in the env
    robot_pos_data = {
                        'joint_ctrl': [],
                        'endeffector': []
                    }
    # simulate and render
    pre_pos = [0.0] * 7
    for _ in range(args.num_steps):
        if viewer.is_alive:
            mujoco.mj_step(model, data)
            viewer.render()
            # data.ctrl = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255]
            random_ctrl = random_action()
            endeffector_pos = get_endeffector_pos(data)
            diff = np.linalg.norm(pre_pos - endeffector_pos * args.scale)
            # print(f'initial diff: {diff}')

            steps = 0
            flag = True
            while (diff > args.delta):
                # data.ctrl = random_ctrl + [0]
                data.qpos = random_ctrl + [0] * 2
                mujoco.mj_step(model, data)
                viewer.render()
                endeffector_pos = get_endeffector_pos(data)
                diff = np.linalg.norm(pre_pos - endeffector_pos * args.scale)
                # print(f'diff: {diff}')

                pre_pos = endeffector_pos * args.scale
                steps += 1
                if steps > args.max_steps:
                    print(f'reach max steps: {steps}')
                    flag = False
                    break

            # save pos in dict
            if flag:
                robot_pos_data['joint_ctrl'].append(random_ctrl)
                robot_pos_data['endeffector'].append(endeffector_pos)
                print(f"save pos, size: {len(robot_pos_data['joint_ctrl'])}")
            
        else:
            break

    # close
    viewer.close()


    # Save position informations in a pickle file
    os.makedirs(args.dir_name, exist_ok=True)
    file_path = os.path.join(args.dir_name, args.file_name)
    if not args.debug:
        with open(file_path, 'wb') as f:
            pickle.dump(robot_pos_data, f)
        print(f'Save data successfully in {file_path}')