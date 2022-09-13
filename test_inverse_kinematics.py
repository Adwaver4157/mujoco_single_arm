import torch

from train_inverse import InverseKinematics

import mujoco
import mujoco_viewer
import numpy as np
import os
import argparse

import time

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

def to_tensor(value, device):
    return torch.tensor(value, dtype=torch.float).to(device)

def to_numpy(value):
    return value.to('cpu').detach().numpy().copy()

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="test inverse kinematics predictor")

    parser.add_argument('--data-path', type=str, default='dataset/franka.pkl')
    parser.add_argument('--save-dir', type=str, default='model')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_path = os.path.join(args.save_dir, 'best_val_model.pt')
    best_checkpoint = torch.load(best_path)
    predictor = InverseKinematics().to(device)
    predictor.load_state_dict(best_checkpoint['model_state_dict'])
    predictor.eval()

    model_base = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
    data_base = mujoco.MjData(model_base)
    # create the viewer object
    viewer_base = mujoco_viewer.MujocoViewer(model_base, data_base)

    model_pred = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
    data_pred = mujoco.MjData(model_pred)
    # create the viewer object
    viewer_pred = mujoco_viewer.MujocoViewer(model_pred, data_pred)

    # simulate and render
    pre = [0.0] * 9
    for _ in range(10000):
        if viewer_base.is_alive and viewer_pred.is_alive:
            mujoco.mj_step(model_base, data_base)
            mujoco.mj_step(model_pred, data_pred)
            viewer_base.render()
            viewer_pred.render()

            random_ctrl = random_action()
            data_base.qpos = random_ctrl + [0] * 2
            endeffector_pos = get_endeffector_pos(data_base)
            pred_qpos = predictor(to_tensor(endeffector_pos, device))
            data_pred.qpos = to_numpy(pred_qpos).tolist() + [0] * 2
            pred_endeffector_pos = get_endeffector_pos(data_pred)

            diff = np.linalg.norm(endeffector_pos - pred_endeffector_pos)
            print(f'Diff: {diff}')
            time.sleep(1)
        else:
            break

    # close
    viewer_base.close()
    viewer_pred.close()
