# How to use MuJoCo through Python bindings
Franka robot has 7 hinge joints and 1 tendon which connects two slide joints (endeffectors).

## How to make motion of the franka robot
You can control the franka robot through data variable which is instance of mujoco.MjData.
The franka robot joints can be instructed by setting data.ctrl parameters.
For instance, the franka robot has 8 controllable parts (7 joints and a tendon), so you can
set data.ctrl parameters as follows.

```python
import mujoco

model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

data.ctrl = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255]
```

The value of data.ctrl means joint or dendon positions. So you can see
the robot joints and a tendon are set to be the parameters, a few steps after you set them.

## MuJoCo coordinates
left: X+
from front to back: Y+
up: Z+

## Setup Docker environments
```
# docker build -t adwaver4157/mujoco .
docker pull adwaver4157/mujoco
./RUN-DOCKER.sh username
```

## (Optional) If you use this in server via ssh, you have to start VNC and then run python file 
```bash
vnc
python franka.py
```