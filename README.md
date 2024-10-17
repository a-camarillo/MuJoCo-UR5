# Trajectory Planning of the UR5 Robotic Arm in MuJoCo

The work in this repository started from Paul Daniel's original
[repository](https://github.com/PaulDanielML/MuJoCo_RL_UR5)
for working with the UR5 robotic arm in MuJoCo but is updated to use the
current [MuJoCo python bindings](https://mujoco.readthedocs.io/en/stable/python.html).
The model files provided in the `UR5+gripper/` directory were taken directly
from the original repository.

## Getting Started

### Clone the repository

HTTPS:

`git clone https://github.com/a-camarillo/MuJoCo-UR5.git`

SSH:

`git clone git@github.com:a-camarillo/MuJoCo-UR5.git`

GitHub CLI:

`gh repo clone a-camarillo/MuJoCo-UR5`

### Setting Up A Python Environment

It is highly recommended to use virtual environments.

#### Using venv

Change into the working directory:

`cd MuJoCo-UR5`

Create the virtual environment *Note: On Windows you might just call `python`*:

`python3 -m venv ./venv_mujoco/`

Activate the virtual environment:

`source ./venv_mujoco/bin/activate`

To exit(deactivate) the virtual environment, simply type:

`deactivate`

[//]: # (TODO: Finish the README) 
