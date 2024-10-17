import time
import numpy as np
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('UR5+gripper/UR5gripper_v3.xml')
d = mujoco.MjData(m)

def generate_trajectory(time_start, time_end, joint_angle_start, joint_angle_end):
    """
    Generate a trajectory from cubic polynomials.
    Solves for the coefficients from the equations:
    angular_position = a0 + a1*t + a2*t^2 + a3*t^3
    angular_velocity = a1 + 2*a2*t + 3*a3*t^2
    angular_acceleration = 2*a2 + 6*a3*t
    """
    divisor = 1 / (time_end-time_start)^3
    coefficients_array = np.array([
        [joint_angle_end*(time_start^2)*(3*time_end-time_start)],
        [6*time_start*(joint_angle_start-joint_angle_end)],
        [3*(time_start+time_end)*(joint_angle_end-joint_angle_start)],
        [2*(joint_angle_start-joint_angle_end)]
    ])
    
    coefficients = coefficients_array*divisor
    return coefficients
# actuator signal determined by actuators specified in xml file
# the array for mjData.ctrl directly corresponds to the order in which
# the actuators are declared in the xml file.
# e.g. d.ctrl[0] is the control signal sent to the first actuator
mujoco.mj_resetData(m, d)

# setting control signal via control callback
def initial_position(data):
    data.qpos[0] = 0
    data.qpos[1] = 0
    data.qpos[2] = -1.57
    data.qpos[3] = 1.57
    data.qpos[4] = 1.57
    data.qpos[5] = 7.5

def mycontroller(model, data):
    data.ctrl[0] = -21*(data.qpos[0]+1.57)-.11*data.qvel[0] 
    data.ctrl[1] = -30*(data.qpos[1]+1.57)-.1*data.qvel[1] 
    data.ctrl[2] = -15*(data.qpos[2]-1.57)-.05*data.qvel[2] 
    data.ctrl[3] = -21*(data.qpos[3]+1.57)-.11*data.qvel[3] 
    data.ctrl[4] = -15*(data.qpos[4]+1.57)-.01*data.qvel[4] 
    data.ctrl[5] = -7.5*(data.qpos[5]+7.5) 

mujoco.set_mjcb_control(mycontroller)

# simulate
with mujoco.viewer.launch_passive(m, d) as viewer:
    initial_position(d)
    viewer.cam.trackbodyid = 0
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.elevation = -30
    viewer.cam.azimuth = -135
    viewer.cam.distance = 3
    viewer.sync()
    while viewer.is_running():
        mujoco.mj_step(m,d)
        viewer.sync() 

