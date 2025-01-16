import time
from ikpy.chain import Chain
import numpy as np
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('UR5+gripper/UR5gripper_v3_2.xml')
d = mujoco.MjData(m)
UR5_chain = Chain.from_urdf_file('UR5+gripper/ur5_gripper.urdf')

setpoints = {
        "shoulder_pan": -1.57,
        "shoulder_lift": -1.57,
        "forearm": 1.57,
        "wrist_1": -1.57,
        "wrist_2": -1.57,
        "wrist_3": -3.14,
        }

setpoints_2 = {
        "shoulder_pan_2": -1.57,
        "shoulder_lift_2": -1.57,
        "forearm_2": 1.57,
        "wrist_1_2": -1.57,
        "wrist_2_2": -1.57,
        "wrist_3_2": -3.14,
        }

positions = [0] + [setpoints[x] for x in setpoints] + [0]
positions_2 = [0] + [setpoints_2[x] for x in setpoints_2] + [0]

def calculate_joint_angles(chain, target_position, initial_position):
    """
    Calculate the arm's joint angles for desired end effector position
    based on inverse kinematics
    """
    joint_angles = chain.inverse_kinematics(
                target_position=target_position,
                initial_position=initial_position,
                )
    # first and last joint angle are considered inactive and are not needed for control
    return joint_angles[1:-1]

def move_end_effector(kinematic_chain, target_coordinates, initial_coordinates):
    """
    Move end effector to desired coordinates
    """
    joint_angles = calculate_joint_angles(kinematic_chain,
                                          target_position=target_coordinates,
                                          initial_position=initial_coordinates
                                          )
    
    # set the joint angle setpoints, mjcb_control will handle the movement
    setpoints["shoulder_pan"] = joint_angles[0]
    setpoints["shoulder_lift"] = joint_angles[1]
    setpoints["forearm"] = joint_angles[2]
    setpoints["wrist_1"] = joint_angles[3]
    setpoints["wrist_2"] = joint_angles[4]
    setpoints["wrist_3"] = joint_angles[5]


#def generate_trajectory(time_start, time_end, joint_angle_start, joint_angle_end):
#    """
#    Generate a trajectory from cubic polynomials.
#    Solves for the coefficients from the equations:
#    angular_position = a0 + a1*t + a2*t^2 + a3*t^3
#    angular_velocity = a1 + 2*a2*t + 3*a3*t^2
#    angular_acceleration = 2*a2 + 6*a3*t
#    """
#    divisor = 1 / (time_end-time_start)^3
#    coefficients_array = np.array([
#        [joint_angle_end*(time_start^2)*(3*time_end-time_start)],
#        [6*time_start*(joint_angle_start-joint_angle_end)],
#        [3*(time_start+time_end)*(joint_angle_end-joint_angle_start)],
#        [2*(joint_angle_start-joint_angle_end)]
#    ])
#    
#    coefficients = coefficients_array*divisor
#    return coefficients

# actuator signal determined by actuators specified in xml file
# the array for mjData.ctrl directly corresponds to the order in which
# the actuators are declared in the xml file.
# e.g. d.ctrl[0] is the control signal sent to the first actuator
mujoco.mj_resetData(m, d)

# setting control signal via control callback
def initial_position(data):
    data.qpos[0] = setpoints["shoulder_pan"]
    data.qpos[1] = setpoints["shoulder_lift"]
    data.qpos[2] = setpoints["forearm"]
    data.qpos[3] = setpoints["wrist_1"]
    data.qpos[4] = setpoints["wrist_2"]
    data.qpos[5] = setpoints["wrist_3"]
    
    data.qpos[8] = setpoints["shoulder_pan"]
    data.qpos[9] = setpoints["shoulder_lift"]
    data.qpos[10] = setpoints["forearm"]
    data.qpos[11] = setpoints["wrist_1"]
    data.qpos[12] = setpoints["wrist_2"]
    data.qpos[13] = setpoints["wrist_3"]

def mycontroller(model, data):
    # 1st manipulator where end effector moves
    data.ctrl[0] = -21*(data.qpos[0]-setpoints["shoulder_pan"])-.11*data.qvel[0] 
    data.ctrl[1] = -30*(data.qpos[1]-setpoints["shoulder_lift"])-.1*data.qvel[1] 
    data.ctrl[2] = -15*(data.qpos[2]-setpoints["forearm"])-.05*data.qvel[2] 
    data.ctrl[3] = -21*(data.qpos[3]-setpoints["wrist_1"])-.11*data.qvel[3] 
    data.ctrl[4] = -15*(data.qpos[4]-setpoints["wrist_2"])-.01*data.qvel[4] 
    data.ctrl[5] = -7.5*(data.qpos[5]-setpoints["wrist_3"]) 
   
    # 2nd manipulator that stays in fixed position
    data.ctrl[7] = -21*(data.qpos[8]-setpoints_2["shoulder_pan_2"])-.11*data.qvel[8] 
    data.ctrl[8] = -30*(data.qpos[9]-setpoints_2["shoulder_lift_2"])-.1*data.qvel[9] 
    data.ctrl[9] = -15*(data.qpos[10]-setpoints_2["forearm_2"])-.05*data.qvel[10] 
    data.ctrl[10] = -21*(data.qpos[11]-setpoints_2["wrist_1_2"])-.11*data.qvel[11] 
    data.ctrl[11] = -15*(data.qpos[12]-setpoints_2["wrist_2_2"])-.01*data.qvel[12] 
    data.ctrl[12] = -7.5*(data.qpos[13]-setpoints_2["wrist_3_2"]) 

def key_callback(keycode):
    if chr(keycode) == ' ':
        setpoints["shoulder_pan"] = -1.57
        setpoints["shoulder_lift"] = -1.57
        setpoints["forearm"] = 1.57
        setpoints["wrist_1"] = -1.57
        setpoints["wrist_2"] = -1.57
        setpoints["wrist_3"] = -3.14

mujoco.set_mjcb_control(mycontroller)

initial_position(d)

# simulate
with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    print(f'tool position: {d.body('robotiq_85_base_link').xpos}')
    viewer.cam.trackbodyid = 0
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.elevation = -30
    viewer.cam.azimuth = -135
    viewer.cam.distance = 3
    viewer.sync()
    move_end_effector(UR5_chain, [0.5, 0.5, 0.5], initial_coordinates=positions)
    time.sleep(5.0)
    while viewer.is_running():
        mujoco.mj_step(m,d)
        print(f'tool position: {d.body('robotiq_85_base_link').xpos}')
        viewer.sync()

