import numpy as np
import mujoco_viewer
import mujoco.viewer
import msm_model
import time
import keyboard  # For capturing keypresses
import threading
import matplotlib.pyplot as plt
from simple_pid import PID
import utils

msm_linear_rack = msm_model.MSMLinear(tb_type=1,
                                      controller_type="closed_loop",
                                      control_on_callback=False)

pid = PID(6, 20000, 8e-5)
pid.sample_time = utils.NN_WORKING_PERIOD
pid.proportional_on_measurement = False
pid.differential_on_measurement = False

reset_requested = False

def key_listener():  # Function to listen for keypress
    global reset_requested
    while True:
        time.sleep(0.4)
        if keyboard.is_pressed('9'):
            reset_requested = True

# Start the key listener thread
key_thread = threading.Thread(target=key_listener, daemon=True)
key_thread.start()

def reset_simulation():
    msm_linear_rack.reset()

#"""
# Create a viewer instance
viewer = mujoco_viewer.MujocoViewer(msm_linear_rack.model, msm_linear_rack.data)
#viewer.cam
viewer.cam.azimuth = 180
viewer.cam.distance = 0.005
#viewer.cam.elevation = -45
step = 0
desired_vel = 0.01
pid.setpoint = 0
tb_velocity_list = []

# Simulation loop
while viewer.is_alive:
    # Apply control policy

    vel = msm_linear_rack.simulation_data["rack_vel"]
    control_error = vel[-1] - desired_vel
    #print(f"control error: {control_error}")
    tb_velocity = msm_linear_rack.data.qvel[msm_linear_rack.tooth_joint_id_list[1]] / utils.e_0
    tb_velocity_list.append(tb_velocity)

    prediction = pid(control_error, dt=utils.NN_WORKING_PERIOD)
    # prediction = 1
    msm_linear_rack.sim_step(prediction)



    # Step the simulation
    # mujoco.mj_step(model, data, nstep=100)  # nstep param can specify how many steps are iterated during one step() function call
    # note, nstep cannot be more than 1 while onstep_compuations are not in the callback (they are in the callback right now)
    viewer.render()

    # Reset simulation if requested
    if reset_requested:
        print("Resetting simulation...")
        reset_simulation()
        reset_requested = False  # Reset the flag

viewer.close()

msm_linear_rack.plot_rack_instant_velocity()
msm_linear_rack.plot_rack_average_velocity()
msm_linear_rack.plot_control_value()

#"""
# time_vec = msm_linear_rack.simulation_data["time"][1:]
# instant_vel_vec = msm_linear_rack.simulation_data["rack_vel"][1:]
# vel_data = np.divide( msm_linear_rack.simulation_data["rack_pos"][1:], time_vec)
# # print(msm_linear_rack.simulation_data["time"])
# # print(msm_linear_rack.simulation_data["rack_pos"])
# # print(vel_data)
# # Plot results
# plt.figure(figsize=(8, 4))
# #plt.plot(time_data, msm_pos_data, label="Joint Position (qpos)")
# plt.plot(time_vec, tb_velocity_list, label="Joint Velocity (qvel)")
# plt.xlabel("Time (s)")
# # plt.ylabel("Position")
# plt.ylabel("Average Velocity")
# plt.title("Joint Velocity Over Time")
# plt.legend()
# plt.grid()
# plt.show()


"""

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()

    while viewer.is_running():  # and time.time() - start < 30:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        control_policy(data, joint_id)
        mujoco.mj_step(model, data)

        # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    # if time_until_next_step > 0:
    #     time.sleep(time_until_next_step)
#"""