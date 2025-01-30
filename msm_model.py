import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import trimesh
import math
import gymnasium as gym
from gymnasium import spaces
import random

import utils

import matplotlib.pyplot as plt

class MSMLinear():

    xml_model = None

    """
            Create a MuJoCo XML file with spheres forming a single rigid body along a trajectory.
            :param tooth_type: # TODO
            :param msm_elements_cnt:  # TODO
            :param steps_per_call defines amount of simulation steps per on step function call.
            :param control_on_callback if True mujoco callback will be used to process control actions (faster), otherwise control will be called explicitly
            By default computes the step count to match controller frequency
    """
    def __init__(self,
                 tooth_type="force_optimal",
                 msm_elements_cnt=4,
                 tb_type=1,
                 controller_type="closed_loop",
                 steps_per_call=None,
                 control_on_callback=True):

        self.tb_type = tb_type
        self.tooth_type = tooth_type
        self.msm_elements_cnt = msm_elements_cnt
        self.control_on_callback = control_on_callback
        self.tooth_height = None
        self.coulomb_friction_ts = None
        self.contact_ratio = None
        self.t_a, self.t_b = None, None
        self.tooth_engage_offset, self.tooth_disengage_offset = 0, 0  # controls the phase at which a tooth is engaged/disengaged [-1, 1]
        self.useful_load = -2  # Newtons, negative value acts against positive motion direction
        self.steps_per_call = steps_per_call
        if self.steps_per_call is None:
            self.steps_per_call = round(utils.NN_WORKING_PERIOD / utils.simulation_timestep)

        # ---- controller settings ----
        if controller_type == "open_loop":
            self.controller_func = self._open_loop_controller
        elif controller_type == "closed_loop":
            self.controller_func = self._closed_loop_controller
        else:
            raise "Unknown controller type"
        self.feedback_threshold = 1  # parameter used to compute the new tooth engagement timing
        self.control_value = 1
        self.enable_blocking = False  # if true the unactuated teeth will still engage the rack if pid value is low
        self.output_vector = np.zeros(self.msm_elements_cnt)  # vector of currently active msm elements with actuation value

        # ---- other ----
        self.simulation_data = {}
        self.tooth_actuator_id_list = []
        self.tooth_joint_id_list = []
        self.msm_plates_init_pos_list = []
        self.control_value_list = np.zeros(self.msm_elements_cnt)

        self._compute_simulation_parameters()
        self.teeth_matrix, self.tooth_profile_mat = self._generate_tooth()

        self.root, self.worldbody, self.asset, self.option, self.default, self.actuator = None, None, None, None, None, None
        self.xml_model = self._generate_xml_model()
        self.model = mujoco.MjModel.from_xml_string(self.xml_model)
        self.data = mujoco.MjData(self.model)

        self._initiate_tooth_plate_ids()
        self.rack_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rack_slider")
        self.rack_init_pos = self.data.qpos[self.rack_joint_id]
        self.rack_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,"rack_actuator")
        self.set_rack_load()

        #mujoco.set_mjcb_passive(self.onstep_computation)  # can this line be called in the desired way?
        if self.control_on_callback:
            mujoco.set_mjcb_control(self.onstep_computation)
        self.reset()

    def _compute_simulation_parameters(self):
        if self.tb_type == 1:
            self.coulomb_friction_ts = (0.6 * utils.msm_A0) * (1e+6) # MPa
        elif self.tb_type == 2:
            self.coulomb_friction_ts = (0.1 * utils.msm_A0) * (1e+6) # MPa
        else:
            raise 'Unknown Twin boundary type, only values of 1 or 2 are possible'

    def _generate_xml_model(self):
        if MSMLinear.xml_model is not None:
            return MSMLinear.xml_model

        self.root = ET.Element("mujoco")  # Create the root of the MuJoCo XML
        self.asset = ET.SubElement(self.root, "asset")
        self.worldbody = ET.SubElement(self.root, "worldbody")
        self.actuator = ET.SubElement(self.root, "actuator")

        # solver settings
        self.option = ET.SubElement(self.root, "option", {
            "timestep": str(utils.simulation_timestep),
            "gravity": "0 0 0",
        })

        # default settings
        self.default = ET.SubElement(self.root, "default")
        ET.SubElement(self.default, "geom",{
            "solimp": "0.95 0.9999 0.001 0.1 2",  # relatively good guess "0.95 0.9999 0.001 0.1 2"  d0, d_widthd, width, midpoint, power
            "solref": "0.000075 0.030",  # (timeconst,dampratio), irelatively good guess 0.000075 0.030
            "margin": "0.00000001",  # relatively good guess 0.0000001 or 0.000001
        })

        # compiler settings
        ET.SubElement(self.root, "compiler", {
            #"euler": "true",  # during compilation tells MuJoCo to create a convex hull around the mesh (encloses mesh)
        })


        # visual settings
        visual = ET.SubElement(self.root, "visual")
        ET.SubElement(visual, "global", offwidth="800", offheight="600")
        ET.SubElement(visual, "map", znear="0.0000001", zfar="100000")

        test_plane = ET.SubElement(self.worldbody, "geom", type="plane", pos="-1 0 0",
                                   quat="0.707 0 0.707 0", size="10 10 10", rgba="1 1 1 1")

        self._generate_linear_rack()
        self._generate_tooth_plates_mesh()

        # Add a custom camera, axisangle last value in degs
        ET.SubElement(self.worldbody, "camera", name="custom_cam2", fovy="0.01", pos="12 0.002 0.0005", quat="0.5 0.5 0.5 0.5")  # axisangle="0 1 0 90" - for vertical rack
        ET.SubElement(self.worldbody, "camera", name="custom_cam3", pos="0.0025 0.0015 0.0005", quat="0.5 0.5 0.5 0.5")  # axisangle="0 1 0 90" - for vertical rack
        ET.SubElement(self.worldbody, "camera", name="custom_cam4", pos="0.005 0.0004 0", quat="0.5 0.5 0.5 0.5")  # axisangle="0 1 0 90" - for vertical rack



        # Save the XML file
        ET.ElementTree(self.root).write("latest_model.xml")
        xml_text = ET.tostring(self.root, encoding="unicode")
        MSMLinear.xml_model = xml_text
        return xml_text

    def _generate_tooth(self):
        if self.tooth_type == "force_optimal":

            self.tooth_height = 0.49 * 1e-3  # meters, the max tooth height is 0.5 mm, but let's keep some margin
            self.t_b = self.tooth_height * np.tan(utils.tooth_alpha)  # meter
            self.t_a = 3e-6  # keep it non-zero for computational purposes
            self.tooth_pitch = 2 * (self.t_a + self.t_b)
            # self.feedback_threshold = 0.98  # 0.995  # the performance analysis simulation sran with this value (0.98)
            self.tooth_disengage_offset = 0.05
            self.new_tooth_engagement_offset = 0.00  # [0, +inf) defines the timing offset for a new tooth to engage
            half_y_grid_vec = np.array([[0.01 * self.t_a,
                                        self.t_a / 2,
                                        #t_b / 2 + t_a / 2,  # extra point for correct collision
                                        self.t_b + self.t_a / 2,
                                        #t_b + t_a / 3 + t_a / 2  # used it matlab, but unneccessary here
                                         ]])
            half_z_grid_vec = np.array([[-0.01 * self.tooth_height,
                                        0,
                                        #self.tooth_height / 2,  # extra point for correct collision
                                        self.tooth_height,  # self.tooth_height*0.99,
                                        #self.tooth_height  # used it matlab, but unneccessary here
                                         ]])
            half_x_grid_vec = np.zeros((1, len(half_y_grid_vec[0,:])))
            half_grid_mat = np.vstack((half_x_grid_vec, half_y_grid_vec, half_z_grid_vec))
            half_grid_mat = np.transpose(half_grid_mat)
        else:
            raise "error, unknown tooth type"

        self.contact_ratio = self.msm_elements_cnt / (2 * (self.t_a / self.t_b + 1))
        grid_mat = half_grid_mat
        for i in range(1, len(grid_mat[:, 0]) + 1):
            new_row = [0,
                       self.tooth_pitch - half_y_grid_vec[0, -i],
                       half_z_grid_vec[0, -i]]
            grid_mat = np.vstack((grid_mat, new_row))

        teeth_mat = grid_mat.copy()
        tooth_cnt = 10
        for i in range(-10, tooth_cnt):
            tmp_mat = grid_mat.copy()
            tmp_mat[:, 1] = tmp_mat[:, 1] + self.tooth_pitch * i
            teeth_mat = np.vstack((teeth_mat, tmp_mat))

        return teeth_mat, grid_mat

    def _generate_linear_rack(self):  # TODO add friction to the rack, friction computation settings must be set correctly
        # rack with teeth
        rack = ET.SubElement(self.worldbody, "body", name="rack", pos="0 0 0")

        ET.SubElement(rack, "joint", {
            "name": "rack_slider",
            "type": "slide",
            "axis": "0 1 0",
            "range": "-2 2",
            "frictionloss": str(utils.rack_static_friction),
            "solimpfriction": "0.95 0.9999 0.001 0.1 2",
            "solreffriction": "0.000075 0.030",
        })
        tooth_plate_profile = self.tooth_profile_mat.copy()
        tooth_plate_profile[2, :] = -tooth_plate_profile[2, :]

        spheres_cnt = len(self.teeth_matrix[:, 0])
        sphere_mass = utils.total_rack_mass / spheres_cnt
        # Generate spheres along the trajectory
        for i in range(spheres_cnt):
            x, y, z = self.teeth_matrix[i, :]
            pos = f"{x} {y} {z}"
            ET.SubElement(rack, "geom", {
                "type": "sphere",
                "size": "0.000015",
                "pos": pos,  # minimum working size="0.000015"
                "rgba": "1 0.3 0.3 1",
                "mass": str(sphere_mass),
                "contype": "1",  # Enable collision
                "conaffinity": "1",  # Respond to collisions
            })

        # Rack actuator is to apply useful load
        ET.SubElement(self.actuator, "general", joint="rack_slider", name="rack_actuator", gainprm="1")

    def set_rack_load(self, useful_load=None):
        if useful_load is not None:
            self.useful_load = useful_load
        self.data.ctrl[self.rack_actuator_id] = self.useful_load

    def _generate_tooth_plates_mesh(self):
        tooth_plate_width = 0.001
        vertices = np.zeros((2 * self.tooth_profile_mat.shape[0], 3))
        vertices[::2] = self.tooth_profile_mat
        vertices[1::2] = self.tooth_profile_mat
        vertices[::2, 0] -= tooth_plate_width / 2
        vertices[1::2, 0] += tooth_plate_width / 2

        faces_cnt = vertices.shape[0]
        faces = []
        for i in range(2, faces_cnt):
            faces.append([i-1, i, i-2])
            faces.append([i-1, i-2, i])

        # Create the mesh using trimesh
        plane_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        # create a convex hull around the mesh (encloses mesh)
        plane_mesh = plane_mesh.convex_hull
        tooth_file_name = "tooth_mesh.obj"

        # Save the mesh to an OBJ file
        plane_mesh.export(tooth_file_name)
        plane_mesh.update_faces(plane_mesh.unique_faces())
        plane_mesh.remove_unreferenced_vertices()
        plane_mesh.fix_normals()

        ET.SubElement(self.asset, "mesh", {
            "name": "tooth_plate_mesh",
            "file": tooth_file_name
        })

        # adding set of teeth

        for i in range(self.msm_elements_cnt):
            tooth_body = ET.SubElement(self.worldbody, "body", {
                "name": f"tooth_plate_{i}",
                "pos": f"0 {self.tooth_pitch*1.5 + 2*self.tooth_pitch*i + i*self.tooth_pitch / self.msm_elements_cnt} {2*self.tooth_height}",  # 0 {self.tooth_pitch/2} {2*self.tooth_height}  {-utils.contraction_initial}
                "axisangle": "0 1 0 180"
            })

            ET.SubElement(tooth_body, "geom", {
                "type": "mesh",
                "mesh": "tooth_plate_mesh",
                "size": "1 1 1",  # Scaling factor
                "contype": "1",  # Enable collision
                "conaffinity": "1",  # Respond to collisions
                "rgba": "0 1 0 1",
                "mass": str(utils.tooth_plate_mass)
            })

            spring_offset = -10e-3  # original matlab value might be different; tested here -5e-3
            if self.tb_type == 1:
                spring_stiffness = 3.40e+2
            elif self.tb_type == 2:
                spring_stiffness = 0.5e+2


            ET.SubElement(tooth_body, "joint", {
                "name": f"tooth_slider_{i}",
                "type": "slide",
                "axis": "0 0 1",
                "stiffness": str(spring_stiffness),
                "springref": str(spring_offset),
                "solimplimit": "0.95 0.9999 0.001 0.1 2",
                "solreflimit": "0.000075 0.100",
                "limited": "true",  # enables joint limits
                "range": f"{-0.075*self.tooth_height} 1",  # positive limit is towards the rack
                "frictionloss": str(self.coulomb_friction_ts + utils.tooth_plate_static_friction),
                "solimpfriction": "0.95 0.9999 0.001 0.1 2",
                "solreffriction": "0.000075 0.030",
            })

            ET.SubElement(self.actuator, "general", {
                "joint": f"tooth_slider_{i}",
                "name": f"tooth_slider_actuator_{i}",
                "gainprm": "1",
                #"ctrlrange": "0 1e-6",

            })

    def _compute_msm_force(self):
        """
        The function output resulting force value of msm crystal actuated by a magnetic field
        :param control_value_list: list of [0, 1] value for corresponding tooth plates.
        If 0 is provided, msm element is fully passive.
        If 1 is provided msm element is actuated by a saturating magnetic field value (0.7 T)
        :return:
        """

        for i in range(self.msm_elements_cnt):
            control_value = min(max(self.output_vector[i], 0), 1)  # value in range [0 1]
            msm_velocity = self.data.qvel[self.tooth_joint_id_list[i]]
            tb_velocity = msm_velocity / utils.e_0
            velocity_sign = np.sign(msm_velocity)
            msm_acceleration = self.data.qacc[self.tooth_joint_id_list[i]]

            dyn_mass = tb_velocity**2 + (utils.msm_mass / (utils.ro * utils.msm_A0) + self.data.qpos[self.tooth_joint_id_list[i]]
                       - self.msm_plates_init_pos_list[i]) * msm_acceleration
            dyn_mass *= utils.ro_k_0_cos_coeff


            if self.tb_type == 1:
                dynamic_ts = 0.81 * tb_velocity * 1e6
            elif self.tb_type == 2:
                dynamic_ts = 0.014 * (abs(tb_velocity) ** 1.4) * np.sign(tb_velocity) * 1e6

            resulting_force = utils.msm_A0 * (control_value * utils.constant_magnetic_stress
                                              - dynamic_ts - dyn_mass)

            self.data.ctrl[self.tooth_actuator_id_list[i]] = resulting_force - velocity_sign * self.coulomb_friction_ts

            # if abs(msm_velocity) < 1e-5 and abs(resulting_force) < self.coulomb_friction_ts:
            #     self.data.ctrl[self.tooth_actuator_id_list[i]] = 0
            # else:
            #     self.data.ctrl[self.tooth_actuator_id_list[i]] = resulting_force - velocity_sign * self.coulomb_friction_ts

    def _open_loop_controller(self):
        initial_idle_time = utils.simulation_timestep * 2
        time = self.data.time
        self.output_vector = np.zeros(self.msm_elements_cnt)
        if time < initial_idle_time:
            return

        cycle_length = 0.025
        cur_active = -1
        frac, integer = math.modf(abs(time / cycle_length))
        for i in range(1, self.msm_elements_cnt+1):  # 1:tooth_plates_cnt
            if frac < i / self.msm_elements_cnt:
                self.output_vector[i-1] = 1
                cur_active = i - 1
                cur_active_output = i-1
                break
        self.output_vector[cur_active_output] = 1
        self.output_vector[self._get_following_active_vec(cur_active)] = 1
        #  print(output_vector)
        #self._compute_msm_force(output_vector)

    def _closed_loop_controller(self):
        cur_offset = (self.data.qpos[self.rack_joint_id]  # rack current pos - rack initial pos
                      - self.rack_init_pos  #+ self.tooth_pitch * 0.25 #+ (
                                 # 1 - self.feedback_threshold) * self.tooth_pitch  # second term is for modifying the engagement timing
                      + 100 * self.tooth_pitch)
        cur_cycle, _ = math.modf(abs(cur_offset) / self.tooth_pitch)
        engagement_range = self.t_b / self.tooth_pitch + self.t_a / self.tooth_pitch - self.tooth_disengage_offset
        for i in range(self.msm_elements_cnt):
            top_switch_threshold = i / self.msm_elements_cnt
            low_switch_threshold = top_switch_threshold - engagement_range
            top_switch_threshold -= self.t_a / self.tooth_pitch + self.tooth_engage_offset
            # if low_switch_threshold < 0:  # wtf for some reason this approach does not replace the second part of the following condition
            #     top_switch_threshold += 1
            #     low_switch_threshold += 1
            if (low_switch_threshold < cur_cycle < top_switch_threshold) \
                    or (low_switch_threshold < 0 and 1 + low_switch_threshold < cur_cycle < 1 + top_switch_threshold):
                self.output_vector[i] = self.control_value
            else:
                self.output_vector[i] = 0

        # TODO this part of algorithm is implemented currently only in one direction
        # Engaging inactive teeth if control value is to low (to prevent rack from sliding away)
        if self.enable_blocking and self.control_value < utils.CONTROL_VALUE_BLOCKING_LIMIT:
            blocking_value = ((utils.CONTROL_VALUE_BLOCKING_LIMIT - self.control_value) *
                              utils.MAX_BLOCKING_OUTPUT_VALUE / utils.CONTROL_VALUE_BLOCKING_LIMIT)
            self.output_vector[self.output_vector < 1e-3] = blocking_value

        return
        # --- old algorithm ---
        currently_active = 0
        for i in range(1, self.msm_elements_cnt):
            cur_offset = (self.data.qpos[self.rack_joint_id]  # rack current pos - rack initial pos
                          - self.rack_init_pos + (1 - self.feedback_threshold) * self.tooth_pitch  # second term is for modifying the engagement timing
                          + 100*self.tooth_pitch)  # third term is to make sure the difference between first terms is always positive (not the best way to implement it
            cur_cycle, _ = math.modf(abs(cur_offset) / self.tooth_pitch)
            # switch_threshold = (i - (1 - self.feedback_threshold)) / self.msm_elements_cnt  # the substraction is to eliminate dead points
            switch_threshold = i / self.msm_elements_cnt  # the substraction is to eliminate dead points
            if cur_cycle < switch_threshold:
                currently_active = i
                break

        next_active_vec = self._get_following_active_vec(currently_active,
                                                         contact_ratio=self.contact_ratio,  # CR is considered
                                                         fraction=self.new_tooth_engagement_offset)
        self.output_vector = np.zeros(self.msm_elements_cnt)
        self.output_vector[next_active_vec] = self.control_value
        self.output_vector[currently_active] = self.control_value

        # TODO this part of algorithm is implemented currently only in one direction
        # Engaging inactive teeth if control value is to low (to prevent rack from sliding away)
        if self.enable_blocking and self.control_value < utils.CONTROL_VALUE_BLOCKING_LIMIT:
            blocking_value = ((utils.CONTROL_VALUE_BLOCKING_LIMIT - self.control_value) *
                              utils.MAX_BLOCKING_OUTPUT_VALUE / utils.CONTROL_VALUE_BLOCKING_LIMIT)
            self.output_vector[self.output_vector < 1e-3] = blocking_value

    def _get_following_active_vec(self, cur_active, velocity_direction=None, contact_ratio=None, fraction=0.0):
        """
        Function computes vec of currently active MSM elements
        :param cur_active:
        :param velocity_direction: Reserved, not implemented
        :param contact_ratio:
        :param fraction:
        :return:
        """
        iteration_range_start = cur_active # (cur_active + 1)
        if contact_ratio is None:
            iteration_range_end = (cur_active + round(self.msm_elements_cnt / 2) - 1)
        else:
            cur_offset = (self.data.qpos[self.rack_joint_id]  # rack current pos - rack initial pos
                          - self.rack_init_pos + 100*self.tooth_pitch)
            cur_cycle, _ = math.modf(abs(cur_offset) / self.tooth_pitch)
            cur_cycle, _ = math.modf(cur_offset * self.msm_elements_cnt)
            iteration_range_end = math.floor(cur_active + contact_ratio - 1 + fraction)
            if iteration_range_end == 0:
                return []
            iteration_range_end += cur_active

            # old version
            # contact_ration_frac, contact_ration_int = math.modf(contact_ratio)
            # fraction_tooth = math.ceil(contact_ration_frac - fraction - 1e-6)
            # iteration_range_end = (cur_active + round(contact_ration_int) - 1 + fraction_tooth)
        following_active_vec = []
        for i in range(iteration_range_start, iteration_range_end+1):
            new_active_val = np.mod(i, self.msm_elements_cnt)
            # if new_active_val < cur_active:  # this condition is from matlab
            #     new_active_val += 1
            following_active_vec.append(new_active_val)
        return following_active_vec

    def _initiate_tooth_plate_ids(self):
        for i in range(self.msm_elements_cnt):
            self.tooth_actuator_id_list.append(mujoco.mj_name2id(self.model,
                                                                 mujoco.mjtObj.mjOBJ_ACTUATOR,
                                                                 f"tooth_slider_actuator_{i}"))
            self.tooth_joint_id_list.append(mujoco.mj_name2id(self.model,
                                                              mujoco.mjtObj.mjOBJ_JOINT,
                                                              f"tooth_slider_{i}"))
            self.msm_plates_init_pos_list.append(self.data.qpos[self.tooth_joint_id_list[i]])

    def _collect_controller_data(self):
        """
        Function collects simulation data with frequency equal to a controller frequency
        :return:
        """
        self.simulation_data["time"] = np.append(self.simulation_data["time"], self.data.time)
        self.simulation_data["rack_pos"] = np.append(self.simulation_data["rack_pos"], self.data.qpos[self.rack_joint_id])
        self.simulation_data["rack_vel"] = np.append(self.simulation_data["rack_vel"], self.data.qvel[self.rack_joint_id])
        self.simulation_data["rack_acc"] = np.append(self.simulation_data["rack_acc"], self.data.qacc[self.rack_joint_id])
        self.simulation_data["rack_tanh_acc"] = np.append(self.simulation_data["rack_tanh_acc"],
                                                          math.tanh(0.001*self.data.qacc[self.rack_joint_id]))
        self.simulation_data["control_value"] = np.append(self.simulation_data["control_value"], self.control_value)

        rack_phase = (self.data.qpos[self.rack_joint_id]  # rack current pos - rack initial pos
                      - self.rack_init_pos + 100 * self.tooth_pitch)
        rack_phase, _ = math.modf(abs(rack_phase) / self.tooth_pitch)
        self.simulation_data["rack_phase"] = np.append(self.simulation_data["rack_phase"], rack_phase)

    def onstep_computation(self, model=None, data=None):
        """
        Function to compute all on-step actions in simulation
        :return:
        """
        self.controller_func()
        self._compute_msm_force()
        # self._collect_controller_data(collect_on_each_step=False)

    def sim_step(self, action):

        self.control_value = action
        if self.control_on_callback:
            mujoco.mj_step(self.model, self.data, nstep=self.steps_per_call)
        else:
            for _ in range(self.steps_per_call):
                self.onstep_computation()
                mujoco.mj_step(self.model, self.data, nstep=1)
        # self.onstep_computation()  # NOTE: steps_per_call must be equal to 1 for the function to work normally
        self._collect_controller_data()


    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Optionally set qpos/qvel to specific values
        # self.data.qpos[:] = np.zeros_like(self.data.qpos)
        # self.data.qvel[:] = np.zeros_like(self.data.qvel)
        self.set_rack_load()
        self.simulation_data = {
            "time": np.zeros(utils.sequence_length),  # np.array([]),
            "rack_pos": np.zeros(utils.sequence_length),
            "rack_vel": np.zeros(utils.sequence_length),
            "rack_acc": np.zeros(utils.sequence_length),
            "rack_tanh_acc": np.zeros(utils.sequence_length),
            "rack_phase": np.zeros(utils.sequence_length),
            "control_value": np.zeros(utils.sequence_length),
        }
        self._collect_controller_data()

    def plot_rack_instant_velocity(self):
        time_vec = self.simulation_data["time"][1:]
        instant_vel_vec = self.simulation_data["rack_vel"][1:]
        # Plot results
        plt.figure(figsize=(8, 4))
        plt.plot(time_vec, instant_vel_vec, label="Rack Velocity, m/s")
        plt.xlabel("Time (s)")
        plt.ylabel("Instant Velocity")
        plt.title("Instant velocity Over Time")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_rack_average_velocity(self):
        time_vec = self.simulation_data["time"][(utils.sequence_length + 1):]
        vel_data = np.divide(self.simulation_data["rack_pos"][(utils.sequence_length + 1):], time_vec)
        # Plot results
        plt.figure(figsize=(8, 4))
        plt.plot(time_vec, vel_data, label="Average Velocity (m/s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Average Velocity")
        plt.title("Joint Position Over Time")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_tb_velocity(self):
        time_vec = self.simulation_data["time"][1:]
        instant_vel_vec = self.simulation_data["rack_vel"][1:]
        vel_data = np.divide(self.simulation_data["rack_pos"][1:], time_vec)
        # Plot results
        plt.figure(figsize=(8, 4))
        plt.plot(time_vec, instant_vel_vec, label="Joint Velocity (qvel)")
        plt.xlabel("Time (s)")
        plt.ylabel("Average Velocity")
        plt.title("Joint Position Over Time")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_control_value(self):
        time_vec = self.simulation_data["time"][1:]
        instant_vel_vec = self.simulation_data["control_value"][1:]
        # Plot results
        plt.figure(figsize=(8, 4))
        plt.plot(time_vec, instant_vel_vec, label="Control value")
        plt.xlabel("Time (s)")
        plt.ylabel("Control value")
        plt.title("Control value Over Time")
        plt.legend()
        plt.grid()
        plt.show()

class MSM_Environment(gym.Env):

    reward_list = []

    @staticmethod
    def plot_expected_reward_history(external_reward_list=None):
        if external_reward_list is not None:
            reward_list = external_reward_list
        elif len(MSM_Environment.reward_list) < 2:
            print("reward history is empty, nothing to plot. Reward list entries: ", MSM_Environment.reward_list)
            return
        else:
            reward_list = MSM_Environment.reward_list
        step_vec = [i for i in range(1, len(reward_list))]
        # Plot results
        plt.figure(figsize=(8, 4))
        plt.plot(step_vec, reward_list[1:], label="Expected reward")
        plt.xlabel("Run number")
        plt.ylabel("Expected reward")
        plt.title("Reward Over runs")
        plt.legend()
        plt.grid()
        plt.show()

    def __init__(self, randomize_setpoint=True, return_observation_sequence=False):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(utils.features_cnt,), dtype=np.float32)
        self.environment = MSMLinear(tb_type=1,
                                controller_type="closed_loop")

        self.simulation_time = 0.2  # seconds
        self.velocity_setpoint = 0.012
        self.velocity_setpoint_list = np.array([])
        self.randomize_setpoint = randomize_setpoint
        self.cur_step = 1
        self.total_reward = 0
        self.return_observation_sequence = return_observation_sequence  # if True, obs matrix will be returned with sequence lengths equal to one defined in utils

        if self.randomize_setpoint:
            print("random setpoint option is set")
        else:
            print(f"a constant setpoint will be used with value {self.velocity_setpoint}")

    def _get_observation(self):
        if self.return_observation_sequence:
            pass  # TODO
        else:
            obs = [
                           self.environment.simulation_data["rack_phase"][-1],
                           self.environment.simulation_data["rack_vel"][-1],
                           self.environment.simulation_data["rack_tanh_acc"][-1],
                           self.velocity_setpoint
                           ]
        return obs

    def step(self, action):
        info = {}
        self.environment.sim_step(action)
        observation = self._get_observation()
        # reward = -(self.velocity_setpoint - self.environment.simulation_data["rack_vel"][-1]) ** 2
        reward = -( ((self.velocity_setpoint - self.environment.simulation_data["rack_vel"][-1]) * 1000) **2 ) \
                 * (1 + abs(float(action)))

        # in gym truncated is responsible for termination based on the time steps limit.
        # But since there is no other termination option in MSM sim (termination based on rack flying away is too rare), using termination flag is reasonable here
        # Most of RL algorithms update after termination
        terminated = (self.environment.simulation_data["time"][-1] > self.simulation_time)
        truncated = False  # TODO check if that is a correct approach

        if abs(self.environment.simulation_data["rack_pos"][-1]) > 0.3:
            terminated = True
            reward -= 1000  # extra penalty for dropping the rack
            print("rack reached the motion limit")


        self.total_reward += reward
        self.cur_step += 1

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # print("reset function is called")
        self.environment.reset()
        info = {}
        if self.randomize_setpoint:
            self.velocity_setpoint = random.random() * 0.013 + 0.002
        observation = self._get_observation()

        self.velocity_setpoint_list = np.zeros(utils.sequence_length)
        self.velocity_setpoint_list = np.append(self.velocity_setpoint_list, self.velocity_setpoint)

        avg_reward = self.total_reward / self.cur_step
        MSM_Environment.reward_list.append(avg_reward)
        # utils.reward_list.append(avg_reward)
        utils.reward_list = np.append(utils.reward_list, avg_reward)
        #print("Average reward for the iteration: ", avg_reward)
        self.total_reward = 0
        self.cur_step = 1

        return observation, info

    def render(self):
        print('render function was called')

    def close(self):
        print('close function was called')
        #sim_plant.SimPool.delete_instance(self.plant)

if __name__ == '__main__':
    # Generate the model
    test_model = MSMLinear()

