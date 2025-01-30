import warnings

import matlab.engine
import time
import matplotlib.pyplot as plt
import numpy as np
import utils
import keras
import os
import tensorflow as tf
import pygad
from simple_pid import PID
import plotly.graph_objects as go
import webbrowser
import random

class SimulinkPlant:

    scaler = None

    @staticmethod
    def create_scaler():
        # X, y = utils.get_pid_data(utils.sequence_length, 2000,
        #                           print_time=True)  # generate_data(sequence_length, num_files)
        _, scaler = utils.normalize_data(None, load_scaler=True, save_scaler=False)
        SimulinkPlant.scaler = scaler

    def __init__(self, modelName='MSM_strain_wave_actuator',
                 path='C:\\Users\\h19343\\files\\PhD\\MSM actuators simualtion\\Simscape model',
                 scaler=None,
                 debug_mode=False,
                 model_index=None,
                 controller="nn"):
        """
        :param modelName:
        :param path:
        :param scaler:
        :param debug_mode: if True, enables sim visualization, adds reporting to console
        :param model_index: if None, then the original model MSM_strain_wave_actuator.slx will be opened,
        otherwise a copy of the model with the corresponding index will be used
        :param controller: defines controller type used, options "nn" for neural network and "pid" for PID controller
        """
        self.modelName = modelName  # The name of the Simulink Model (To be placed in the same directory as the Python Code)
        self.path = path
        # Logging the variables
        self.data_dict = {}  # "time" "position" "velocity" "tahn_acceleration" "desired_velocity" "new_pid_value"
        self.steps_cnt = 0
        self.model = None  # self.get_nn_model()
        self.tooth_pitch = 0
        self.pygad_solution = None
        self.eng = None
        self.debug_mode = debug_mode
        self.controller = controller
        self.velocity_setpoint = -0.005  # -0.01
        if scaler is None:
            if SimulinkPlant.scaler is None:
                SimulinkPlant.create_scaler()
        else:
            SimulinkPlant.scaler = scaler

        self.connect_to_matlab()
        self.set_static_simulation_params(model_index)
        if self.controller == "pid":
            self.pid = self.init_pid_controller()

    def get_nn_model(self, folder='pid_neural_networks', controller_name="test_controller_tf_2_10"):  # controller_name="test_controller"
        model = keras.models.load_model(os.path.join(folder, controller_name))
        return model

    def init_pid_controller(self):
        pid = PID(6, 20000, 8e-5)
        pid.sample_time = utils.NN_WORKING_PERIOD
        pid.proportional_on_measurement = False
        pid.differential_on_measurement = False

        return pid
    def set_nn_model(self, model):
        self.model = model

    def connect_to_matlab(self):
        print("Starting matlab")
        t0 = time.time()
        self.eng = matlab.engine.start_matlab()  # TODO here can be a string "-desktop" argument. What does it do??? Also background option is available
        print('connected to matlab in: ', time.time() - t0, ' seconds')

    def set_static_simulation_params(self, model_index=None):
        # the function called once during the plant initialization
        if  model_index is not None:  # call a copy of the model if model index is provided
            self.modelName = self.modelName + str(model_index)
        print("uploaded model name: ", self.modelName)
        self.eng.eval("addpath('{}')".format(self.path), nargout=0)
        if self.debug_mode:
            self.eng.eval("enable_model_visualization(true);", nargout=0)
        else:
            self.eng.eval("enable_model_visualization(false);", nargout=0)
        self.eng.eval("model = '{}';".format(self.modelName), nargout=0)
        self.eng.eval("load_system(model)", nargout=0)

    def set_desired_velocity(self, type="random"):
        if type == "random":
            self.velocity_setpoint = -random.random() * 0.009 - 0.001
        else:
            self.velocity_setpoint = None

    def init_simulation_params(self):
        self.eng.eval("simulation_pause_time = {:F};".format(utils.NN_WORKING_PERIOD), nargout=0)
        self.eng.eval("has_external_controller = {};".format(1), nargout=0)
        self.eng.eval("external_control_value = {:F};".format(0.5), nargout=0)
        self.eng.eval("SIMULATION_TIME = {:F};".format(0.03), nargout=0)  # format(0.02)  format(0.02)
        self.eng.eval("controller_velocity_setpoint = {:F};".format(self.velocity_setpoint), nargout=0)  # format(-0.010)
        self.eng.eval("MSM_mechanism_simulation_properties", nargout=0)

        self.tooth_pitch = self.eng.eval("tooth_pitch;", nargout=1)
        self.eng.eval("model = '{}';".format(self.modelName), nargout=0)
        #self.eng.eval("load_system(model)", nargout=0)
        self.eng.set_param(self.modelName, 'SimulationCommand', 'start', 'SimulationCommand', 'pause', nargout=0)

    def reset_plant(self):
        if self.debug_mode:
            print("reseting system")
        #self.eng.eval("if exist('model','var'); close_system(model); end", nargout=0)
        self.eng.eval("clear", nargout=0)
        self.init_simulation_params()
        self.steps_cnt = 0
        self.data_dict = {}

    def disconnect(self):
        self.eng.set_param(self.modelName, 'SimulationCommand', 'stop', nargout=0)
        self.eng.quit()

    def get_controller_input(self, pygad_solution=None):

        feature_columns = ['mod_rack_position', 'rack_position', 'rack_velocity', 'tahn_rack_acceleration',
                           'desired_velocity']
        pos = np.array(self.data_dict["position"])[-utils.sequence_length:]  # .reshape(series_length, 1)
        mod_pos = np.mod(pos, self.tooth_pitch)
        vel = np.array(self.data_dict["velocity"])[-utils.sequence_length:]  # .reshape(series_length, 1)
        acc = np.array(self.data_dict["tahn_rack_acceleration"])[-utils.sequence_length:]  #  .reshape(series_length, 1)
        desired_vel = np.array(self.data_dict["desired_velocity"])[-utils.sequence_length:]  #  .reshape(series_length, 1)
        data = np.hstack((mod_pos, vel, acc, desired_vel))  # Five features used previously: np.hstack((mod_pos, pos, vel, acc, desired_vel))
        data = data.reshape(1, pos.shape[0], utils.features_cnt)
        if self.controller == "pid":
            self.pid.setpoint = 0
            # print("pid setpoint: ", self.pid.setpoint, " ; input data : ", vel[-1])
            prediction = self.pid(-vel[-1] + desired_vel[-1], dt=utils.NN_WORKING_PERIOD)
            # print("pid prediction: ", prediction)
        elif self.controller == "nn":
            data, _ = utils.normalize_data(data, scaler=SimulinkPlant.scaler)
            padded_data = tf.keras.utils.pad_sequences(data, maxlen=utils.sequence_length, padding='pre', dtype='float32')  # pre padding is correct. post padding adds zeros to end
            if not (pygad_solution is None):
                prediction = pygad.kerasga.predict(model=self.model,
                                                   solution=pygad_solution,
                                                   data=padded_data)
            else:
                prediction = self.model.predict(padded_data, verbose=0)  # this approach works faster than prediction with pydag function
            # print('prediction: ', prediction)
            # float(control_signal[-1][-1][-1])
            if np.isnan(float(prediction[-1][-1][-1])):
                print('control signal is nan')
            prediction = float(prediction[-1][-1][-1])  # original shape is (1, series_length, 1)
        else:
            warnings.warn("Controller type setup incorrectly for the simplant")
            prediction = None

        return prediction  # outputting only latest data

    def get_sim_data(self):
        time_vec = self.eng.eval("out.sim_results.Time;", nargout=1)
        idx_list = [0]
        if isinstance(time_vec, float):
            self.data_dict["time"] = [time_vec]
            self.data_dict["position"] = [self.eng.eval("out.sim_results.Data(:, 1);", nargout=1)]
            self.data_dict["velocity"] = [self.eng.eval("out.sim_results.Data(:, 2);", nargout=1)]
            self.data_dict["tahn_rack_acceleration"] = [self.eng.eval("out.controller_state.Data(:, 6);", nargout=1)]
            self.data_dict["desired_velocity"] = [self.eng.eval("out.controller_state.Data(:, 7);", nargout=1)]
            self.data_dict["new_pid_value"] = [self.eng.eval("out.controller_state.Data(:, 3);", nargout=1)]
            return
        time_vec = np.array(time_vec)
        for i in range(len(time_vec)):  # getting the datapoints with the frequency of the controller
            if time_vec[i] > time_vec[idx_list[-1]] + utils.NN_WORKING_PERIOD:
                idx_list.append(i)
        time_vec = np.array(time_vec)
        self.data_dict["time"] = time_vec[idx_list]
        self.data_dict["position"] = np.array(self.eng.eval("out.sim_results.Data(:, 1);", nargout=1))[idx_list]
        self.data_dict["velocity"] = np.array(self.eng.eval("out.sim_results.Data(:, 2);", nargout=1))[idx_list]
        self.data_dict["tahn_rack_acceleration"] = np.array(self.eng.eval("out.controller_state.Data(:, 6);", nargout=1))[idx_list]
        self.data_dict["desired_velocity"] = np.array(self.eng.eval("out.controller_state.Data(:, 7);", nargout=1))[idx_list]
        self.data_dict["new_pid_value"] = np.array(self.eng.eval("out.controller_state.Data(:, 3);", nargout=1))[idx_list]

        # #Helper Function to get Plant Output and Time History
        # self.data_dict["time"] = self.eng.eval("out.sim_results.Time;", nargout=1)
        # # sim_results: Data 1 - position, Data 2 - velocity
        # self.data_dict["position"] = self.eng.eval("out.sim_results.Data(:, 1);", nargout=1)
        # self.data_dict["velocity"] = self.eng.eval("out.sim_results.Data(:, 2);", nargout=1)
        # # controller_state: 1: forward pid; 2 reverse pid; 3 new pid value; 4 initial pid value;
        # # 5 controller switch spike; 6 tahn(acc*0.001); 7 desired velocity
        # self.data_dict["tahn_rack_acceleration"] = self.eng.eval("out.controller_state.Data(:, 6);", nargout=1)
        # self.data_dict["desired_velocity"] = self.eng.eval("out.controller_state.Data(:, 7);", nargout=1)
        # self.data_dict["new_pid_value"] = self.eng.eval("out.controller_state.Data(:, 3);", nargout=1)

    def get_last_datapoint_or_zero(self):
        # TODO
        pass
        # if self.data_dict:
        #     feature_columns = ['mod_rack_position', 'rack_position', 'rack_velocity', 'tahn_rack_acceleration',
        #                        'desired_velocity']

    def step(self, action):  # TODO
        pass
        """
        if self.eng.get_param(self.modelName, 'SimulationStatus') != ('stopped' or 'terminating'):
            observation = np.float32(np.array([0, 0, 0, 0, 0]))  # TODO put some reasonable check if we have any data to put into observations otherwise fill with zeros

        # TODO need to set control action, get observation, set simulation to continue

        while self.eng.get_param(self.modelName, 'SimulationStatus') != 'paused' or \
                (self.eng.get_param(self.modelName, 'SimulationStatus') != ('stopped' or 'terminating')):
            time.sleep(0.001)  # waiting for the step to be finished


        return observation, reward, terminated, truncated
        """

    def run_simulation(self, pygad_solution=None):
        """
        Runs a complete simulation for SIMULATION_TIME duration or until the rack is stuck
        :param pygad_solution: if pygad_solution instance is provided, then pydag function is used to compute NN output, works slower than computation with TF
                               if pydag_solution is None, then TF function is used to compute NN output
        :return: returns average fitness value as MSE of velocity tracking, velocity points are obtained with the frequency of the controller.
        """
        t0 = time.time()
        while (self.eng.get_param(self.modelName, 'SimulationStatus') != ('stopped' or 'terminating')):
            if self.eng.get_param(self.modelName, 'SimulationStatus') == 'paused':
                # Pause the Simulation for each timestep
                #self.eng.set_param(self.modelName, 'SimulationCommand', 'continue', 'SimulationCommand', 'pause', nargout=0)
                self.eng.eval("simulation_pause_time = {:F};".format(utils.NN_WORKING_PERIOD * (self.steps_cnt + 1)), nargout=0)
                self.get_sim_data()
                if self.steps_cnt > 0:  # not enough data on the 0th iteration
                    control_signal = self.get_controller_input(pygad_solution)

                    # if np.isnan(control_signal):
                    #     print('NAN controller signal is obtained, stopping simulation')
                    #     self.eng.set_param(self.modelName, 'SimulationCommand', 'stop', nargout=0)
                    #     return -float(10e10)  # returning very small fitness

                    self.eng.eval("external_control_value = {:F};".format(float(control_signal)), nargout=0)
                self.steps_cnt += 1
                self.eng.set_param(self.modelName, 'SimulationCommand', 'update', 'SimulationCommand', 'continue', nargout=0)
        unequal_sign_cnt = np.sum((np.sign(self.data_dict["velocity"]) != np.sign(self.data_dict["desired_velocity"])).astype(int))
        mse = -np.mean(np.square(self.data_dict["velocity"] - self.data_dict["desired_velocity"]))
        fitness = mse + mse * unequal_sign_cnt
        #fitness = fitness[-1]

        return fitness
        print('Simulation finished, simulation time: ', time.time() - t0, ' seconds')

class SimPool:
    pool_state_list = []
    use_separate_matlab_simulation_file = False

    @staticmethod
    def initialize():
        idx = None
        if SimPool.use_separate_matlab_simulation_file:
            idx = len(SimPool.pool_state_list) + 1
        instance = SimulinkPlant(debug_mode=False, model_index=idx)
        state_dict = {"instance": instance, "is_available": True}
        SimPool.pool_state_list.append(state_dict)

    @staticmethod
    def get_instance():
        # check if there is no available instance
        available_list = [x for x in SimPool.pool_state_list if x["is_available"]]
        if len(available_list) == 0:
            SimPool.initialize()
            SimPool.pool_state_list[-1]["is_available"] = False
            print('New SimPlant instance required, updated total number of instances: ', len(SimPool.pool_state_list))
            return SimPool.pool_state_list[-1]["instance"]
        else:
            available_list[0]["is_available"] = False
            return available_list[0]["instance"]

    @staticmethod
    def delete_instance(instance):
        # deletes the instance from pool, reducing the pool size
        for i in range(len(SimPool.pool_state_list)):
            if instance == SimPool.pool_state_list[i]["instance"]:
                del SimPool.pool_state_list[i]
                print('A SimPlant instance was deleted from the pool, updated total number of instances: ',
                      len(SimPool.pool_state_list))
                return

        print("Error, the instance to be cleared is not found in the SimPool")


    @staticmethod
    def release_instance(instance):
        # thorws an error if the instance is not in the list
        for i in range(len(SimPool.pool_state_list)):
            if instance == SimPool.pool_state_list[i]["instance"]:
                #instance.clear()
                SimPool.pool_state_list[i]["is_available"] = True
                return

        print("Error, the instance to be cleared is not found in the SimPool")


#"""
model = utils.get_nn_model(folder="pygad_networks_py_12", controller_name="-0.0022440439_fitness.keras")
t0 = time.time()


# plant = SimulinkPlant(common_debug_mode=False, controller="pid")
# plant.reset_plant()
# plant.set_nn_model(model)
# fitness = plant.run_simulation()
# # data collecting
# pid_pos = np.array(plant.data_dict["position"]).ravel()
# pid_time = np.array(plant.data_dict["time"]).ravel()
# pid_vel = np.array(plant.data_dict["velocity"]).ravel()
# pid_desired_vel = np.array(plant.data_dict["desired_velocity"]).ravel()
# pid_control_value = np.array(plant.data_dict["new_pid_value"]).ravel()
# # plant.set_nn_model(model)
# #plant.reset_plant()
# #fitness = plant.run_simulation()
# plant.disconnect()
# print('PID: len pos ', len(pid_pos), ' len vel ', len(pid_vel), ' len time', len(pid_time))
# print('plant control calls: ', plant.steps_cnt, ' desired control calls: ', 0.04 / utils.NN_WORKING_PERIOD)
# print('plant fitness: ', fitness)
pid_pos = np.array((0, 0))
pid_time = np.array((0, 0))
pid_vel = np.array((0, 0))
pid_desired_vel = np.array((0, 0))
pid_control_value = np.array((0, 0))


plant = SimulinkPlant(debug_mode=True, controller="nn")
plant.reset_plant()
t0 = time.time()
plant.set_nn_model(model)
fitness = plant.run_simulation()
# data collecting
nn_pos = np.array(plant.data_dict["position"]).ravel()
nn_time = np.array(plant.data_dict["time"]).ravel()
nn_vel = np.array(plant.data_dict["velocity"]).ravel()
nn_desired_vel = np.array(plant.data_dict["desired_velocity"]).ravel()
nn_control_value = np.array(plant.data_dict["new_pid_value"]).ravel()
# plant.set_nn_model(model)
#plant.reset_plant()
#fitness = plant.run_simulation()
plant.disconnect()
print('NN: len pos ', len(nn_pos), ' len vel ', len(nn_vel), ' len time', len(nn_time))
print('NN pos value: ', nn_pos)
print('plant control calls: ', plant.steps_cnt, ' desired control calls: ', 0.04 / utils.NN_WORKING_PERIOD)
print('plant fitness: ', fitness)
# opening plots in firefox
browser_path = 'C:\\Program Files\\Mozilla Firefox\\firefox.exe'
webbrowser.register('f_fox', None, webbrowser.BackgroundBrowser(browser_path))



# create plotly traces for plotting
trace_pid_vel = go.Scatter(x=pid_time, y=pid_vel, mode='lines', name='PID velocity')
trace_nn_vel = go.Scatter(x=nn_time, y=nn_vel, mode='lines', name='NN velocity')
trace_desired_vel = go.Scatter(x=nn_time, y=nn_desired_vel, mode='lines', name='Desired velocity')
# Create the figure and add traces
fig = go.Figure()
fig.add_trace(trace_pid_vel)
fig.add_trace(trace_nn_vel)
fig.add_trace(trace_desired_vel)

# Update layout for readability
fig.update_layout(
    title="Velocity",
    xaxis_title="Time",
    yaxis_title="NN velocity vs PID velocity",
    legend=dict(x=0.1, y=0.9)
)
html_file = 'temp_plot.html'
fig.write_html(html_file)
webbrowser.get('f_fox').open_new_tab(f'file:\\{os.path.abspath(html_file)}')


trace_pid_pos = go.Scatter(x=pid_time, y=pid_pos, mode='lines', name='PID position')
trace_nn_pos = go.Scatter(x=nn_time, y=nn_pos, mode='lines', name='NN position')
fig = go.Figure()
fig.add_trace(trace_pid_pos)
fig.add_trace(trace_nn_pos)
# Update layout for readability
fig.update_layout(
    title="Position",
    xaxis_title="Time",
    yaxis_title="NN position vs PID position",
    legend=dict(x=0.1, y=0.9)
)
html_file = 'temp_plot1.html'
fig.write_html(html_file)
webbrowser.get('f_fox').open_new_tab(f'file:\\{os.path.abspath(html_file)}')


trace_pid_control_value = go.Scatter(x=pid_time, y=pid_control_value, mode='lines', name='PID control value')
trace_nn_control_value = go.Scatter(x=nn_time, y=nn_control_value, mode='lines', name='NN control value')
fig = go.Figure()
fig.add_trace(trace_pid_control_value)
fig.add_trace(trace_nn_control_value)
# Update layout for readability
fig.update_layout(
    title="Control input",
    xaxis_title="Time",
    yaxis_title="NN control value vs PID control value",
    legend=dict(x=0.1, y=0.9)
)
html_file = 'temp_plot2.html'
fig.write_html(html_file)
webbrowser.get('f_fox').open_new_tab(f'file:\\{os.path.abspath(html_file)}')


# plt.plot(np.array(plant.data_dict["position"]))
# plt.title('absolute position')
# plt.ylabel('value')
# plt.xlabel('index')
# plt.show()
#
# plt.plot(np.array(plant.data_dict["time"]), np.array(plant.data_dict["velocity"]))
# plt.plot(np.array(plant.data_dict["time"]), np.array(plant.data_dict["desired_velocity"]))
# plt.title('velocity')
# plt.ylabel('value, ms')
# plt.xlabel('time, seconds')
# plt.legend(['real velocity', 'desired velocity'], loc='upper left')
# plt.show()
#
# plt.plot(np.array(plant.data_dict["time"]), np.array(plant.data_dict["new_pid_value"]))
# plt.title('PID')
# plt.ylabel('value')
# plt.xlabel('index')
# plt.show()
#"""