import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from keras.layers import Dense
# from keras.models import Sequential
import keras
import pygad.kerasga
import pygad
#import utilities
from datetime import datetime
import logging
import logging.config
import sim_plant
import os
import sys
import tensorflow as tf
import time
import msm_model
# import multiprocessing




def fitness_func(self, solution, sol_idx):
    global keras_ga, model, ga_instance, total_simulation_runs, best_fitness, network_folder
    str_iteration_data = "gen_" + str(ga_instance.generations_completed) + "_solution_idx_" + str(sol_idx)
    #print("simulation", str_iteration_data, datetime.now().strftime("%d/%model/%Y %H:%M:%S"))
    tmp_model = tf.keras.models.clone_model(model)
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=tmp_model,
                                                                 weights_vector=solution)  # model=model
    tmp_model.set_weights(weights=model_weights_matrix)
    #try:
    if use_mujoco:  # mujoco simulation will be run instead of matlab
        env = msm_model.MSM_Environment()
        obs = env.reset()
        done = False
        while not done:
            prediction = TODO
            env.step(prediction)


    else:  # matlab sim
        if True:
            plant = sim_plant.SimPool.get_instance()
            plant.set_nn_model(tmp_model)
            fitness_list = []
            for i in range(2):
                if i == 0:
                    plant.set_desired_velocity(type="random_low")
                else:
                    plant.set_desired_velocity(type="random_high")
                plant.reset_plant()
                tmp_fitness = plant.run_simulation()  # pygad_solution=solution
                fitness_list.append(tmp_fitness)
            fitness = np.mean(fitness_list)
            sim_plant.SimPool.release_instance(plant)
            str_exec_info = ' mean fitness: ' + str(fitness) + ' , steps in simulation : ' + str(plant.steps_cnt) + ', date: ' + datetime.now().strftime("%d/%model/%Y %H:%M:%S")
            print('solution index: ', sol_idx, str_exec_info)
            logging.info(str_iteration_data + str_exec_info)
        #except Exception as error:
        else:
            error_msg = "error has occurred during the simulation"
            logging.info(error_msg)
            logging.error(error)
            print(error_msg)
            print(error)
            fitness = -1
            sim_plant.SimPool.delete_instance(plant)

    if fitness > best_fitness:  # saving the model if it has a new best fitness
        new_best_solution_msg = "a solution with better fitness was found, saving the model. old fitness: " + str(
            best_fitness) + "; new fitness: " + str(fitness) + "; simulation number: " + str(total_simulation_runs)
        save_model_msg = "model with better fitness was saved successfully"
        print(new_best_solution_msg)
        logging.info(new_best_solution_msg)
        if sys.version_info.minor > 10:
            str_model_extension = ".keras"
        else:
            str_model_extension = ""
        tmp_model.save(os.path.join(network_folder, str(round(fitness, 10)) + '_fitness' + str_model_extension))
        print(save_model_msg)
        logging.info(save_model_msg)
        best_fitness = fitness

    total_simulation_runs += 1
    return fitness

def callback_generation(ga_instance):    # TODO problem with call on this function, it seems to be called in parallel with the new simulations, check pygad documentation
    global total_simulation_runs, simulation_cnt_on_last_run, model, start_sim_time  # previously argument is ga_instance, but it seems that the function did not work properly
    avg_time = round((time.time() - start_sim_time) / (total_simulation_runs - simulation_cnt_on_last_run + 1), 3)
    generation_txt = "---- Generation = {generation}; total simulations count = {sim_cnt}; average time per simulation: {avg_time}, seconds".\
        format(generation=ga_instance.generations_completed, sim_cnt=total_simulation_runs, avg_time=avg_time)
    # ' ; average time per simulation: ', round((time.time() - start_sim_time) / total_simulation_runs, 3), ' seconds'
    print(generation_txt)
    logging.info(generation_txt)
    # print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    # solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # tmp_model = tf.keras.models.clone_model(model)
    # model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=tmp_model, weights_vector=solution)
    # tmp_model.set_weights(weights=model_weights_matrix)
    # tmp_model.save(os.path.join('pygad_networks', str(round(solution_fitness, 8)) + '_fitness'))
    # save_model_txt = 'model was saved on a new generation'
    # print(save_model_txt)
    # logging.info(generation_txt + '; ' + save_model_txt)


logging.basicConfig(filename="nn_training_log.txt",
                    filemode='a',
                    format='%(asctime)s, %(levelname)s %(message)s',  # '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%model-%d,%H:%M:%S',
                    level=logging.INFO)
logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
                          })



# plant = sim_plant.SimulinkPlant(common_debug_mode=True)
total_simulation_runs = 3059
simulation_cnt_on_last_run = total_simulation_runs
# best_fitness = -0.0002598702  # DO NOT USE ga_instance.best_solution() IT BREAKS THE EXECUTION
start_sim_time = time.time()
sim_plant.SimulinkPlant.create_scaler()

py_version = sys.version_info.minor  # I had two network implementations, one for py 3.10 and another for py 3.12
if py_version > 10:
    best_fitness = -0.0006459109  # fitness given by the initial STABLE 1 network
    network_folder = "pygad_networks_py_12"
    model_name = "-0.0006459109_fitness.keras"  # model trained with batch normalization
    #model = keras.models.load_model(os.path.join(network_folder, '-0.0002598702_fitness'))
    model = keras.models.load_model(os.path.join(network_folder, model_name))
else:
    best_fitness = -0.0002598702
    network_folder = "pygad_networks"
    model_name = '-0.0002598702_fitness'
    model = keras.models.load_model(os.path.join(network_folder, model_name))  # no_batchnorm_controller_tf_2_10 test_controller_tf_2_10

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=50)  # 25

    # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 10000  # Number of generations.
num_parents_mating = 5  # 5  # Number of solutions to be selected as parents in the mating pool.
initial_population = keras_ga.population_weights  # Initial population of network weights
parent_selection_type = "sss"  # Type of parent selection. previously "sss"
crossover_type = "single_point" # before: "single_point"  # Type of the crossover operator. Previously "uniform"
mutation_type = "random"  # "random" "adaptive" # Type of the mutation operator.
# TODO Probably increase in the mutation percent genes is needed to get a better chance of finding a global optimum
mutation_percent_genes = 1  # (20, 3) # when adaptive mutation is used first element is mutation rate of a low-quality solution and secod is for high-quality
# 0.01 is too small
    #  5  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists. First stable NN trained with (10, 1) parameters
keep_parents = -1  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing. previously tested with 2 parents

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parallel_processing=5,  # 4 works well with 2024b matlab; 6 is max (still working well)
                       # parallel_processing=["process", 4],  # 4,
                       #  delay_after_gen= 15.0,  # delay after gen was not set originally
                       on_generation=callback_generation,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents)


if __name__ == '__main__':
    # Only spawn processes if this is the main module
    # multiprocessing.freeze_support()  # Optional for Windows; not needed if not freezing the code

    ga_instance.run()
    use_mujoco = True

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()  # NOTE; best_solution() does not work correctly during run(). DO NOT USE IT THERE


    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)
    model.save(os.path.join('pygad_networks', str(round(solution_fitness, 10)) + '_fitness'))

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    # ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)
    ga_instance.plot_fitness()

    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    print('simulations run: ', total_simulation_runs, ' ; average time per simulation: ', round((time.time()-start_sim_time) / total_simulation_runs, 3), ' seconds')


