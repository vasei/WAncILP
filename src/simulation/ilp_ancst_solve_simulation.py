import time

from src.ilp.ancst import generate_gl_rooted_model_and_solve_it, generate_model_and_solve_it, \
    generate_model_and_solve_it_l1
from src.simulation.solve_utils import compute_parameters_from_folder_address, read_all_perturbed_trees_in_trial, \
    get_simulated_data_folder_addresses, check_file_exist
import numpy as np

from src.utils import create_ancestry_matrix_from_adjacency, create_adjacency_from_ancestry, save_plot

ilp_file_name = "ilp_ancst_tree"
# ilp_solver = generate_model_and_solve_it_l1
ilp_solver = generate_model_and_solve_it


def solve(folder_address, trial_number, ancst_ilp_solver):
    n, k, cp, pc, bm, nr, seed = compute_parameters_from_folder_address(folder_address)
    print('Loading input trees...')
    tree_list_adjs = read_all_perturbed_trees_in_trial(folder_address, trial_number)
    ancst_matrices = [create_ancestry_matrix_from_adjacency(x) for x in tree_list_adjs]
    a = np.mean(ancst_matrices, axis=0)
    print("solving ilp", time.asctime())
    ancst_matrix = ancst_ilp_solver(a)
    print("solving ilp finished", time.asctime())
    ilp_adj_matrix = create_adjacency_from_ancestry(ancst_matrix)
    print('Saving result tree...')
    np.savetxt(folder_address + "/trial" + str(trial_number) + "/%s.csv" % ilp_file_name, ilp_adj_matrix, delimiter=",")
    save_plot(ilp_adj_matrix, folder_address + "/trial" + str(trial_number) + "/%s.jpg" % ilp_file_name)
    print('Program finished!')


folder_addresses = get_simulated_data_folder_addresses()
for folder_address in folder_addresses:
    n, k, cp, pc, bm, nr, seed = compute_parameters_from_folder_address(folder_address)
    if seed >= 46:
        for i in range(100):
            start_time = time.time()
            print("#############################")
            print(folder_address, "trial%d" % i)
            if check_file_exist(folder_address + "/trial%d" % i + "/%s.jpg" % ilp_file_name):
                print("already solved")
                continue
            solve(folder_address, i, ilp_solver)
            print("time: ", time.time() - start_time)
