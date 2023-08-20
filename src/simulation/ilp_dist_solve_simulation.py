import sys
import time

from src.ilp.dist import generate_model_and_solve_it, generate_model_and_solve_it_l2, generate_model_and_solve_it_inf
from src.simulation.solve_utils import compute_parameters_from_folder_address, read_all_perturbed_trees_in_trial, \
    get_simulated_data_folder_addresses, check_file_exist, read_array_from_csv_file
import numpy as np

from src.utils import create_ancestry_matrix_from_adjacency, create_adjacency_from_ancestry, save_plot, \
    create_distance_matrix_from_adjacency, create_adjacency_from_dist, turn_into_directed_tree

input_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
fromIndex = int(sys.argv[2]) if len(sys.argv) > 2 else 0
toIndex = int(sys.argv[3]) if len(sys.argv) > 3 else 100
timelimit = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

ilp_solver_l1 = generate_model_and_solve_it
# ilp_solver_l2 = generate_model_and_solve_it_l2
# ilp_solver_linf = generate_model_and_solve_it_inf
ilp_file_name = "ilp_dist_tree"


def solve(folder_address, trial_number, dist_ilp_solver, timelimit=None, partial_solution=None):
    n, k, cp, pc, bm, nr, seed = compute_parameters_from_folder_address(folder_address)
    print('Loading input trees...')
    tree_list_adjs = read_all_perturbed_trees_in_trial(folder_address, trial_number)
    dist_matrices = [create_distance_matrix_from_adjacency(x) for x in tree_list_adjs]

    if partial_solution is None:
        tree_0 = read_array_from_csv_file(folder_address + "/trial" + str(trial_number) + "/ptree_0.csv")
        starting_point = create_distance_matrix_from_adjacency(tree_0)
    else:
        starting_point = partial_solution
    a = np.mean(ancst_matrices, axis=0)
    print("solving ilp", time.asctime())
    D, E, Z, status = dist_ilp_solver(a, starting_point_distance_matrix=starting_point,
                                      timelimit=timelimit)
    print("solving ilp finished", time.asctime())
    if status == 'integer optimal solution' or status == 'integer optimal, tolerance':
        ilp_adj_matrix = create_adjacency_from_dist(D)
        ilp_adj_matrix = turn_into_directed_tree(ilp_adj_matrix, 0)
        print('Saving result tree...')
        np.savetxt(folder_address + "/trial" + str(trial_number) + "/%s.csv" % ilp_file_name, ilp_adj_matrix,
                   delimiter=",")
        save_plot(ilp_adj_matrix, folder_address + "/trial" + str(trial_number) + "/%s.jpg" % ilp_file_name)
    else:
        np.savetxt(folder_address + "/trial" + str(trial_number) + "/partial_solution_D.csv", D, delimiter=",")

    print('Program finished!')


folder_addresses = get_simulated_data_folder_addresses()
for folder_address in folder_addresses:
    n, k, cp, pc, bm, nr, seed = compute_parameters_from_folder_address(folder_address)
    if seed >= 46:
        for i in range(fromIndex, toIndex):
            start_time = time.time()
            print("#############################")
            print(folder_address, "trial%d" % i)
            if check_file_exist(folder_address + "/trial%d" % i + "/%s.jpg" % ilp_file_name):
                print("already solved")
                continue
            partial_solution_address = folder_address + "/trial%d" % i + "/partial_solution_D.csv"
            if check_file_exist(partial_solution_address):
                partial_solution = read_array_from_csv_file(partial_solution_address)
            else:
                partial_solution = None
            solve(folder_address, i, ilp_solver_l1, timelimit=timelimit, partial_solution=partial_solution)
            print("time: ", time.time() - start_time)
