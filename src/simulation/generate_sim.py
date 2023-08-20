import random
import numpy as np

from src.simulation.solve_utils import check_file_exist
from src.utils import make_directory, create_random_adjacency_matrix, draw_tree
from src.simulation.simulate import *

base_folder_address = "data"


def generate_simulations(tree_nodes,
                         number_of_perturbed_trees,
                         number_of_simulations,
                         change_probability=.9,
                         pc_change=True,
                         bm_change=True,
                         nr_change=True,
                         seed=0):
    random.seed(seed)
    parent_folder_address = base_folder_address + "/n%d_k%d_cp%d_pc%d_bm%d_nr%d_seed%d" % (
        tree_nodes, number_of_perturbed_trees, change_probability * 10,
        0 if pc_change is False else 1,
        0 if bm_change is False else 1,
        0 if nr_change is False else 1,
        seed)

    for i in range(number_of_simulations):
        folder_address = parent_folder_address + "/trial%d" % i
        make_directory(folder_address)
        if check_file_exist(folder_address + "/ptree_%d.csv" % (number_of_perturbed_trees - 1)):
            continue
        adj_mat = create_random_adjacency_matrix(tree_nodes, seed=random.randint(0, 10000000))
        ccf = generate_ccf(adj_mat, seed=random.randint(0, 10000000))
        np.savetxt(folder_address + "/true_tree.csv", adj_mat, delimiter=",")
        np.savetxt(folder_address + "/true_ccf.csv", ccf, delimiter=",")
        save_plot(adj_mat, folder_address + "/true_tree.jpg")
        for j in range(number_of_perturbed_trees):
            print(i, j)
            perturbed_mat = np.copy(adj_mat)
            per_ccf = np.copy(ccf)
            if pc_change:
                perturbed_mat, per_ccf = perturb_tree_substitute_parent_child(perturbed_mat, per_ccf,
                                                                              change_probability,
                                                                              seed=random.randint(0, 10000000))
            if bm_change:
                perturbed_mat, per_ccf = perturb_tree_move_branch(perturbed_mat, per_ccf, change_probability,
                                                                  seed=random.randint(0, 10000000))
            if nr_change:
                perturbed_mat, per_ccf = perturb_tree_delete_node(perturbed_mat, per_ccf, change_probability,
                                                                  seed=random.randint(0, 10000000))

            np.savetxt(folder_address + "/ptree_%d.csv" % j, perturbed_mat, delimiter=",")
            np.savetxt(folder_address + "/pccf_%d.csv" % j, per_ccf, delimiter=",")
            save_plot(perturbed_mat, folder_address + "/ptree_%d.jpg" % j)


# generate_simulations(10, 5, 100, change_probability=.9, pc_change=True, bm_change=False, nr_change=False, seed=0)
# generate_simulations(10, 5, 100, change_probability=.9, pc_change=True, bm_change=False, nr_change=True, seed=1)
# generate_simulations(10, 5, 100, change_probability=.9, pc_change=False, bm_change=True, nr_change=True, seed=2)
# generate_simulations(10, 5, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=False, seed=3)
# generate_simulations(10, 5, 100, change_probability=.9, pc_change=False, bm_change=False, nr_change=True, seed=4)
# generate_simulations(10, 5, 100, change_probability=.9, pc_change=False, bm_change=True, nr_change=False, seed=5)
# generate_simulations(10, 5, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=True, seed=6)
#
# generate_simulations(10, 10, 100, change_probability=.9, pc_change=True, bm_change=False, nr_change=False, seed=7)
# generate_simulations(10, 10, 100, change_probability=.9, pc_change=True, bm_change=False, nr_change=True, seed=8)
# generate_simulations(10, 10, 100, change_probability=.9,pc_change=False, bm_change=True, nr_change=True, seed=9)
# generate_simulations(10, 10, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=False, seed=10)
# generate_simulations(10, 10, 100, change_probability=.9, pc_change=False, bm_change=False, nr_change=True, seed=11)
# generate_simulations(10, 10, 100, change_probability=.9, pc_change=False, bm_change=True, nr_change=False, seed=12)
# generate_simulations(10, 10, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=True, seed=13)
#
# generate_simulations(20, 10, 100, change_probability=.9, pc_change=True, bm_change=False, nr_change=False, seed=14)
# generate_simulations(20, 10, 100, change_probability=.9, pc_change=True, bm_change=False, nr_change=True, seed=15)
# generate_simulations(20, 10, 100, change_probability=.9, pc_change=False, bm_change=True, nr_change=True, seed=16)
# generate_simulations(20, 10, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=False, seed=17)
# generate_simulations(20, 10, 100, change_probability=.9, pc_change=False, bm_change=False, nr_change=True, seed=18)
# generate_simulations(20, 10, 100, change_probability=.9, pc_change=False, bm_change=True, nr_change=False, seed=19)
# generate_simulations(20, 10, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=True, seed=20)

# generate_simulations(20, 20, 100, change_probability=.9, pc_change=True, bm_change=False, nr_change=False, seed=21)
# generate_simulations(20, 20, 100, change_probability=.9, pc_change=True, bm_change=False, nr_change=True, seed=22)
# generate_simulations(20, 20, 100, change_probability=.9, pc_change=False, bm_change=True, nr_change=True, seed=23)
# generate_simulations(20, 20, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=False, seed=24)
# generate_simulations(20, 20, 100, change_probability=.9, pc_change=False, bm_change=False, nr_change=True, seed=25)
# generate_simulations(20, 20, 100, change_probability=.9, pc_change=False, bm_change=True, nr_change=False, seed=26)
# generate_simulations(20, 20, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=True, seed=27)
#
# generate_simulations(20, 40, 100, change_probability=.9, pc_change=True, bm_change=False, nr_change=False, seed=28)
# generate_simulations(20, 40, 100, change_probability=.9, pc_change=True, bm_change=False, nr_change=True, seed=29)
# generate_simulations(20, 40, 100, change_probability=.9, pc_change=False, bm_change=True, nr_change=True, seed=30)
# generate_simulations(20, 40, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=False, seed=31)
# generate_simulations(20, 40, 100, change_probability=.9, pc_change=False, bm_change=False, nr_change=True, seed=32)
# generate_simulations(20, 40, 100, change_probability=.9, pc_change=False, bm_change=True, nr_change=False, seed=33)
# generate_simulations(20, 40, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=True, seed=34)

# generate_simulations(10, 20, 100, change_probability=.9, pc_change=True, bm_change=False, nr_change=False, seed=35)
# generate_simulations(10, 20, 100, change_probability=.9, pc_change=True, bm_change=False, nr_change=True, seed=36)
# generate_simulations(10, 20, 100, change_probability=.9, pc_change=False, bm_change=True, nr_change=True, seed=37)
# generate_simulations(10, 20, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=False, seed=38)
# generate_simulations(10, 20, 100, change_probability=.9, pc_change=False, bm_change=False, nr_change=True, seed=39)
# generate_simulations(10, 20, 100, change_probability=.9, pc_change=False, bm_change=True, nr_change=False, seed=40)
# generate_simulations(10, 20, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=True, seed=41)


# generate_simulations(7, 3, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=True, seed=42)
# generate_simulations(7, 10, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=True, seed=43)
# generate_simulations(7, 15, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=True, seed=44)

# generate_simulations(4, 3, 100, change_probability=.9, pc_change=True, bm_change=True, nr_change=True, seed=45)

# generate_simulations(10, 5, 100, change_probability=.5, pc_change=True, bm_change=True, nr_change=False, seed=46)
# generate_simulations(10, 5, 100, change_probability=.5, pc_change=True, bm_change=True, nr_change=True, seed=47)
# generate_simulations(10, 10, 100, change_probability=.5, pc_change=True, bm_change=True, nr_change=False, seed=48)
# generate_simulations(10, 10, 100, change_probability=.5, pc_change=True, bm_change=True, nr_change=True, seed=49)
# generate_simulations(10, 20, 100, change_probability=.5, pc_change=True, bm_change=True, nr_change=False, seed=50)
# generate_simulations(10, 20, 100, change_probability=.5, pc_change=True, bm_change=True, nr_change=True, seed=51)
