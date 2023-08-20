from glob import glob

import os.path
import numpy as np
import os

os.environ["PATH"] += os.pathsep + 'C://Program Files//Graphviz//bin'

def compute_parameters_from_folder_address(folder_address: str):
    folder_address = folder_address.replace("\\", "/")
    folder_name = folder_address.split("/")[-1]
    parts = folder_name.split("_")
    n = int(parts[0][1:])
    k = int(parts[1][1:])
    cp = int(parts[2][2:]) * .1
    pc = int(parts[3][2:])
    bm = int(parts[4][2:])
    nr = int(parts[5][2:])
    seed = int(parts[6][4:])
    return n, k, cp, pc, bm, nr, seed


def read_all_perturbed_trees_in_trial(folder_address, trial_number):
    adj_mat_list = []
    n, k, cp, pc, bm, nr, seed = compute_parameters_from_folder_address(folder_address)
    for i in range(k):
        file_address = folder_address + "/trial" + str(trial_number) + "/ptree_%d.csv" % i
        temp_adj_mat = read_array_from_csv_file(file_address)
        adj_mat_list.append(temp_adj_mat)
    return adj_mat_list


def read_array_from_csv_file(file_address):
    return np.genfromtxt(file_address, delimiter=",")


def convert_tree_adjacency_to_tree_dict(adj_mat):
    tree_dict = {}
    n = adj_mat.shape[0]
    for i in range(n):
        for j in range(n):
            if adj_mat[i, j] == 1:
                if (i,) in tree_dict:
                    tree_dict[(i,)].append((j,))
                else:
                    tree_dict[(i,)] = [(j,)]
    return tree_dict


def create_adjacency_from_cp_dict(cp_dict, n):
    adj_mat = np.zeros((n, n))
    for j_tup in cp_dict:
        for i in cp_dict[j_tup]:
            for j in j_tup:
                adj_mat[i, j] = 1
    return adj_mat


def get_simulated_data_folder_addresses():
    folders = glob("src/simulation/data/*")
    return folders


def check_file_exist(file_address):
    return os.path.isfile(file_address)
