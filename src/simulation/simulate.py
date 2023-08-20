import networkx as nx
import numpy as np

from src.utils import *


def perturb_tree_substitute_parent_child(adjacency_matrix, ccf, success_prob, seed):
    np.random.seed(seed)
    perturbed_adjacency = np.copy(adjacency_matrix)
    perturbed_ccf = np.copy(ccf)
    while np.random.random(1)[0] < success_prob:
        selected_node = np.random.randint(adjacency_matrix.shape[0])
        parent = find_parent_node(perturbed_adjacency, selected_node)
        root = find_root_node(perturbed_adjacency)
        if parent is not None and parent != root:
            r = np.random.random(1)[0]
            if r * ccf[parent] > ccf[parent] - ccf[selected_node]:
                temp_row = np.copy(perturbed_adjacency[parent, :])
                temp_column = np.copy(perturbed_adjacency[:, parent])
                perturbed_adjacency[parent, :] = perturbed_adjacency[selected_node, :]
                perturbed_adjacency[:, parent] = perturbed_adjacency[:, selected_node]
                perturbed_adjacency[selected_node, :] = temp_row
                perturbed_adjacency[:, selected_node] = temp_column
                perturbed_adjacency[parent, selected_node] = 0
                perturbed_adjacency[selected_node, parent] = 1
                temp_ccf = perturbed_ccf[parent]
                perturbed_ccf[parent] = perturbed_ccf[selected_node]
                perturbed_ccf[selected_node] = temp_ccf
    return perturbed_adjacency, perturbed_ccf


def perturb_tree_delete_node(adjacency_matrix, ccf, success_prob, seed):
    np.random.seed(seed)
    perturbed_adjacency = np.copy(adjacency_matrix)
    perturbed_ccf = np.copy(ccf)
    while np.random.random(1)[0] < success_prob:
        selected_node = np.random.randint(adjacency_matrix.shape[0])
        parent = find_parent_node(perturbed_adjacency, selected_node)
        root = find_root_node(perturbed_adjacency)
        if parent is not None and parent != root:
            r = np.random.random(1)[0]
            if r > ccf[selected_node]:
                perturbed_adjacency[parent, selected_node] = 0
                perturbed_adjacency[root, selected_node] = 1
                perturbed_ccf[selected_node] = 0.0000001
                for i in range(adjacency_matrix.shape[0]):
                    if perturbed_adjacency[selected_node, i] == 1:
                        perturbed_adjacency[parent, i] = 1
                        perturbed_adjacency[selected_node, i] = 0
    return perturbed_adjacency, perturbed_ccf


def compute_empty_ccf(adj_mat, ccf):
    empty_ccf = np.zeros((adj_mat.shape[0],))
    for i in range(len(empty_ccf)):
        children = find_children(adj_mat, i)
        empty_ccf[i] = ccf[i] - sum(ccf[children])
    return empty_ccf


def perturb_tree_move_branch(adjacency_matrix, ccf, success_prob, seed):
    np.random.seed(seed)
    perturbed_adjacency = np.copy(adjacency_matrix)
    perturbed_ccf = np.copy(ccf)
    while np.random.random(1)[0] < success_prob:
        empty_ccf = compute_empty_ccf(perturbed_adjacency, ccf)
        selected_node = np.random.randint(adjacency_matrix.shape[0])
        possible_parents = np.where(empty_ccf > ccf[selected_node])[0]
        if len(possible_parents) > 0:
            selected_parent = possible_parents[np.random.randint(len(possible_parents))]
            parent = find_parent_node(perturbed_adjacency, selected_node)
            perturbed_adjacency[parent, selected_node] = 0
            perturbed_adjacency[selected_parent, selected_node] = 1
    return perturbed_adjacency, perturbed_ccf


def generate_ccf(adjacency_matrix, seed=0):
    ccf = np.zeros((adjacency_matrix.shape[0],))
    roots = find_possible_root_nodes(adjacency_matrix)
    if len(roots) != 1:
        raise ValueError("Input matrix has no roots or more than one root node!")
    ccf[roots[0]] = 1
    while sum(ccf == 0) > 0:
        zeros = np.where(ccf == 0)[0]
        for zero in zeros:
            parent = find_parent_node(adjacency_matrix, zero)
            if ccf[parent] > 0 and ccf[zero] == 0:
                children = find_children(adjacency_matrix, parent)
                if len(children) > 0:
                    new_ccfs = generate_random_numbers_to_sum(len(children) + 1, ccf[parent], seed)
                    i = 0
                    for child in children:
                        ccf[child] = new_ccfs[i]
                        i += 1
    return ccf


def generate_random_numbers_to_sum(n, s, seed=0):
    np.random.seed(seed)
    randoms = np.random.random(n)
    randoms = randoms / sum(randoms)
    randoms = randoms * s
    return randoms
