import numpy as np
import networkx as nx
from networkx import NetworkXError, NetworkXUnfeasible
import matplotlib.pyplot as plt
# import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from pathlib import Path


# def has_cycle(adj_or_ancst_matrix):
#     n = adj_or_ancst_matrix.shape[0]
#     k_reachability_matrix = adj_or_ancst_matrix.copy()
#     for i in range(n):
#         k_reachability_matrix = np.matmul(k_reachability_matrix, adj_or_ancst_matrix)
#         if np.trace(k_reachability_matrix) != 0:
#             return True
#     return False


def create_adjacency_from_ancestry(ancst_matrix):
    # # TODO: https://mathematica.stackexchange.com/questions/33638/remove-redundant-dependencies-from-a-directed-acyclic-graph?newreg=ee77f66fe9f74d7ba8c92c05630b305d
    # n = ancst_matrix.shape[0]
    # adj_matrix = np.zeros((n, n))
    #
    # if has_cycle(ancst_matrix):
    #     return False
    #
    # resolved_nodes = np.zeros((n, ))
    #
    # last_iteration_leaves = np.zeros((n, ))
    #
    # while sum(resolved_nodes == 0) > 0:
    #     this_round_leaves = np.zeros((n, ))
    #     for i in range(n):
    #         if not resolved_nodes[i]:
    #             if sum(ancst_matrix[i, resolved_nodes == 0]) == 0:
    #                 resolved_nodes[i] = 1
    #
    #                 this_round_leaves[i] = 1
    #
    #                 for j in range(n):
    #                     if last_iteration_leaves[j]:
    #                         if ancst_matrix[i, j]:
    #                             adj_matrix[i, j] = 1
    #
    #     last_iteration_leaves = this_round_leaves
    # return adj_matrix
    g = nx.from_numpy_array(ancst_matrix, create_using=nx.DiGraph)
    try:
        ts = list(nx.topological_sort(g))
    except (NetworkXError, NetworkXUnfeasible):
        return False

    n = ancst_matrix.shape[0]
    adj_matrix = np.zeros((n, n))

    for i, child in enumerate(ts[-1::-1]):
        for parent in ts[n - i - 2::-1]:
            if abs(ancst_matrix[parent, child] - 1) < 0.000001:
                adj_matrix[parent, child] = 1
                break

    new_ancst_matrix = create_ancestry_matrix_from_adjacency(adj_matrix)

    for i in range(n):
        for j in range(n):
            if abs(new_ancst_matrix[i, j] - ancst_matrix[i, j]) > 0.0000001:
                return False

    return adj_matrix


# def create_adjacency_matrix_from_edge_list(directed_tree_edges, number_of_nodes):
#     edges = list(directed_tree_edges)
#     adj = np.zeros((number_of_nodes, number_of_nodes))
#     for edge in edges:
#         adj[edge[0], edge[1]] = 1
#     return adj


def find_possible_root_nodes(ancst_matrix_or_directed_adj_matrix):
    return np.where(np.sum(ancst_matrix_or_directed_adj_matrix, axis=0) == 0)[0]


def number_of_root_nodes(ancst_matrix):
    return sum(np.sum(ancst_matrix, axis=0) == 0)


def is_connected(ancst_matrix):
    #  assuming valid ancst matrix
    nr = number_of_root_nodes(ancst_matrix)
    if nr == 1:
        return True
    else:
        return False


#     n = ancst_matrix.shape[0]
#     undirected_ancst_matrix = ancst_matrix + np.transpose(ancst_matrix)
#     an = undirected_ancst_matrix
#     sumAn =
#     for i in range(n):
#         an =  np.matmul(an, undirected_ancst_matrix)
#         sumAn = sumAn + an
#
#     if sum(sumAn == 0) > 0:
#         return False
#     else:
#         return True

def create_random_adjacency_matrix(n, seed=0):
    tree = nx.random_tree(n=n, seed=seed, create_using=nx.DiGraph)
    return nx.to_numpy_array(tree)


def create_ancestry_matrix_from_adjacency(adj_matrix: np.array) -> np.array:
    g = nx.DiGraph(adj_matrix)
    length = dict(nx.all_pairs_shortest_path_length(g))
    ancst_matrix = np.array([[length.get(m, {}).get(n, 0) > 0 for m in g.nodes] for n in g.nodes], dtype=np.int32)
    return np.transpose(ancst_matrix)


def create_random_ancestry_matrix(n, seed) -> np.array:
    adj_matrix = create_random_adjacency_matrix(n, seed)
    return create_ancestry_matrix_from_adjacency(adj_matrix)


def draw_tree(tree_adjacency_matrix, file_address=None):
    g = nx.from_numpy_array(tree_adjacency_matrix, create_using=nx.DiGraph)
    if not nx.is_tree(g):
        return False
    pos = graphviz_layout(g, prog="dot")
    nx.draw(g, pos, with_labels=True)
    if file_address is not None:
        plt.savefig(file_address)
    plt.show()


def save_plot(tree_adjacency_matrix, file_address, labels=None, node_size=300, font_size=12, dpi=100, graphviz=False):
    g = nx.from_numpy_array(tree_adjacency_matrix, create_using=nx.DiGraph)
    # if not nx.is_tree(g):
    #     return False
    if graphviz == False:
        pos = graphviz_layout(g, prog="dot")
        if labels is None:
            nx.draw_networkx(g, pos, with_labels=True, font_size=font_size, node_size=node_size)
        else:
            nx.draw_networkx(g, pos, with_labels=True, labels=labels, font_size=font_size, node_size=node_size)
        plt.savefig(file_address, dpi=dpi)
        plt.clf()
    else:
        if labels is not None:
            nx.relabel_nodes(g, labels, False)
        A = nx.nx_agraph.to_agraph(g)
        A.draw(file_address, prog="dot")


def draw_tree_unidrected(tree_adjacency_matrix):
    g = nx.from_numpy_array(tree_adjacency_matrix)
    if not nx.is_tree(g):
        return False
    pos = graphviz_layout(g, prog="dot")
    nx.draw(g, pos, with_labels=True)
    plt.show()


def add_simply_inferred_ancestry_relations(partial_ancestry_matrix):
    return create_ancestry_matrix_from_adjacency(partial_ancestry_matrix)


def adding_1_greedy_algorithm_top_down(partial_ancestry_matrix):
    # Greediness: TopDown: nodes woth most 1s should be parents
    n = partial_ancestry_matrix.shape[0]
    partial_ancestry_matrix = add_simply_inferred_ancestry_relations(partial_ancestry_matrix)
    number_of_ones = np.sum(partial_ancestry_matrix, axis=1)
    sorted_indices = np.argsort(number_of_ones)
    sorted_indices = np.flip(sorted_indices)

    # The node with the most number of ones should be the root!
    for i in range(n):
        if i != sorted_indices[0]:
            partial_ancestry_matrix[sorted_indices[0], i] = 1

    for ind1 in range(1, n):
        i = sorted_indices[ind1]
        for k in range(n - ind1 - 1):
            for ind2 in range(ind1 + 1, n):
                j = sorted_indices[ind2]
                temp = partial_ancestry_matrix[i, :] + partial_ancestry_matrix[j, :]
                if sum(temp == 2) > 0:
                    temp = (temp >= 1)
                    temp.astype(int)
                    temp[j] = 1
                    partial_ancestry_matrix[i, :] = temp
                    partial_ancestry_matrix = add_simply_inferred_ancestry_relations(partial_ancestry_matrix)
    return partial_ancestry_matrix
    # possible_root_nodes = find_possible_root_nodes(partial_ancestry_matrix)
    # lest_changes = n ** 2
    # for proot_index in possible_root_nodes:
    #     temp_partial_ancestry_matrix = partial_ancestry_matrix.copy()
    #     for i in range(n):
    #         if i != proot_index:
    #             temp_partial_ancestry_matrix[proot_index, i] = 1
    #         temp_partial_ancestry_matrix


def get_available_parent(parents_list, j, counter=0):
    # if j has no parent returns j else return its an ancestor of j with no parent,
    # if there is no such ancestor for j raises value error
    if counter > len(parents_list):
        raise ValueError("circular relation in parents list")
    if parents_list[j] == -1:
        return j
    else:
        return get_available_parent(parents_list, parents_list[j], counter=counter + 1)


def adding_1_greedy_algorithm_bottom_up(partial_ancestry_matrix):
    # Greediness: BottomUp: nodes woth least 1s should be childs
    n = partial_ancestry_matrix.shape[0]
    partial_ancestry_matrix = add_simply_inferred_ancestry_relations(partial_ancestry_matrix)
    number_of_ones = np.sum(partial_ancestry_matrix, axis=1)
    sorted_indices = np.argsort(number_of_ones)

    if number_of_ones[sorted_indices[0]] != 0:
        raise ValueError("This partial ancestry matrix contains no leaves!")

    if number_of_ones[sorted_indices[-1]] == 0:
        # all zero matrix
        partial_ancestry_matrix[0, 1:] = np.ones((1, n - 1))
        return partial_ancestry_matrix

    # Nodes with no ones are leaves
    parents_list = np.zeros((n,), dtype=int) - 1
    done_list = []
    while len(done_list) < n:
        for i in range(n):
            ind = sorted_indices[i]
            if ind not in done_list:
                done_list.append(ind)
                break
        for j in range(n):
            if partial_ancestry_matrix[ind, j] == 1:
                available_parent = get_available_parent(parents_list, j)
                if available_parent != ind:
                    parents_list[available_parent] = ind
                    partial_ancestry_matrix[ind, available_parent] = 1
                    partial_ancestry_matrix = add_simply_inferred_ancestry_relations(partial_ancestry_matrix)
                    number_of_ones = np.sum(partial_ancestry_matrix, axis=1)
                    sorted_indices = np.argsort(number_of_ones)

    # Now the row with most ones should be the root
    for i in range(n):
        if i != sorted_indices[-1]:
            partial_ancestry_matrix[sorted_indices[-1], i] = 1

    return partial_ancestry_matrix


def find_maximum_arborescence_from_adj(arbitrary_matrix):
    g = nx.from_numpy_array(arbitrary_matrix, create_using=nx.DiGraph)
    arborescence_tree = nx.maximum_spanning_arborescence(g)
    return nx.to_numpy_array(arborescence_tree)


def ADD(adj1, adj2):
    if adj1.shape != adj2.shape:
        m = min(adj1.shape[0], adj2.shape[0])
        adj1 = adj1[:m, :m]
        adj2 = adj2[:m, :m]
    anc1 = create_ancestry_matrix_from_adjacency(adj1)
    anc2 = create_ancestry_matrix_from_adjacency(adj2)
    return np.sum(np.abs(anc1 - anc2))


def PCD(adj1, adj2):
    if adj1.shape != adj2.shape:
        m = min(adj1.shape[0], adj2.shape[0])
        adj1 = adj1[:m, :m]
        adj2 = adj2[:m, :m]
    return np.sum(np.abs(adj1 - adj2))


def ddistl1(adj1, adj2):
    if adj1.shape != adj2.shape:
        m = min(adj1.shape[0], adj2.shape[0])
        adj1 = adj1[:m, :m]
        adj2 = adj2[:m, :m]
    dist1 = create_distance_matrix_from_adjacency(adj1)
    dist2 = create_distance_matrix_from_adjacency(adj2)
    return np.sum(np.abs(dist1 - dist2))


def ddistl2(adj1, adj2):
    if adj1.shape != adj2.shape:
        m = min(adj1.shape[0], adj2.shape[0])
        adj1 = adj1[:m, :m]
        adj2 = adj2[:m, :m]
    dist1 = create_distance_matrix_from_adjacency(adj1)
    dist2 = create_distance_matrix_from_adjacency(adj2)
    return np.sum(np.abs(dist1 - dist2) ** 2)


def ddistlinf(adj1, adj2):
    if adj1.shape != adj2.shape:
        m = min(adj1.shape[0], adj2.shape[0])
        adj1 = adj1[:m, :m]
        adj2 = adj2[:m, :m]
    dist1 = create_distance_matrix_from_adjacency(adj1)
    dist2 = create_distance_matrix_from_adjacency(adj2)
    return np.max(np.abs(dist1 - dist2))


def create_adjacency_from_dist(dist_matrix):
    adj = np.abs(dist_matrix - 1) < .01
    return adj.astype(int)


def create_distance_matrix_from_adjacency_old(adj_matrix):
    n = adj_matrix.shape[0]
    adj_matrix = np.transpose(adj_matrix) + adj_matrix
    distance_matrix = np.identity(n)
    distance_matrix -= 1
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] == -1:
                for k in range(n):
                    if adj_matrix[j, k] == 1 and distance_matrix[i, k] > -1:
                        distance_matrix[i, j] = distance_matrix[i, k] + 1
                        distance_matrix[j, i] = distance_matrix[i, j]
                        continue
                    if adj_matrix[i, k] == 1 and distance_matrix[j, k] > -1:
                        distance_matrix[i, j] = distance_matrix[j, k] + 1
                        distance_matrix[j, i] = distance_matrix[i, j]
                        continue
    return distance_matrix


def create_distance_matrix_from_adjacency(adj_matrix):
    n = adj_matrix.shape[0]
    adj_matrix = np.transpose(adj_matrix) + adj_matrix
    g = nx.from_numpy_array(adj_matrix)
    p = dict(nx.shortest_path_length(g))
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if j in p[i]:
                distance_matrix[i, j] = p[i][j]
                distance_matrix[j, i] = p[i][j]
            else:
                distance_matrix[i, j] = n // 2
                distance_matrix[j, i] = n // 2
    return distance_matrix


def turn_into_directed_tree(undirected_adj_matrix, root_index):
    n = undirected_adj_matrix.shape[0]
    directed_adj_matrix = np.zeros((n, n))
    queue = [root_index]
    while len(queue) > 0:
        i = queue[0]
        queue = queue[1:]
        for j in range(n):
            if undirected_adj_matrix[i, j] == 1 and directed_adj_matrix[j, i] == 0:
                directed_adj_matrix[i, j] = 1
                queue.append(j)
    return directed_adj_matrix


def find_root_node(dir_adj_matrix):
    roots = find_possible_root_nodes(dir_adj_matrix)
    if len(roots) != 1:
        raise ValueError("Not a rooted tree!")
    else:
        return roots[0]


def find_parent_node(dir_adj_matrix, i):
    possible_parents = np.where(dir_adj_matrix[:, i] == 1)[0]
    if len(possible_parents) > 1:
        raise ValueError("multiple possible parents!")
    if len(possible_parents) == 1:
        return possible_parents[0]
    else:
        return None


def find_children(dir_adj_matrix, i):
    children = np.where(dir_adj_matrix[i, :] == 1)[0]
    return children


def make_directory(folder_address):
    path = Path(folder_address)
    path.mkdir(parents=True, exist_ok=True)


def L1(mat1, mat2):
    return np.sum(np.abs(mat1 - mat2))


def L2(mat1, mat2):
    return np.sum((mat1 - mat2) ** 2)


def LInf(mat1, mat2):
    return np.max(np.abs(mat1 - mat2))


def compare_matrices(m1, m2, epsilon=0.00001):
    n1 = m1.shape[0]
    n2 = m2.shape[0]
    if n1 != n2:
        return False
    for i in range(n1):
        for j in range(n2):
            if abs(m1[i, j] - m2[i, j]) > epsilon:
                return False
    return True


def compute_cons(tree_maps, target_tree_map):
    s = 0
    for i in range(len(tree_maps)):
        t_anc = tree_maps[i]
        s += L1(t_anc, target_tree_map)
    return s


def solveCoTPBF(tree_map, tree_set):
    found_min = 100000
    selected_trees = []
    for t_map in tree_set:
        s = compute_cons(tree_map, t_map)
        if s == found_min:
            selected_trees.append(t_map)
        elif s < found_min:
            selected_trees = [t_map]
            found_min = s
    return found_min, selected_trees
