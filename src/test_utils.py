import unittest
import numpy as np
from utils import *
import networkx as nx


#
# class TestHasCycle(unittest.TestCase):
#
#     def setUp(self) -> None:
#         np.random.seed(0)
#
#     def test_zeros(self):
#         ancst_matrix = np.zeros((5, 5))
#         self.assertEqual(has_cycle(ancst_matrix), False)
#
#     def test_ones(self):
#         ancst_matrix = np.ones((5, 5))
#         self.assertEqual(has_cycle(ancst_matrix), True)
#
#     def test_random_tree(self):
#
#
#     def test_path(self):


class TestCreateAncestryFromAdjacency(unittest.TestCase):

    def test_transitivity(self):
        n = 10
        seed = 1
        for i in range(10):
            a = create_random_ancestry_matrix(n, seed)
            b = create_random_adjacency_matrix(n, seed)
            c = create_ancestry_matrix_from_adjacency(b)
            self.assertEqual(sum(sum(a == c)), n ** 2)
            seed += 1

    def test_path_tree(self):
        n = 10
        g = nx.path_graph(n, create_using=nx.DiGraph)
        adj_matrix = nx.to_numpy_array(g)
        ancst_matrix = np.triu(np.ones((n, n)))
        np.fill_diagonal(ancst_matrix, 0)
        a = create_ancestry_matrix_from_adjacency(adj_matrix)
        self.assertEqual(sum(sum(a == ancst_matrix)), n ** 2)

    def test_star_tree(self):
        n = 10
        adj_matrix = np.zeros((n, n))
        adj_matrix[0, 1:] = 1
        ancst_matrix = adj_matrix.copy()
        a = create_ancestry_matrix_from_adjacency(adj_matrix)
        self.assertEqual(sum(sum(a == ancst_matrix)), n ** 2)

    def test_multi_path_graph(self):
        n = 10
        g = nx.path_graph(n, create_using=nx.DiGraph)
        part_adj_matrix = nx.to_numpy_array(g)
        part_ancst_matrix = np.triu(np.ones((n, n)))
        np.fill_diagonal(part_ancst_matrix, 0)
        adj_matrix = np.zeros((2 * n, 2 * n))
        adj_matrix[0:n, 0:n] = part_adj_matrix
        adj_matrix[n:2 * n, n:2 * n] = part_adj_matrix
        ancst_matrix = np.zeros((2 * n, 2 * n))
        ancst_matrix[0:n, 0:n] = part_ancst_matrix
        ancst_matrix[n:2 * n, n:2 * n] = part_ancst_matrix
        a = create_ancestry_matrix_from_adjacency(adj_matrix)
        self.assertEqual(sum(sum(a == ancst_matrix)), (2 * n) ** 2)

    def test_multi_path_graph_with_different_lengths(self):
        n_1 = 10
        n_2 = 5
        g_1 = nx.path_graph(n_1, create_using=nx.DiGraph)
        adj_matrix_1 = nx.to_numpy_array(g_1)
        ancst_matrix_1 = np.triu(np.ones((n_1, n_1)))
        np.fill_diagonal(ancst_matrix_1, 0)
        g_2 = nx.path_graph(n_2, create_using=nx.DiGraph)
        adj_matrix_2 = nx.to_numpy_array(g_2)
        ancst_matrix_2 = np.triu(np.ones((n_2, n_2)))
        np.fill_diagonal(ancst_matrix_2, 0)
        adj_matrix = np.zeros((n_1 + n_2, n_1 + n_2))
        adj_matrix[0:n_1, 0:n_1] = adj_matrix_1
        adj_matrix[n_1:n_1 + n_2, n_1:n_1 + n_2] = adj_matrix_2
        ancst_matrix = np.zeros((n_1 + n_2, n_1 + n_2))
        ancst_matrix[0:n_1, 0:n_1] = ancst_matrix_1
        ancst_matrix[n_1:n_1 + n_2, n_1:n_1 + n_2] = ancst_matrix_2
        a = create_ancestry_matrix_from_adjacency(adj_matrix)
        self.assertEqual(sum(sum(a == ancst_matrix)), (n_1 + n_2) ** 2)

    def test_sample_trees_with_overlapping_leaves_1(self):
        # 0 -> 1 and 2
        # 3 -> 1 and 2
        adj_matrix = np.zeros((4, 4))
        adj_matrix[0, 1] = 1
        adj_matrix[0, 2] = 1
        adj_matrix[3, 1] = 1
        adj_matrix[3, 2] = 1
        ancst_matrix = adj_matrix.copy()
        a = create_ancestry_matrix_from_adjacency(adj_matrix)
        self.assertEqual(sum(sum(a == ancst_matrix)), 4 ** 2)
        self.assertEqual(number_of_root_nodes(ancst_matrix), 2)

    def test_sample_trees_with_overlapping_leaves_2(self):
        # 0 -> 1
        # 1 -> 2 and 3
        # 4 -> 3 and 5
        adj_matrix = np.zeros((6, 6))
        adj_matrix[0, 1] = 1
        adj_matrix[1, 2] = 1
        adj_matrix[1, 3] = 1
        adj_matrix[4, 3] = 1
        adj_matrix[4, 5] = 1
        ancst_matrix = adj_matrix.copy()
        ancst_matrix[0, 2] = 1
        ancst_matrix[0, 3] = 1
        a = create_ancestry_matrix_from_adjacency(adj_matrix)
        self.assertEqual(sum(sum(a == ancst_matrix)), 6 ** 2)
        self.assertEqual(number_of_root_nodes(ancst_matrix), 2)


class TestCreateAdjacencyFromAncestry(unittest.TestCase):

    def test_transitivity(self):
        n = 10
        seed = 1
        for i in range(10):
            a = create_random_adjacency_matrix(n, seed)
            b = create_random_ancestry_matrix(n, seed)
            c = create_adjacency_from_ancestry(b)
            self.assertEqual(sum(sum(a == c)), 100)
            seed += 1

    def test_path_tree(self):
        n = 10
        g = nx.path_graph(n, create_using=nx.DiGraph)
        adj_matrix = nx.to_numpy_array(g)
        ancst_matrix = np.triu(np.ones((n, n)))
        np.fill_diagonal(ancst_matrix, 0)
        a = create_adjacency_from_ancestry(ancst_matrix)
        self.assertEqual(sum(sum(a == adj_matrix)), n ** 2)

    def test_star_tree(self):
        n = 10
        adj_matrix = np.zeros((n, n))
        adj_matrix[0, 1:] = 1
        ancst_matrix = adj_matrix.copy()
        a = create_adjacency_from_ancestry(ancst_matrix)
        self.assertEqual(sum(sum(a == adj_matrix)), n ** 2)

    def test_multi_path_graph(self):
        n = 10
        g = nx.path_graph(n, create_using=nx.DiGraph)
        part_adj_matrix = nx.to_numpy_array(g)
        part_ancst_matrix = np.triu(np.ones((n, n)))
        np.fill_diagonal(part_ancst_matrix, 0)
        adj_matrix = np.zeros((2 * n, 2 * n))
        adj_matrix[0:n, 0:n] = part_adj_matrix
        adj_matrix[n:2 * n, n:2 * n] = part_adj_matrix
        ancst_matrix = np.zeros((2 * n, 2 * n))
        ancst_matrix[0:n, 0:n] = part_ancst_matrix
        ancst_matrix[n:2 * n, n:2 * n] = part_ancst_matrix
        a = create_adjacency_from_ancestry(ancst_matrix)
        self.assertEqual(sum(sum(a == adj_matrix)), (2 * n) ** 2)

    def test_multi_path_graph_with_different_lengths(self):
        n_1 = 10
        n_2 = 5
        g_1 = nx.path_graph(n_1, create_using=nx.DiGraph)
        adj_matrix_1 = nx.to_numpy_array(g_1)
        ancst_matrix_1 = np.triu(np.ones((n_1, n_1)))
        np.fill_diagonal(ancst_matrix_1, 0)
        g_2 = nx.path_graph(n_2, create_using=nx.DiGraph)
        adj_matrix_2 = nx.to_numpy_array(g_2)
        ancst_matrix_2 = np.triu(np.ones((n_2, n_2)))
        np.fill_diagonal(ancst_matrix_2, 0)
        adj_matrix = np.zeros((n_1 + n_2, n_1 + n_2))
        adj_matrix[0:n_1, 0:n_1] = adj_matrix_1
        adj_matrix[n_1:n_1 + n_2, n_1:n_1 + n_2] = adj_matrix_2
        ancst_matrix = np.zeros((n_1 + n_2, n_1 + n_2))
        ancst_matrix[0:n_1, 0:n_1] = ancst_matrix_1
        ancst_matrix[n_1:n_1 + n_2, n_1:n_1 + n_2] = ancst_matrix_2
        a = create_adjacency_from_ancestry(ancst_matrix)
        self.assertEqual(sum(sum(a == adj_matrix)), (n_1 + n_2) ** 2)


class TestAdding1Algorithm(unittest.TestCase):

    def test_add_simply_inferred_ancestry_relations(self):
        # 0 > 1
        # 1 > 2
        # 1 > 3
        # 3 > 4
        # 0 > 4
        partial_ancst_matrix = np.array([[0, 1, 0, 0, 1],
                                         [0, 0, 1, 1, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 1],
                                         [0, 0, 0, 0, 0]])
        ans = add_simply_inferred_ancestry_relations(partial_ancst_matrix)
        self.assertEqual(ans[0, 2], 1)
        self.assertEqual(ans[0, 3], 1)
        self.assertEqual(ans[1, 4], 1)

    def test_add_simply_inferred_ancestry_relations_2(self):
        n = 30
        seed = 20
        np.random.seed(seed)
        adj_mat = create_random_adjacency_matrix(n, seed)
        draw_tree(adj_mat)
        ancst_mat = create_ancestry_matrix_from_adjacency(adj_mat)
        ancst_deleted_many = ancst_mat.copy()
        ancst_deleted_few = ancst_mat.copy()
        for i in range(n):
            for j in range(n):
                if ancst_mat[i, j] == 1:
                    r = np.random.random()
                    if r < .6:
                        ancst_deleted_many[i, j] = 0
                    if r < .1:
                        ancst_deleted_few[i, j] = 0

        ancst_deleted_many_completed = add_simply_inferred_ancestry_relations(ancst_deleted_many)
        ancst_deleted_few_completed = add_simply_inferred_ancestry_relations(ancst_deleted_few)

        for i in range(n):
            for j in range(n):
                self.assertLessEqual(ancst_deleted_many_completed[i, j], ancst_deleted_few_completed[i, j])

        deleted_many_possible_root_nodes = find_possible_root_nodes(ancst_deleted_many_completed)
        deleted_few_possible_root_nodes = find_possible_root_nodes(ancst_deleted_few_completed)

        for possible_deleted_few_root in deleted_few_possible_root_nodes:
            self.assertIn(possible_deleted_few_root, deleted_many_possible_root_nodes)

        temp_sum = -1
        deleted_many_best_root = -1
        for possible_root in deleted_many_possible_root_nodes:
            if np.sum(ancst_deleted_many_completed[possible_root, :]) > temp_sum:
                deleted_many_best_root = possible_root
                temp_sum = np.sum(ancst_deleted_many_completed[possible_root, :])

        temp_sum = -1
        deleted_few_best_root = -1
        for possible_deleted_few_root in deleted_few_possible_root_nodes:
            if np.sum(ancst_deleted_few_completed[possible_deleted_few_root, :]) > temp_sum:
                deleted_few_best_root = possible_deleted_few_root
                temp_sum = np.sum(ancst_deleted_few_completed[possible_deleted_few_root, :])

        for i in range(n):
            if i != deleted_many_best_root:
                ancst_deleted_many_completed[deleted_many_best_root, i] = 1
            if i != deleted_few_best_root:
                ancst_deleted_few_completed[deleted_few_best_root, i] = 1

        ancst_deleted_few_completed_completed = add_simply_inferred_ancestry_relations(ancst_deleted_few_completed)
        ancst_deleted_many_completed_completed = add_simply_inferred_ancestry_relations(ancst_deleted_many_completed)

        for i in range(n):
            for j in range(n):
                self.assertEqual(ancst_deleted_many_completed_completed[i, j], ancst_deleted_many_completed[i, j])
                self.assertEqual(ancst_deleted_few_completed_completed[i, j], ancst_deleted_few_completed[i, j])

        # tree_1 = create_adjacency_from_ancestry(ancst_deleted_many_completed)
        # draw_tree(tree_1)
        # tree_2 = create_adjacency_from_ancestry(ancst_deleted_few_completed)
        # draw_tree(tree_2)

        self.assertLessEqual(np.sum(ancst_deleted_many_completed), np.sum(ancst_deleted_few_completed))
        self.assertLessEqual(np.sum(ancst_deleted_few_completed), np.sum(ancst_mat))

    def test_algorithm_logic_2(self):
        # simple rhombus DAG
        # TODO: this should be tested on an algorithm that returns tree
        n = 4
        c = np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
        c_completed = add_simply_inferred_ancestry_relations(c)
        c_possible_root_nodes = find_possible_root_nodes(c_completed)
        c_sum_ones = -1
        c_best_root = -1
        for c_p_root in c_possible_root_nodes:
            if np.sum(c_completed[c_p_root, :]) > c_sum_ones:
                c_best_root = c_p_root
                c_sum_ones = np.sum(c_completed[c_p_root, :])
        for i in range(n):
            if i != c_best_root:
                c_completed[c_best_root, i] = 1
        c_completed_completed = add_simply_inferred_ancestry_relations(c_completed)

        for i in range(n):
            for j in range(n):
                self.assertEqual(c_completed_completed[i, j], c_completed[i, j])

        tree_1 = create_adjacency_from_ancestry(c_completed)
        if tree_1:
            draw_tree(tree_1)
            ancst = create_ancestry_matrix_from_adjacency(tree_1)
            for i in range(n):
                for j in range(n):
                    self.assertEqual(ancst[i, j], c_completed[i, j])


# class TestGreedyAlgorithmTopDown(unittest.TestCase):
#     def test_adding_1_greedy_algorithm_1(self):
#         n = 4
#         c = np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
#         computed_ancestry = adding_1_greedy_algorithm_top_down(c)
#         tree_1 = create_adjacency_from_ancestry(computed_ancestry)
#         draw_tree(tree_1)
#         self.assertEqual(np.sum(computed_ancestry), 6)
#
#     def test_adding_1_greedy_algorithm_2(self):
#         n = 5
#         p1 = .7
#         p2 = .2
#
#         for iteration in range(121, 1000):
#             print(iteration)
#             seed = iteration
#             np.random.seed(seed)
#             a = create_random_adjacency_matrix(n, seed)
#             # draw_tree(a)
#             b = create_ancestry_matrix_from_adjacency(a)
#             c = b.copy()
#             d = b.copy()
#             for i in range(n):
#                 for j in range(n):
#                     if b[i, j] == 1:
#                         r = np.random.random()
#                         if r < p1:
#                             c[i, j] = 0
#                         if r < p2:
#                             d[i, j] = 0
#
#             c_completed = adding_1_greedy_algorithm_top_down(c)
#             d_completed = adding_1_greedy_algorithm_top_down(d)
#
#             tree_1 = create_adjacency_from_ancestry(c_completed)
#             # draw_tree(tree_1)
#             tree_2 = create_adjacency_from_ancestry(d_completed)
#             # draw_tree(tree_2)
#
#             sum_c = np.sum(c_completed)
#             sum_d = np.sum(d_completed)
#             sum_b = np.sum(b)
#             print(sum_c, sum_d, sum_b)
#             self.assertLessEqual(sum_c, sum_d)
#             self.assertLessEqual(sum_d, sum_b)


class TestGreedyAlgorithmBottomUP(unittest.TestCase):

    def test_get_available_parent_empty_list(self):
        parents_list = np.zeros((2,)) - 1
        p1 = get_available_parent(parents_list, 0)
        self.assertEqual(p1, 0)
        p2 = get_available_parent(parents_list, 1)
        self.assertEqual(p2, 1)

    def test_get_available_parent_path_tree(self):
        path_graph_parents_list = np.arange(-1, 9)
        for i in range(10):
            self.assertEqual(get_available_parent(path_graph_parents_list, i), 0)

    def test_get_available_parent_circle(self):
        circle_graph_parents_list = np.arange(-1, 9)
        circle_graph_parents_list[0] = 9
        for i in range(10):
            self.assertRaises(ValueError, get_available_parent, circle_graph_parents_list, i)

    def test_adding_1_greedy_algorithm_1(self):
        n = 4
        c = np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
        computed_ancestry = adding_1_greedy_algorithm_bottom_up(c)
        tree_1 = create_adjacency_from_ancestry(computed_ancestry)
        draw_tree(tree_1)
        self.assertEqual(np.sum(computed_ancestry), 6)

    # def test_adding_1_greedy_algorithm_2(self):
    #     # n=5, p1=.7, p2=.2, iteration=121
    #     n = 5
    #     p1 = .7
    #     p2 = .2
    #
    #     for iteration in range(121, 2000):
    #         print(iteration)
    #         seed = iteration
    #         np.random.seed(seed)
    #         a = create_random_adjacency_matrix(n, seed)
    #         draw_tree(a)
    #         b = create_ancestry_matrix_from_adjacency(a)
    #         c = b.copy()
    #         d = b.copy()
    #         for i in range(n):
    #             for j in range(n):
    #                 if b[i, j] == 1:
    #                     r = np.random.random()
    #                     if r < p1:
    #                         c[i, j] = 0
    #                     if r < p2:
    #                         d[i, j] = 0
    #
    #         c_completed = adding_1_greedy_algorithm_bottom_up(c)
    #         d_completed = adding_1_greedy_algorithm_bottom_up(d)
    #
    #         tree_1 = create_adjacency_from_ancestry(c_completed)
    #         draw_tree(tree_1)
    #         tree_2 = create_adjacency_from_ancestry(d_completed)
    #         draw_tree(tree_2)
    #
    #         sum_c = np.sum(c_completed)
    #         sum_d = np.sum(d_completed)
    #         sum_b = np.sum(b)
    #         print(sum_c, sum_d, sum_b)
    #         self.assertLessEqual(sum_c, sum_d)
    #         self.assertLessEqual(sum_d, sum_b)


class TestCreateDistanceMatrixFromAdjacency(unittest.TestCase):
    def test_path_tree(self):
        n = 5
        g = nx.path_graph(n, create_using=nx.DiGraph)
        adj_matrix = nx.to_numpy_array(g)
        dist_matrix = np.array([
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0]
        ])
        a = create_distance_matrix_from_adjacency(adj_matrix)
        for i in range(n):
            for j in range(n):
                self.assertEqual(a[i, j], dist_matrix[i, j])

    def test_star_tree(self):
        n = 10
        adj_matrix = np.zeros((n, n))
        adj_matrix[0, 1:] = 1
        dist_matrix = np.ones((n, n))
        dist_matrix -= np.identity(n)
        dist_matrix *= 2
        dist_matrix[0, 1:] = np.ones((1, n - 1))
        dist_matrix[1:, 0] = np.ones((1, n - 1))
        a = create_distance_matrix_from_adjacency(adj_matrix)
        for i in range(n):
            for j in range(n):
                self.assertEqual(a[i, j], dist_matrix[i, j])


class TestTurnIntoDirectedTree(unittest.TestCase):

    def test_random_1(self):
        n = 10
        dir_adj_matrix = create_random_adjacency_matrix(n)
        r = find_possible_root_nodes(dir_adj_matrix)[0]
        undirected_adj_matrix = dir_adj_matrix + np.transpose(dir_adj_matrix)
        s = turn_into_directed_tree(undirected_adj_matrix, r)
        for i in range(n):
            for j in range(n):
                self.assertEqual(s[i, j], dir_adj_matrix[i, j])

    def test_random_2(self):
        n = 100
        dir_adj_matrix = create_random_adjacency_matrix(n)
        r = find_possible_root_nodes(dir_adj_matrix)[0]
        undirected_adj_matrix = dir_adj_matrix + np.transpose(dir_adj_matrix)
        s = turn_into_directed_tree(undirected_adj_matrix, r)
        for i in range(n):
            for j in range(n):
                self.assertEqual(s[i, j], dir_adj_matrix[i, j])


class TestFindParentNode(unittest.TestCase):

    def test_1(self):
        adj_matrix = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        self.assertEqual(find_parent_node(adj_matrix, 0), None)
        self.assertEqual(find_parent_node(adj_matrix, 1), 0)
        self.assertEqual(find_parent_node(adj_matrix, 2), 1)

    def test_2(self):
        adj_matrix = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        self.assertEqual(find_parent_node(adj_matrix, 0), None)
        self.assertEqual(find_parent_node(adj_matrix, 1), 0)
        self.assertRaises(ValueError, find_parent_node, adj_matrix, 2)
