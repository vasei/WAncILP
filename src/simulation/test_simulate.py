import unittest
from src.simulation.simulate import *
import networkx as nx


class TestGenerateRandomNumbersToSum(unittest.TestCase):

    def test_1(self):
        numbers = generate_random_numbers_to_sum(2, 1)
        self.assertAlmostEqual(sum(numbers), 1, 6)

    def test_2(self):
        numbers = generate_random_numbers_to_sum(10, .2)
        self.assertAlmostEqual(sum(numbers), .2, 6)

    def test_3(self):
        numbers = generate_random_numbers_to_sum(100, .02)
        self.assertAlmostEqual(sum(numbers), .02, 6)


class TestGenerateCCF(unittest.TestCase):
    def test_1(self):
        n = 100
        adj = create_random_adjacency_matrix(n, 104)
        ccf = generate_ccf(adj)
        for i in range(n):
            children = find_children(adj, i)
            self.assertGreater(ccf[i], 0)
            self.assertLess(sum(ccf[children]), ccf[i])


class TestComputeEmptyCCF(unittest.TestCase):
    def test_1(self):
        n = 500
        adj = create_random_adjacency_matrix(n, 104)
        ccf = generate_ccf(adj)
        empty_ccf = compute_empty_ccf(adj, ccf)
        self.assertAlmostEqual(sum(empty_ccf), 1, 5)


class TestPerturbTreeSubstituteParentChild(unittest.TestCase):

    def test_success_prob_0(self):
        n = 5
        adj_mat = create_random_adjacency_matrix(n, 0)
        ccf = generate_ccf(adj_mat)
        pert_adj_mat, pert_ccf = perturb_tree_substitute_parent_child(adj_mat, ccf, 0, 0)
        for i in range(n):
            for j in range(n):
                self.assertEqual(adj_mat[i, j], pert_adj_mat[i, j])

    def test_1(self):
        n = 5
        adj_mat = create_random_adjacency_matrix(n, 0)
        ccf = generate_ccf(adj_mat)
        pert_adj_mat, pert_ccf = perturb_tree_substitute_parent_child(adj_mat, ccf, 0.9, 0)
        g1 = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
        g2 = nx.from_numpy_array(pert_adj_mat, create_using=nx.DiGraph)
        self.assertEqual(nx.is_isomorphic(g1, g2), True)

    def test_2(self):
        n = 20
        adj_mat = create_random_adjacency_matrix(n, 0)
        ccf = generate_ccf(adj_mat)
        pert_adj_mat, pert_ccf = perturb_tree_substitute_parent_child(adj_mat, ccf, 0.9, 0)
        g1 = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
        g2 = nx.from_numpy_array(pert_adj_mat, create_using=nx.DiGraph)
        self.assertEqual(nx.is_isomorphic(g1, g2), True)


class TestPerturbTreeDeleteNode(unittest.TestCase):

    def test_success_prob_0(self):
        n = 100
        adj_mat = create_random_adjacency_matrix(n, 0)
        ccf = generate_ccf(adj_mat)
        pert_adj_mat, pert_ccf = perturb_tree_delete_node(adj_mat, ccf, 0, 10)
        for i in range(n):
            for j in range(n):
                self.assertEqual(adj_mat[i, j], pert_adj_mat[i, j])

    def test_1(self):
        n = 5
        adj_mat = create_random_adjacency_matrix(n, 0)
        ccf = generate_ccf(adj_mat)
        pert_adj_mat, pert_ccf = perturb_tree_delete_node(adj_mat, ccf, 0.9, 2)
        root1 = find_root_node(adj_mat)
        children1 = find_children(adj_mat, root1)
        root2 = find_root_node(pert_adj_mat)
        children2 = find_children(pert_adj_mat, root2)
        self.assertGreater(len(children2), len(children1))
        for child in children1:
            self.assertIn(child, children2)

    def test_2(self):
        n = 100
        adj_mat = create_random_adjacency_matrix(n, 0)
        ccf = generate_ccf(adj_mat)
        pert_adj_mat, pert_ccf = perturb_tree_delete_node(adj_mat, ccf, 0.9, 5)
        root1 = find_root_node(adj_mat)
        children1 = find_children(adj_mat, root1)
        root2 = find_root_node(pert_adj_mat)
        children2 = find_children(pert_adj_mat, root2)
        self.assertGreater(len(children2), len(children1))
        for child in children1:
            self.assertIn(child, children2)


class TestPerturbTreeMoveBranch(unittest.TestCase):

    def test_success_prob_0(self):
        n = 100
        adj_mat = create_random_adjacency_matrix(n, 0)
        ccf = generate_ccf(adj_mat)
        pert_adj_mat, pert_ccf = perturb_tree_move_branch(adj_mat, ccf, 0, 10)
        for i in range(n):
            for j in range(n):
                self.assertEqual(adj_mat[i, j], pert_adj_mat[i, j])

    def test_1(self):
        n = 5
        adj_mat = create_random_adjacency_matrix(n, 0)
        ccf = generate_ccf(adj_mat)
        pert_adj_mat, pert_ccf = perturb_tree_move_branch(adj_mat, ccf, 0.9, 2)
        root1 = find_root_node(adj_mat)
        root2 = find_root_node(pert_adj_mat)
        self.assertEqual(root1, root2)
        for i in range(n):
            for j in range(n):
                if adj_mat[i, j] != pert_adj_mat[i, j]:
                    flag = True
        self.assertEqual(flag, True)

    def test_2(self):
        n = 100
        adj_mat = create_random_adjacency_matrix(n, 0)
        ccf = generate_ccf(adj_mat)
        pert_adj_mat, pert_ccf = perturb_tree_move_branch(adj_mat, ccf, 0.9, 5)
        root1 = find_root_node(adj_mat)
        root2 = find_root_node(pert_adj_mat)
        self.assertEqual(root1, root2)
        flag = False
        for i in range(n):
            for j in range(n):
                if adj_mat[i, j] != pert_adj_mat[i, j]:
                    flag = True
        self.assertEqual(flag, True)
