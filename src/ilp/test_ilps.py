import numpy as np
from unittest import TestCase

from src.ilp.ancst import generate_model_and_solve_it, generate_gl_rooted_model_and_solve_it
from src.utils import number_of_root_nodes, draw_tree, create_adjacency_from_ancestry, create_adjacency_from_dist, \
    draw_tree_unidrected
from src.ilp.dist import generate_model_and_solve_it as generate_model_and_solve_it2


class TestAncstILP(TestCase):

    def test_generate_model_1(self):
        a = np.zeros((3, 3))
        ancst_matrix = generate_model_and_solve_it(a)
        self.assertEqual(number_of_root_nodes(ancst_matrix), 1)

    def test_generate_model_2(self):
        a = np.zeros((7, 7))
        ancst_matrix = generate_model_and_solve_it(a)
        self.assertEqual(number_of_root_nodes(ancst_matrix), 1)
        self.assertEqual(sum(sum(ancst_matrix)), 6)

    def test_generate_model_3(self):
        a = np.ones((7, 7))
        ancst_matrix = generate_model_and_solve_it(a)
        self.assertEqual(number_of_root_nodes(ancst_matrix), 1)
        self.assertEqual(sum(sum(ancst_matrix)), 6 + 5 + 4 + 3 + 2 + 1)
        # adj_matrix = create_adjacency_from_ancestry(ancst_matrix)
        # draw_tree(adj_matrix)

    def test_generate_model_4(self):
        a = np.zeros((20, 20))
        ancst_matrix = generate_model_and_solve_it(a)
        self.assertEqual(number_of_root_nodes(ancst_matrix), 1)
        self.assertEqual(sum(sum(ancst_matrix)), 19)

    def test_generate_model_5(self):
        np.random.seed(0)
        a = np.random.random((10, 10))
        ancst_matrix = generate_model_and_solve_it(a)
        self.assertEqual(number_of_root_nodes(ancst_matrix), 1)
        adj_matrix = create_adjacency_from_ancestry(ancst_matrix)
        draw_tree(adj_matrix)

    def test_gl_model_1(self):
        np.random.seed(0)
        a = np.random.random((10, 10))
        ancst_matrix = generate_gl_rooted_model_and_solve_it(a)
        self.assertGreaterEqual(number_of_root_nodes(ancst_matrix), 1)
        adj_matrix = create_adjacency_from_ancestry(ancst_matrix)
        draw_tree(adj_matrix)


class TestDistILP(TestCase):
    def test_generate_model_1(self):
        a = np.zeros((3, 3))
        dist_matrix, _, _, _ = generate_model_and_solve_it2(a)
        print("DIST", dist_matrix)
        adj_matrix = create_adjacency_from_dist(dist_matrix)
        print("ADJ", adj_matrix)
        draw_tree_unidrected(adj_matrix)

    def test_model_1(self):
        np.random.seed(0)
        a = np.random.random((5, 5))
        dist_matrix, _, _, _ = generate_model_and_solve_it2(a)
        print("DIST", dist_matrix)
        adj_matrix = create_adjacency_from_dist(dist_matrix)
        print("ADJ", adj_matrix)
        draw_tree_unidrected(adj_matrix)

    def test_model_2(self):
        np.random.seed(10)
        a = np.random.random((8, 8))
        dist_matrix, _, _, _ = generate_model_and_solve_it2(a)
        print("DIST", dist_matrix)
        adj_matrix = create_adjacency_from_dist(dist_matrix)
        print("ADJ", adj_matrix)
        draw_tree_unidrected(adj_matrix)

    def test_model_3(self):
        a = np.array([[0, 1, 1, 1], [1, 0, 2, 2], [1, 2, 0, 2], [1, 2, 2, 0]])
        b = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
        D, E, Z, status = generate_model_and_solve_it2(a)
        print("DIST", D)
        adj_matrix = create_adjacency_from_dist(D)
        print("ADJ", adj_matrix)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(D[i, j], a[i, j], 2)
                self.assertAlmostEqual(E[i, j], b[i, j], 2)
                self.assertAlmostEqual(adj_matrix[i, j], b[i, j], 2)

        z_ones = [(2, 3, 1), (3, 2, 1), (2, 4, 1), (4, 2, 1), (3, 4, 1), (4, 3, 1)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if (i + 1, j + 1, k + 1) in z_ones:
                        self.assertAlmostEqual(Z[i, j, k], 1, 2)
                    else:
                        self.assertAlmostEqual(Z[i, j, k], 0, 2)

        draw_tree_unidrected(adj_matrix)

    def test_model_4(self):
        np.random.seed(10)
        a = np.random.randint(0, 5, (5, 5))
        dist_matrix, _, _, _ = generate_model_and_solve_it2(a)
        print("DIST", dist_matrix)
        adj_matrix = create_adjacency_from_dist(dist_matrix)
        print("ADJ", adj_matrix)
        draw_tree_unidrected(adj_matrix)

    def test_model_4(self):
        # np.random.seed(10)
        n = 8
        a = np.random.randint(0, 5, (5, 5))
        dist_matrix, _, _, _ = generate_model_and_solve_it2(a)
        print("DIST", dist_matrix)
        adj_matrix = create_adjacency_from_dist(dist_matrix)
        print("ADJ", adj_matrix)
        draw_tree_unidrected(adj_matrix)