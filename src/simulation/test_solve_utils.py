import unittest

import numpy as np

from src.simulation.generate_sim import generate_simulations
from src.simulation.solve_utils import *


class TestComputeParametersFromFolderAddress(unittest.TestCase):

    def test_1(self):
        folder_address = "A/B/DFsdf/n10_k5_cp9_pc0_bm0_nr1_seed4"
        n, k, cp, pc, bm, nr, seed = compute_parameters_from_folder_address(folder_address)
        self.assertEqual(n, 10)
        self.assertEqual(k, 5)
        self.assertEqual(cp, .9)
        self.assertEqual(pc, 0)
        self.assertEqual(bm, 0)
        self.assertEqual(nr, 1)
        self.assertEqual(seed, 4)


class TestReadAllPerturbedTreesInTrial(unittest.TestCase):

    def test_1(self):
        generate_simulations(10, 5, 1, change_probability=.9, pc_change=False, bm_change=False, nr_change=True,
                             seed=4)
        folder_address = "data/n10_k5_cp9_pc0_bm0_nr1_seed4"
        trial_number = 0
        adj_matrices = read_all_perturbed_trees_in_trial(folder_address, trial_number)
        self.assertEqual(len(adj_matrices), 5)


class TestConvertTreeAdjacencyToTreeDict(unittest.TestCase):

    def test_1(self):
        adj_mat = np.array([[0, 1, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
        tree_dict = convert_tree_adjacency_to_tree_dict(adj_mat)
        self.assertEqual(len(tree_dict), 2)
        self.assertEqual(tree_dict[(0,)], [(1,)])
        self.assertEqual(tree_dict[(1,)], [(2,)])


class TestCreateAdjacencyFromCPDict(unittest.TestCase):

    def test_1(self):
        n = 10
        cp_dict = {(6,): (9,), (5,): (9,), (3,): (9,), (9,): (0,), (8,): (2,), (7,): (2,), (2,): (0,), (4,): (0,),
                   (1,): (0,)}
        adj_mat = create_adjacency_from_cp_dict(cp_dict, n)
        self.assertEqual(np.sum(adj_mat), 9)
