# WAncILP (Weighted Ancestry Integer Linear Program)

This repository contains the code accompany the paper titled: "The Centroid Tree Problem: A novel approach
forsummarizing phylogenies in tumor tree inference".
This work is done by Hamed Vasei under the supervision of professors Mohammad Hadi Foroughmand Araabi and Amir
Daneshagar.
Any correspondence can be directed to Prof. Foroughmand at foroughmand@sharif.edu.

## Requirements

All python requirements are written in the `requirements.txt` file.
In order to solve mixed integer
programms [CPlex Optimizer](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer) should be
installed on the machine.
In order to visualize graphs the [Graphviz](https://graphviz.org/) needs to be installed.

## Generating Simulated Data

You can use the function `generate_simulations` from `src/simulatoin/generate_sim.py` file:

```
generate_simulations(tree_nodes,
                     number_of_perturbed_trees,
                     number_of_simulations,
                     change_probability=.9,
                     pc_change=True,
                     bm_change=True,
                     nr_change=True,
                     seed=0)
```

All simulation settings that are used in the paper are commented in `src/simulatoin/generate_sim.py`.

## Running ILPs

Ancestry ILPs can be solved by running the file `src/simulation/ilp_ancst_solve_simulation.py`.
 