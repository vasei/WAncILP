import docplex.mp.model as cpx
import pandas as pd
import numpy as np


def generate_model_and_solve_it(initial_matrix, multi=False):
    n = initial_matrix.shape[0]
    index_set = range(1, n + 1)
    m = {(i, j): initial_matrix[i - 1, j - 1] for i in index_set for j in index_set}

    opt_model = cpx.Model(name="RANCE L2 IP Model")

    x_vars = {(i, j): opt_model.binary_var(name="x_{0}_{1}".format(i, j))
              for i in index_set for j in index_set}

    y_vars = {i: opt_model.integer_var(lb=0, ub=n, name="y_{0}".format(i)) for i in index_set}
    z_vars = {i: opt_model.binary_var(name="z_{0}".format(i)) for i in index_set}

    constraints = {("t", i, j, k): opt_model.add_constraint(
        ct=x_vars[i, j] + x_vars[j, k] - x_vars[i, k] <= 1,
        ctname="constraint_t_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set}
    if multi is False:
        constraints.update({("nc", i, j): opt_model.add_constraint(
            ct=x_vars[i, j] + x_vars[j, i] <= 1,
            ctname="constraint_nc_{0}_{1}".format(i, j))
            for i in index_set for j in index_set})
    constraints.update({("cd", i, j): opt_model.add_constraint(
        ct=x_vars[i, k] + x_vars[j, k] - x_vars[i, j] - x_vars[j, i] <= 1,
        ctname="constraint_cd_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set if i != j and j != k and i != k})
    constraints.update({("y", i): opt_model.add_constraint(
        ct=opt_model.sum(x_vars[i, j] for j in index_set) == y_vars[i],
        ctname="constraint_y_{0}".format(i))
        for i in index_set})
    constraints.update({"ri": opt_model.add_constraint(
        ct=opt_model.sum(z_vars[i] for i in index_set) == 1,
        ctname="constraint_ri")})
    constraints.update({("re", i): opt_model.add_constraint(
        ct=(n - 1) * z_vars[i] - y_vars[i] <= 0,
        ctname="constraint_re_{0}".format(i))
        for i in index_set})
    constraints.update({("or", i): opt_model.add_constraint(
        ct=(n - 2) * (z_vars[i] + 1) - y_vars[i] >= 0,
        ctname="constraint_or_{0}".format(i))
        for i in index_set})

    objective = opt_model.sum(x_vars[i, j] - 2 * x_vars[i, j] * m[i, j]
                              for i in index_set
                              for j in index_set)

    opt_model.minimize(objective)
    print("start_solving")
    opt_model.solve()
    print(opt_model.solve_details)

    X = np.zeros((n, n))
    for i in index_set:
        for j in index_set:
            X[i - 1, j - 1] = x_vars[i, j].sv
    # opt_df = pd.DataFrame.from_dict(x_vars, orient="index",
    #                                 columns=["variable_object"])
    # opt_df.index = pd.MultiIndex.from_tuples(opt_df.index,
    #                                          names=["column_i", "column_j"])
    # opt_df.reset_index(inplace=True)
    # opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.solution_value)
    return X


def generate_model_and_solve_it_l1(initial_matrix):
    n = initial_matrix.shape[0]
    index_set = range(1, n + 1)
    m = {(i, j): initial_matrix[i - 1, j - 1] for i in index_set for j in index_set}

    opt_model = cpx.Model(name="RANCE L1 IP Model")

    x_vars = {(i, j): opt_model.binary_var(name="x_{0}_{1}".format(i, j))
              for i in index_set for j in index_set}

    y_vars = {i: opt_model.integer_var(lb=0, ub=n, name="y_{0}".format(i)) for i in index_set}
    z_vars = {i: opt_model.binary_var(name="z_{0}".format(i)) for i in index_set}

    u_vars = {(i, j): opt_model.continuous_var(lb=0, ub=n - 1, name="u_{0}_{1}".format(i, j))
              for i in index_set for j in index_set}

    constraints = {("t", i, j, k): opt_model.add_constraint(
        ct=x_vars[i, j] + x_vars[j, k] - x_vars[i, k] <= 1,
        ctname="constraint_t_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set}
    constraints.update({("nc", i, j): opt_model.add_constraint(
        ct=x_vars[i, j] + x_vars[j, i] <= 1,
        ctname="constraint_nc_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})
    constraints.update({("cd", i, j): opt_model.add_constraint(
        ct=x_vars[i, k] + x_vars[j, k] - x_vars[i, j] - x_vars[j, i] <= 1,
        ctname="constraint_cd_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set if i != j and j != k and i != k})
    constraints.update({("y", i): opt_model.add_constraint(
        ct=opt_model.sum(x_vars[i, j] for j in index_set) == y_vars[i],
        ctname="constraint_y_{0}".format(i))
        for i in index_set})
    constraints.update({"ri": opt_model.add_constraint(
        ct=opt_model.sum(z_vars[i] for i in index_set) == 1,
        ctname="constraint_ri")})
    constraints.update({("re", i): opt_model.add_constraint(
        ct=(n - 1) * z_vars[i] - y_vars[i] <= 0,
        ctname="constraint_re_{0}".format(i))
        for i in index_set})
    constraints.update({("or", i): opt_model.add_constraint(
        ct=(n - 2) * (z_vars[i] + 1) - y_vars[i] >= 0,
        ctname="constraint_or_{0}".format(i))
        for i in index_set})
    constraints.update({("u_vars_1", i, j): opt_model.add_constraint(
        ct=u_vars[i, j] >= x_vars[i, j] - m[i, j],
        ctname="constraint_u_vars_1_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})
    constraints.update({("u_vars_2", i, j): opt_model.add_constraint(
        ct=u_vars[i, j] >= m[i, j] - x_vars[i, j],
        ctname="constraint_u_vars_2_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})

    objective = opt_model.sum(u_vars[i, j]
                              for i in index_set
                              for j in index_set)

    opt_model.minimize(objective)
    print("start_solving")
    opt_model.solve()
    print(opt_model.solve_details)

    X = np.zeros((n, n))
    for i in index_set:
        for j in index_set:
            X[i - 1, j - 1] = x_vars[i, j].sv
    # opt_df = pd.DataFrame.from_dict(x_vars, orient="index",
    #                                 columns=["variable_object"])
    # opt_df.index = pd.MultiIndex.from_tuples(opt_df.index,
    #                                          names=["column_i", "column_j"])
    # opt_df.reset_index(inplace=True)
    # opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.solution_value)
    return X


def generate_gl_rooted_model_and_solve_it(initial_matrix):
    n = initial_matrix.shape[0]
    index_set = range(1, n + 1)
    m = {(i, j): initial_matrix[i - 1, j - 1] for i in index_set for j in index_set}

    opt_model = cpx.Model(name="RADJE IP Model")

    x_vars = {(i, j): opt_model.binary_var(name="x_{0}_{1}".format(i, j))
              for i in index_set for j in index_set}

    constraints = {("t", i, j, k): opt_model.add_constraint(
        ct=x_vars[i, j] + x_vars[j, k] - x_vars[i, k] <= 1,
        ctname="constraint_t_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set}
    constraints.update({("nc", i, j): opt_model.add_constraint(
        ct=x_vars[i, j] + x_vars[j, i] <= 1,
        ctname="constraint_nc_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})
    constraints.update({("cd", i, j): opt_model.add_constraint(
        ct=x_vars[i, k] + x_vars[j, k] - x_vars[i, j] - x_vars[j, i] <= 1,
        ctname="constraint_cd_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set if i != j and j != k and i != k})

    objective = opt_model.sum(x_vars[i, j] - 2 * x_vars[i, j] * m[i, j]
                              for i in index_set
                              for j in index_set)

    opt_model.minimize(objective)
    print("start_solving")
    opt_model.solve()
    print(opt_model.solve_details)

    X = np.zeros((n, n))
    for i in index_set:
        for j in index_set:
            X[i - 1, j - 1] = x_vars[i, j].sv

    return X
