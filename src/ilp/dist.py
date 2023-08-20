import docplex.mp.model as cpx
import numpy as np


def generate_model_and_solve_it(initial_matrix, timelimit=None, starting_point_distance_matrix=None):
    n = initial_matrix.shape[0]
    index_set = range(1, n + 1)
    m = {(i, j): initial_matrix[i - 1, j - 1] for i in index_set for j in index_set}

    opt_model = cpx.Model(name="NDIST IP Model")
    if (timelimit):
        opt_model.parameters.timelimit = timelimit

    # TODO: ub=n - 1
    u_vars = {(i, j): opt_model.continuous_var(lb=0, ub=n - 1, name="u_{0}_{1}".format(i, j))
              for i in index_set for j in index_set}
    d_vars = {(i, j): opt_model.integer_var(lb=0, ub=n - 1, name="d_{0}_{1}".format(i, j))
              for i in index_set for j in index_set}
    e_vars = {(i, j): opt_model.binary_var(name="e_{0}_{1}".format(i, j)) for i in index_set for j in index_set}
    z_vars = {(i, j, k): opt_model.binary_var(name="z_{0}_{1}_{2}".format(i, j, k)) for i in index_set for j in
              index_set for k in index_set}

    constraints = {("tr", i, j, k): opt_model.add_constraint(
        ct=d_vars[i, j] - d_vars[j, k] - d_vars[i, k] <= 0,
        ctname="constraint_triangular_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set if i != j and j != k and k != i}
    constraints.update({("sy", i, j): opt_model.add_constraint(
        ct=d_vars[i, j] - d_vars[j, i] == 0,
        ctname="constraint_symmetry_{0}_{1}".format(i, j))
        for i in index_set for j in index_set if i != j})
    constraints.update({("refl", i): opt_model.add_constraint(
        ct=d_vars[i, i] == 0,
        ctname="constraint_refl_{0}".format(i))
        for i in index_set})
    # constraints.update({("dist", i, j): opt_model.add_constraint(
    #     ct=d_vars[i, j] >= 1,
    #     ctname="constraint_dist_{0}_{1}".format(i, j))
    #     for i in index_set for j in index_set if i != j})
    # constraints.update({("edge", i, j): opt_model.add_constraint(
    #     ct=e_vars[i, j] + d_vars[i, j] >= 2,
    #     ctname="constraint_edge_{0}_{1}".format(i, j))
    #     for i in index_set for j in index_set if i != j})
    constraints.update({("sy2", i, j): opt_model.add_constraint(
        ct=e_vars[i, j] - e_vars[j, i] == 0,
        ctname="constraint_symmetry2_{0}_{1}".format(i, j))
        for i in index_set for j in index_set if i != j})
    constraints.update({("refl2", i): opt_model.add_constraint(
        ct=e_vars[i, i] == 0,
        ctname="constraint_refl2_{0}".format(i))
        for i in index_set})
    constraints.update({"edge_number": opt_model.add_constraint(
        ct=opt_model.sum(e_vars[i, j] for i in index_set for j in index_set) == 2 * n - 2,
        ctname="constraint_edge_number")})
    constraints.update({("z_dist", i): opt_model.add_constraint(
        ct=opt_model.sum(z_vars[i, j, k] for k in index_set) == d_vars[i, j] - 1,
        ctname="constraint_z_dist_{0}_{1}".format(i, j))
        for i in index_set for j in index_set if i != j})
    constraints.update({("z_triple", i, j): opt_model.add_constraint(
        ct=z_vars[i, j, k] + z_vars[i, k, j] + z_vars[j, k, i] <= 1,
        ctname="constraint_z_triple_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set})
    # # constraints.update({("refl", i): opt_model.add_constraint(
    # #     ct=opt_model.sum(d_vars[i, j] for j in index_set) == e_vars[i],
    # #     ctname="constraint_refl_{0}".format(i))
    # #     for i in index_set})
    constraints.update({("sy3", i, j, k): opt_model.add_constraint(
        ct=z_vars[i, j, k] - z_vars[j, i, k] == 0,
        ctname="constraint_symmetry3_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set})
    constraints.update({("path_start", i, j): opt_model.add_constraint(
        ct=z_vars[i, i, j] == 0,
        ctname="constraint_path_start_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})
    constraints.update({("path_end", i, j): opt_model.add_constraint(
        ct=z_vars[i, j, j] == 0,
        ctname="constraint_path_end_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})
    constraints.update({("z_tree_mid", i, j, k): opt_model.add_constraint(
        ct=e_vars[i, j] - z_vars[i, k, j] - z_vars[j, k, i] <= 0,
        ctname="constraint_z_tree_mid_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set if i != j and i != k and j != k})
    constraints.update({("additive_distance", i, j, k): opt_model.add_constraint(
        ct=d_vars[i, j] + 2 * n * (1 - z_vars[i, j, k]) - d_vars[j, k] - d_vars[i, k] >= 0,
        ctname="constraint_additive_distance_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set if i != j and i != k and j != k})
    constraints.update({("u_vars_1", i, j): opt_model.add_constraint(
        ct=u_vars[i, j] >= d_vars[i, j] - m[i, j],
        ctname="constraint_u_vars_1_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})
    constraints.update({("u_vars_2", i, j): opt_model.add_constraint(
        ct=u_vars[i, j] >= m[i, j] - d_vars[i, j],
        ctname="constraint_u_vars_2_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})

    objective = opt_model.sum(u_vars[i, j]
                              for i in index_set
                              for j in index_set)

    opt_model.minimize(objective)

    if starting_point_distance_matrix is not None:
        print("setting starting point")
        warmstart = opt_model.new_solution()
        for i in range(n):
            for j in range(i + 1, n):
                warmstart.add_var_value(d_vars[i + 1, j + 1], starting_point_distance_matrix[i, j])
                warmstart.add_var_value(d_vars[j + 1, i + 1], starting_point_distance_matrix[j, i])
                if starting_point_distance_matrix[i, j] == 1:
                    warmstart.add_var_value(e_vars[j + 1, i + 1], 1)
                    warmstart.add_var_value(e_vars[i + 1, j + 1], 1)
        opt_model.add_mip_start(warmstart)

    print("start_solving")
    opt_model.solve()
    print(opt_model.solve_details)

    D = np.zeros((n, n))
    E = np.zeros((n, n))
    Z = np.zeros((n, n, n))
    for i in index_set:
        for j in index_set:
            D[i - 1, j - 1] = d_vars[i, j].sv
            E[i - 1, j - 1] = e_vars[i, j].sv
            for k in index_set:
                Z[i - 1, j - 1, k - 1] = z_vars[i, j, k].sv
    # opt_df = pd.DataFrame.from_dict(x_vars, orient="index",
    #                                 columns=["variable_object"])
    # opt_df.index = pd.MultiIndex.from_tuples(opt_df.index,
    #                                          names=["column_i", "column_j"])
    # opt_df.reset_index(inplace=True)
    # opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.solution_value)
    return D, E, Z, opt_model.solve_details.status


def generate_model_and_solve_it_l2(initial_matrix, timelimit=None, starting_point_distance_matrix=None):
    n = initial_matrix.shape[0]
    index_set = range(1, n + 1)
    m = {(i, j): initial_matrix[i - 1, j - 1] for i in index_set for j in index_set}

    opt_model = cpx.Model(name="NDIST IP Model")
    if (timelimit):
        opt_model.parameters.timelimit = timelimit

    d_vars = {(i, j): opt_model.integer_var(lb=0, ub=n - 1, name="d_{0}_{1}".format(i, j))
              for i in index_set for j in index_set}
    e_vars = {(i, j): opt_model.binary_var(name="e_{0}_{1}".format(i, j)) for i in index_set for j in index_set}
    z_vars = {(i, j, k): opt_model.binary_var(name="z_{0}_{1}_{2}".format(i, j, k)) for i in index_set for j in
              index_set for k in index_set}

    constraints = {("tr", i, j, k): opt_model.add_constraint(
        ct=d_vars[i, j] - d_vars[j, k] - d_vars[i, k] <= 0,
        ctname="constraint_triangular_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set if i != j and j != k and k != i}
    constraints.update({("sy", i, j): opt_model.add_constraint(
        ct=d_vars[i, j] - d_vars[j, i] == 0,
        ctname="constraint_symmetry_{0}_{1}".format(i, j))
        for i in index_set for j in index_set if i != j})
    constraints.update({("refl", i): opt_model.add_constraint(
        ct=d_vars[i, i] == 0,
        ctname="constraint_refl_{0}".format(i))
        for i in index_set})
    # constraints.update({("dist", i, j): opt_model.add_constraint(
    #     ct=d_vars[i, j] >= 1,
    #     ctname="constraint_dist_{0}_{1}".format(i, j))
    #     for i in index_set for j in index_set if i != j})
    # constraints.update({("edge", i, j): opt_model.add_constraint(
    #     ct=e_vars[i, j] + d_vars[i, j] >= 2,
    #     ctname="constraint_edge_{0}_{1}".format(i, j))
    #     for i in index_set for j in index_set if i != j})
    constraints.update({("sy2", i, j): opt_model.add_constraint(
        ct=e_vars[i, j] - e_vars[j, i] == 0,
        ctname="constraint_symmetry2_{0}_{1}".format(i, j))
        for i in index_set for j in index_set if i != j})
    constraints.update({("refl2", i): opt_model.add_constraint(
        ct=e_vars[i, i] == 0,
        ctname="constraint_refl2_{0}".format(i))
        for i in index_set})
    constraints.update({"edge_number": opt_model.add_constraint(
        ct=opt_model.sum(e_vars[i, j] for i in index_set for j in index_set) == 2 * n - 2,
        ctname="constraint_edge_number")})
    constraints.update({("z_dist", i): opt_model.add_constraint(
        ct=opt_model.sum(z_vars[i, j, k] for k in index_set) == d_vars[i, j] - 1,
        ctname="constraint_z_dist_{0}_{1}".format(i, j))
        for i in index_set for j in index_set if i != j})
    constraints.update({("z_triple", i, j): opt_model.add_constraint(
        ct=z_vars[i, j, k] + z_vars[i, k, j] + z_vars[j, k, i] <= 1,
        ctname="constraint_z_triple_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set})
    # # constraints.update({("refl", i): opt_model.add_constraint(
    # #     ct=opt_model.sum(d_vars[i, j] for j in index_set) == e_vars[i],
    # #     ctname="constraint_refl_{0}".format(i))
    # #     for i in index_set})
    constraints.update({("sy3", i, j, k): opt_model.add_constraint(
        ct=z_vars[i, j, k] - z_vars[j, i, k] == 0,
        ctname="constraint_symmetry3_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set})
    constraints.update({("path_start", i, j): opt_model.add_constraint(
        ct=z_vars[i, i, j] == 0,
        ctname="constraint_path_start_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})
    constraints.update({("path_end", i, j): opt_model.add_constraint(
        ct=z_vars[i, j, j] == 0,
        ctname="constraint_path_end_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})
    constraints.update({("z_tree_mid", i, j, k): opt_model.add_constraint(
        ct=e_vars[i, j] - z_vars[i, k, j] - z_vars[j, k, i] <= 0,
        ctname="constraint_z_tree_mid_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set if i != j and i != k and j != k})
    constraints.update({("additive_distance", i, j, k): opt_model.add_constraint(
        ct=d_vars[i, j] + 2 * n * (1 - z_vars[i, j, k]) - d_vars[j, k] - d_vars[i, k] >= 0,
        ctname="constraint_additive_distance_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set if i != j and i != k and j != k})

    objective = opt_model.sum(d_vars[i, j] ** 2 + m[i, j] ** 2 - 2 * d_vars[i, j] * m[i, j]
                              for i in index_set
                              for j in index_set)

    opt_model.minimize(objective)

    if starting_point_distance_matrix is not None:
        print("setting starting point")
        warmstart = opt_model.new_solution()
        for i in range(n):
            for j in range(i + 1, n):
                warmstart.add_var_value(d_vars[i + 1, j + 1], starting_point_distance_matrix[i, j])
                warmstart.add_var_value(d_vars[j + 1, i + 1], starting_point_distance_matrix[j, i])
                if starting_point_distance_matrix[i, j] == 1:
                    warmstart.add_var_value(e_vars[j + 1, i + 1], 1)
                    warmstart.add_var_value(e_vars[i + 1, j + 1], 1)
        opt_model.add_mip_start(warmstart)

    print("start_solving")
    opt_model.solve()
    print(opt_model.solve_details)

    D = np.zeros((n, n))
    E = np.zeros((n, n))
    Z = np.zeros((n, n, n))
    for i in index_set:
        for j in index_set:
            D[i - 1, j - 1] = d_vars[i, j].sv
            E[i - 1, j - 1] = e_vars[i, j].sv
            for k in index_set:
                Z[i - 1, j - 1, k - 1] = z_vars[i, j, k].sv
    # opt_df = pd.DataFrame.from_dict(x_vars, orient="index",
    #                                 columns=["variable_object"])
    # opt_df.index = pd.MultiIndex.from_tuples(opt_df.index,
    #                                          names=["column_i", "column_j"])
    # opt_df.reset_index(inplace=True)
    # opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.solution_value)
    return D, E, Z, opt_model.solve_details.status


def generate_model_and_solve_it_inf(initial_matrix, timelimit=None, starting_point_distance_matrix=None):
    n = initial_matrix.shape[0]
    index_set = range(1, n + 1)
    m = {(i, j): initial_matrix[i - 1, j - 1] for i in index_set for j in index_set}

    opt_model = cpx.Model(name="NDIST IP Model")
    if (timelimit):
        opt_model.parameters.timelimit = timelimit

    # TODO: ub=n - 1
    u_var = opt_model.continuous_var(lb=0, ub=n - 1, name="u")
    d_vars = {(i, j): opt_model.integer_var(lb=0, ub=n - 1, name="d_{0}_{1}".format(i, j))
              for i in index_set for j in index_set}
    e_vars = {(i, j): opt_model.binary_var(name="e_{0}_{1}".format(i, j)) for i in index_set for j in index_set}
    z_vars = {(i, j, k): opt_model.binary_var(name="z_{0}_{1}_{2}".format(i, j, k)) for i in index_set for j in
              index_set for k in index_set}

    constraints = {("tr", i, j, k): opt_model.add_constraint(
        ct=d_vars[i, j] - d_vars[j, k] - d_vars[i, k] <= 0,
        ctname="constraint_triangular_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set if i != j and j != k and k != i}
    constraints.update({("sy", i, j): opt_model.add_constraint(
        ct=d_vars[i, j] - d_vars[j, i] == 0,
        ctname="constraint_symmetry_{0}_{1}".format(i, j))
        for i in index_set for j in index_set if i != j})
    constraints.update({("refl", i): opt_model.add_constraint(
        ct=d_vars[i, i] == 0,
        ctname="constraint_refl_{0}".format(i))
        for i in index_set})
    # constraints.update({("dist", i, j): opt_model.add_constraint(
    #     ct=d_vars[i, j] >= 1,
    #     ctname="constraint_dist_{0}_{1}".format(i, j))
    #     for i in index_set for j in index_set if i != j})
    # constraints.update({("edge", i, j): opt_model.add_constraint(
    #     ct=e_vars[i, j] + d_vars[i, j] >= 2,
    #     ctname="constraint_edge_{0}_{1}".format(i, j))
    #     for i in index_set for j in index_set if i != j})
    constraints.update({("sy2", i, j): opt_model.add_constraint(
        ct=e_vars[i, j] - e_vars[j, i] == 0,
        ctname="constraint_symmetry2_{0}_{1}".format(i, j))
        for i in index_set for j in index_set if i != j})
    constraints.update({("refl2", i): opt_model.add_constraint(
        ct=e_vars[i, i] == 0,
        ctname="constraint_refl2_{0}".format(i))
        for i in index_set})
    constraints.update({"edge_number": opt_model.add_constraint(
        ct=opt_model.sum(e_vars[i, j] for i in index_set for j in index_set) == 2 * n - 2,
        ctname="constraint_edge_number")})
    constraints.update({("z_dist", i): opt_model.add_constraint(
        ct=opt_model.sum(z_vars[i, j, k] for k in index_set) == d_vars[i, j] - 1,
        ctname="constraint_z_dist_{0}_{1}".format(i, j))
        for i in index_set for j in index_set if i != j})
    constraints.update({("z_triple", i, j): opt_model.add_constraint(
        ct=z_vars[i, j, k] + z_vars[i, k, j] + z_vars[j, k, i] <= 1,
        ctname="constraint_z_triple_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set})
    # # constraints.update({("refl", i): opt_model.add_constraint(
    # #     ct=opt_model.sum(d_vars[i, j] for j in index_set) == e_vars[i],
    # #     ctname="constraint_refl_{0}".format(i))
    # #     for i in index_set})
    constraints.update({("sy3", i, j, k): opt_model.add_constraint(
        ct=z_vars[i, j, k] - z_vars[j, i, k] == 0,
        ctname="constraint_symmetry3_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set})
    constraints.update({("path_start", i, j): opt_model.add_constraint(
        ct=z_vars[i, i, j] == 0,
        ctname="constraint_path_start_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})
    constraints.update({("path_end", i, j): opt_model.add_constraint(
        ct=z_vars[i, j, j] == 0,
        ctname="constraint_path_end_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})
    constraints.update({("z_tree_mid", i, j, k): opt_model.add_constraint(
        ct=e_vars[i, j] - z_vars[i, k, j] - z_vars[j, k, i] <= 0,
        ctname="constraint_z_tree_mid_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set if i != j and i != k and j != k})
    constraints.update({("additive_distance", i, j, k): opt_model.add_constraint(
        ct=d_vars[i, j] + 2 * n * (1 - z_vars[i, j, k]) - d_vars[j, k] - d_vars[i, k] >= 0,
        ctname="constraint_additive_distance_{0}_{1}_{2}".format(i, j, k))
        for i in index_set for j in index_set for k in index_set if i != j and i != k and j != k})
    constraints.update({("u_vars_1", i, j): opt_model.add_constraint(
        ct=u_var >= d_vars[i, j] - m[i, j],
        ctname="constraint_u_vars_1_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})
    constraints.update({("u_vars_2", i, j): opt_model.add_constraint(
        ct=u_var >= m[i, j] - d_vars[i, j],
        ctname="constraint_u_vars_2_{0}_{1}".format(i, j))
        for i in index_set for j in index_set})

    objective = opt_model.sum(u_var)

    opt_model.minimize(objective)

    if starting_point_distance_matrix is not None:
        print("setting starting point")
        warmstart = opt_model.new_solution()
        for i in range(n):
            for j in range(i + 1, n):
                warmstart.add_var_value(d_vars[i + 1, j + 1], starting_point_distance_matrix[i, j])
                warmstart.add_var_value(d_vars[j + 1, i + 1], starting_point_distance_matrix[j, i])
                if starting_point_distance_matrix[i, j] == 1:
                    warmstart.add_var_value(e_vars[j + 1, i + 1], 1)
                    warmstart.add_var_value(e_vars[i + 1, j + 1], 1)
        opt_model.add_mip_start(warmstart)

    print("start_solving")
    opt_model.solve()
    print(opt_model.solve_details)

    D = np.zeros((n, n))
    E = np.zeros((n, n))
    Z = np.zeros((n, n, n))
    for i in index_set:
        for j in index_set:
            D[i - 1, j - 1] = d_vars[i, j].sv
            E[i - 1, j - 1] = e_vars[i, j].sv
            for k in index_set:
                Z[i - 1, j - 1, k - 1] = z_vars[i, j, k].sv
    # opt_df = pd.DataFrame.from_dict(x_vars, orient="index",
    #                                 columns=["variable_object"])
    # opt_df.index = pd.MultiIndex.from_tuples(opt_df.index,
    #                                          names=["column_i", "column_j"])
    # opt_df.reset_index(inplace=True)
    # opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.solution_value)
    return D, E, Z, opt_model.solve_details.status
