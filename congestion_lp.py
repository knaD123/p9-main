from ortools.linear_solver import pywraplp


def congestion_lp(graph, capacities, demands):  # Inputs networkx directed graph, dict of capacities, dict of demands
    def demand(i, d):
        if demands[d][0] == i:  # source
            return 1
        elif demands[d][1] == i:  # destination
            return -1
        else:
            return 0  # intermediate

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return

    # alpha is the minimal coefficient for capacities needed s.t. all demands do not exceed capacities
    alpha = solver.NumVar(0, solver.infinity(), "alpha")

    # Flow variables for solver
    f = {(i, j, d): solver.NumVar(0, 1, "{}->{}=>{}->{}".format(i, j, demands[d][0], demands[d][1])) for
         (i, j) in graph.edges for d in range(len(demands))}

    # Flow conservation constraints: total flow balance at node i for each demand d
    # must be 0 if i is an intermediate node, 1 if i is the source of demand d, and
    # -1 if i is the destination.
    for i in graph.nodes:
        for d in range(len(demands)):
            solver.Add(sum(f[i, j, d] for j in graph.nodes if (i, j) in graph.edges) -
                       sum(f[j, i, d] for j in graph.nodes if (i, j) in graph.edges) ==
                       demand(i, d))

    # Capacity constraints: weighted sum of flow variables must be contained in the
    # total capacity installed on the arc (i, j)
    for (i, j) in graph.edges:
        solver.Add((sum(demands[d][2] * f[i, j, d] for d in range(len(demands)))) <=
                   capacities[i, j] * alpha)

    # Minimizing alpha is equivalent with minimizing utility of link with maximal utility
    solver.Minimize(alpha)
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Minimal maximal utilization =', solver.Objective().Value())
        return solver.Objective().Value()
    else:
        print(solver.Objective().Value())
        print('The problem does not have an optimal solution.')
