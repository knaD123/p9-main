import json
import math
import os
import yaml
from ortools.linear_solver import pywraplp
import sys
from mpls_fwd_gen import *


def main(graphfile, demandsfile, print_flows=False, unsplittable_flow=True):
    with open(demandsfile, "r") as file:
        flows_with_load = [[x, y, int(z)] for [x, y, z] in yaml.load(file, Loader=yaml.BaseLoader)]

    # Sort the flows
    flows_with_load = sorted(flows_with_load, key=lambda x: x[2], reverse=True)[:math.ceil(len(flows_with_load))]

    with open(graphfile) as f:
        topo_data = json.load(f)

    # Load link capacities
    link_caps = {}
    for f in topo_data["network"]["links"]:
        src = f["from_router"]
        tgt = f["to_router"]
        if src != tgt:
            link_caps[(src,tgt)] = f.get("bandwidth", 0)
            if f["bidirectional"]:
                link_caps[(tgt, src)] = f.get("bandwidth", 0)

    graph = topology_from_aalwines_json(graphfile, visualize=False).to_directed()

    minimized_max_util = congestion_lp(graph, link_caps, flows_with_load)

    result_folder = "results/lp/" + graphfile.split("/")[1]
    os.makedirs(result_folder, exist_ok=True)
    result_file = os.path.join(result_folder, "default")

    results = dict()
    results["topology"] = graphfile
    results["demandfile"] = demandsfile
    results["minimal_maximal_utilization"] = minimized_max_util
    results["unsplittable_flows"] = unsplittable_flow

    with open(os.path.join(result_folder, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

def congestion_lp(graph, capacities, demands, print_flows=True, unsplittable_flow=True):  # Inputs networkx directed graph, dict of capacities, dict of demands
    def demand(i, d):
        if demands[d][0] == i:  # source
            return 1
        elif demands[d][1] == i:  # destination
            return -1
        else:
            return 0  # intermediate

    if unsplittable_flow:
        solver = pywraplp.Solver.CreateSolver('SCIP')
    else:
        solver = pywraplp.Solver.CreateSolver('GLOP')

    #take percent
    demands = sorted(demands, key=lambda x: x[2], reverse=True)[:math.ceil(len(demands) * 0.2)]

    # alpha is the minimal coefficient for capacities needed s.t. all demands do not exceed capacities
    alpha = solver.NumVar(0, solver.infinity(), "alpha")

    # Flow variables for solver
    if unsplittable_flow:
        f = {(i, j, d): solver.IntVar(0, 1, "{}->{}=>{}->{}".format(i, j, demands[d][0], demands[d][1])) for
             (i, j) in graph.edges for d in range(len(demands))}
    else:
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
        if print_flows:
            for d in range(len(demands)):
                for (i, j) in graph.edges:
                    if f[i,j,d].SolutionValue() > 0:
                        print(f[i,j,d].name() + ": " + str(f[i,j,d].SolutionValue()))
        print('Minimal maximal utilization =', solver.Objective().Value())
        return solver.Objective().Value()
    else:
        print(solver.Objective().Value())
        print('The problem does not have an optimal solution.')

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
