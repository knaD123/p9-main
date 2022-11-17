import os
import statistics
import json
import argparse
import yaml
from collections import defaultdict

alg_to_name = dict()
alg_to_name.update({f"inout-disjoint_max-mem={i}_path-heuristic=semi_disjoint_paths": f"FBR SD ({i})" for i in range(50)})
alg_to_name.update({f"inout-disjoint_max-mem={i}_path-heuristic=greedy_min_congestion": f"FBR GC ({i})" for i in range(50)})
alg_to_name.update({f"inout-disjoint_max-mem={i}_path-heuristic=shortest_path": f"FBR SP ({i})" for i in range(50)})
alg_to_name.update({f"tba-complex_max-mem={i}": f"TBA-C ({i})" for i in range(50)})

alg_to_name["rsvp-fn"] = "RSVP"
alg_to_name["gft"] = "GFT"

variable_to_label = dict()
variable_to_label["weighted_max_congestion"] = "Max Congestion"
variable_to_label["connectivity"] = "Connectivity"
variable_to_label["weighted_path_stretch"] = "Path Stretch"
variable_to_label["weighted_fortz_thorup_sum"] = "Weighted Ftz Thrp"

type_to_legend_pos = {
    "weighted_max_congestion": "north west",
    "connectivity": "south east",
    "weighted_path_stretch": "north west",
    "weighted_fortz_thorup_sum": "north west"
}

# Plot options generator
line_options = [
    "[mark=none, color=orange, loosely dashed, thick]",
    "[mark=none, color=magenta, solid, thick]",
    "[mark=none, color=red, solid, thick]",
    "[mark=none, color=gray, densely dashed, thick]",
    "[mark=none, color=green, dotted, thick]",
    "[mark=none, color=blue, dashed, thick]",
    "[mark=none, color=black, dash dot, thick]"
]

type_to_legend_style = {
    "connectivity": ['legend cell align=left'],
    "weighted_max_congestion": ['legend cell align=left'],
    "weighted_max_stretch": ['legend cell align=left'],
    "weighted_fortz_thorup_sum": ['legend cell align=left'],


}

axis_option = {
    "connectivity": r"ylabel={Weighted Average Connectivity}, legend pos= {south east}, legend style = {legend cell align=left}",
    "weighted_max_congestion": r"ylabel={Weighted Average Max Congestion}, legend pos= {south east}, legend style = {legend cell align=left}",
    "weighted_path_stretch": r"ylabel={Weighted Average Path Stretch}, legend pos= {south east}, legend style = {legend cell align=left}",
    "weighted_fortz_thorup_sum": r"ylabel={Weighted Average Fortz Thorup Sum}, legend pos= {south east}, legend style = {legend cell align=left}",
}

def default_packages():
    return ['inputenc', 'amsmath', 'amsfonts', 'amssymb', 'xcolor', 'tikz', 'pgfplots']

get_latex_packages = defaultdict(default_packages)


def add_data_points(data, variable, f):
    for alg, tops in sorted(data.items(), key=lambda x: x[0]):
        f.writelines([
            f"\\addplot{next(option_gen)} coordinates{{\n"
        ])
        tops_filtered = []
        for top in tops.values():
            if variable in top:
                tops_filtered.append(top)

        for index, top in enumerate(sorted(tops_filtered, key=lambda x: x[variable]), start=0):
            f.write(f"({index}, {top[variable]})\n")

        f.writelines([
            "};\n"
        ])

# Compute Fortz and Thorup score

def create_plot(data, variable, directory):
    with open(f"{directory}/{variable}.tex", "w") as f:
        alg_joined = ', '.join(alg_names)
        f.writelines([
            r"\documentclass[margin=10,varwidth]{standalone}\usepackage[utf8]{inputenc}\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{xcolor} \usepackage{tikz} \usepackage{pgfplots}",
            "\n",
            "\\begin{document}\n"
            "\\begin{tikzpicture}\n",
            f"\\begin{{axis}}[ylabel={{{variable_to_label[variable]}}}, legend pos= {{south east}}, legend style = {{legend cell align=left}}]\n",
            f"\\legend{{{alg_joined}}}\n"
        ])

        add_data_points(data, variable, f)

        f.writelines([
            "\\end{axis}\n"
            "\\end{tikzpicture}\n"
            "\\end{document}\n"
        ])

def plot(variable, data, args, topologies):
    # Process data

    #Trim runs that didn't finish for all algorithms
    if args.filter_unfinished_topologies:
        for topology in topologies:
            remove_topology = False
            for alg_data in data.values():
                if not topology in alg_data:
                    remove_topology = True

    latex_packages = get_latex_packages[variable]
    variable_name = variable_to_label[variable]
    legend_pos = type_to_legend_pos[variable]
    legend_style = type_to_legend_style[variable]
    legend_order = sorted([alg_to_name[alg] for alg in data.keys()])

    # Add all graph styling
    output = ""
    output += r"\documentclass[margin=10,varwidth]{standalone}"
    for package in latex_packages:
        output += rf"\usepackage{{{package}}}"
    output += r"\n"
    output += r"\begin{document}\n" + \
              r"\begin{tikzpicture}\n" + \
              r"\begin{axis}"
    output += rf"[{axis_option[variable]}]\n"
    output += rf"\legend{{{', '.join(legend_order)}}}\n"

    # Add each plot

    for alg, alg_data in data.items():
        output = add_data_to_output(alg, )
    print(output)
    pass

def data_subset(topologies, input_dir):
    data = {}
    for alg_dir in os.listdir(input_dir):
        alg_data = dict()
        for topology in [x for x in os.listdir(os.path.join(input_dir, alg_dir)) if x in topologies]:
            topo_file = os.path.join(input_dir, alg_dir, topology, "results.json")
            try:
                with open(topo_file, "r") as f:
                    alg_data[topology] = json.load(f)
            except FileNotFoundError:
                raise Exception(f"{topo_file} does not exist")
        data[alg_dir] = alg_data

    return data

if __name__ == "__main__":
    # Parse input and output directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--path_stretch", action="store_true")
    parser.add_argument("--max_congestion", action="store_true")
    parser.add_argument("--connectivity", action="store_true")
    parser.add_argument("--filter_unfinished_topologies", action="store_false", help="Filter a topology from the data if not all algorithms finished")

    parser.add_argument("--topology_info", default="topology_info.json", help="Json file describing the topologies")
    parser.add_argument("--num_failed_links", type=int, help='Results for failure scenarios with exactly <num_failed_links> failed links')
    parser.add_argument("--max_nodes", type=int, help='Results for networks with up to <max_nodes> nodes')
    parser.add_argument("--topologies", help='Provide the path to a list of names of topologies you want results for')
    args = parser.parse_args()

    # Load data
    data = {}
    input_dir = args.input_dir

    # Load info of topologies
    with open(args.topology_info, "r") as f:
        topology_info = json.load(f)

    # Only load from specific topologies if path provided. By default use all topologies
    if args.topologies:
        with open(args.topologies, "r") as f:
            topologies = yaml.load(f)
    else:
        topologies = topology_info.keys()
    data = data_subset(topologies, input_dir)

    # Filter topologies that are too large
    if args.max_nodes:
        pop_list = [topology[0] for topology in topology_info.items() if topology[1]["num_nodes"] > args.max_nodes]
        for topologies in data.values():
            for topology in pop_list:
                topologies.pop(topology, None)

    # Only compute for failure scenarios with k <args.num_failed_links> failed links
    if args.num_failed_links:
        for topologies in data.values():
            for topology in topologies.values():
                new_runs = []
                for run in topology["runs"]:
                    if run["failed_links#"] == args.num_failed_links:
                        new_runs.append(run)
                topology["runs"] = new_runs
                topology["failure_scenarios#"] = len(new_runs)

    line_options_gen = (x for x in line_options)

    #Generate plots
    if args.connectivity:
        plot("connectivity", data, args, topologies)


    # Generate aggregated data for each algorithm
    for alg, tops in results.items():
        for name, top in tops.items():
            # Run should have finished for all algorithms
            cont = False
            for algo in results.values():
                if not name in algo:
                    cont = True
                    break
            if cont:

                continue
            # Calculate congestion weighted by link probability
            normalization_sum = 0
            max_congestion = 0
            path_stretch = 0
            delivered_packet_rate = 0
            ftz_score = 0
            with open(top["topology"], "r") as t:
                total_links = len(json.load(t)["network"]["links"])

            for run in top["runs"]:
                failed_links = run["failed_links#"]
                probability = 0.001 ** failed_links * (1 - 0.001) ** (total_links - failed_links)
                normalization_sum += probability
                max_congestion += run["max_congestion"] * probability
                delivered_packet_rate += run["delivered_packet_rate"] * probability
                path_stretch += run["path_stretch"] * probability
                ftz_score += run["fortz_thorup_sum"] * probability

            top["weighted_max_congestion"] = max_congestion / normalization_sum
            top["weighted_delivered_packet_rate"] = delivered_packet_rate / normalization_sum
            top["weighted_path_stretch"] = path_stretch / normalization_sum
            top["weighted_fortz_thorup_sum"] = ftz_score / normalization_sum

        """max_congestion = 0.0
        delivered_packet_rate = 0.0
        path_stretch = 0.0
        top_len = len(tops)
        for top in tops.values():
            max_congestion += top["weighted_max_congestion"]
            delivered_packet_rate += top["weighted_delivered_packet_rate"]
            path_stretch += top["weighted_path_stretch"]

        results[alg]["average_max_congestion"] = max_congestion / top_len
        results[alg]["average_delivered_packet_rate"] = delivered_packet_rate / top_len
        results[alg]["average_path_stretch"] = path_stretch / top_len"""

    variables = [
        "weighted_max_congestion",
        "weighted_delivered_packet_rate",
        "weighted_path_stretch",
        "weighted_fortz_thorup_sum"]

    for v in variables:
        create_plot(results, v, args.output_dir)

