from random import shuffle
from os.path import isdir, join
import os
from os import listdir
import json
import argparse
import yaml
from collections import defaultdict

alg_to_name = dict()
alg_to_name.update({f"inout-disjoint_max-mem={i}_path-heuristic=semi_disjoint_paths": f"FBR({i}) SD" for i in range(50)})
alg_to_name.update({f"inout-disjoint_max-mem={i}_path-heuristic=greedy_min_congestion": f"FBR({i}) GC" for i in range(50)})
alg_to_name.update({f"inout-disjoint_max-mem={i}_path-heuristic=shortest_path": f"FBR({i}) SP" for i in range(50)})
alg_to_name.update({f"inout-disjoint_max-mem={i}_path-heuristic=nielsens_heuristic": f"FBR({i}) Nielsens" for i in range(50)})
alg_to_name.update({f"inout-disjoint-old_max-mem={i}": f"FBR({i}) OLD" for i in range(50)})

population = [100, 200]
crossover = [0.9,0.95]
mutation = [0.7,0.8]
generations = [100,200]


for i in range(10):
    for j in population:
        for k in generations:
            for c in crossover:
                for m in mutation:
                    alg_to_name.update({f"inout-disjoint_max-mem={i}_path-heuristic=essence_p={j}_c={c}_m={m}_g={k}": f"FBR({i}) p:{j} g:{k} c:{c} m:{m} essence"})

for i in range(10):
    for j in population:
        for k in generations:
            for c in crossover:
                for m in mutation:
                    alg_to_name[f"inout-disjoint_max-mem={i}_path-heuristic=essence_p={j}_c={c}_m={m}_g={k}"] = f"FBR({i}) p:{j} g:{k} c:{c} m:{m} essence"

for i in range(100):
    alg_to_name[f"inout-disjoint_max-mem={i}_path-heuristic=semi_disjoint_paths"] = f"FBR({i}) SD"
    alg_to_name[f"inout-disjoint_max-mem={i}_path-heuristic=greedy_min_congestion"] = f"FBR({i}) GC"
    alg_to_name[f"inout-disjoint_max-mem={i}_path-heuristic=shortest_path"] = f"FBR({i}) SP"
    alg_to_name[f"inout-disjoint_max-mem={i}_path-heuristic=nielsens_heuristic"] = f"FBR({i}) Nielsens"
    alg_to_name[f"inout-disjoint-old_max-mem={i}"] = f"FBR({i}) OLD"
    alg_to_name[f"inout-disjoint_max-mem={i}_path-heuristic=nielsens_heuristic_max_s=10000"] = f"FBR({i}) Nielsens"




for i in range(50):
    alg_to_name.update({f"inout-disjoint_max-mem={i}_path-heuristic=benjamins_heuristic{j}": f"FBR({i}) Benj({j})" for j in range(50)})
alg_to_name.update({f"tba-complex_max-mem={i}": f"TBA-C ({i})" for i in range(50)})

alg_to_name["rsvp-fn"] = "RSVP"
alg_to_name["gft"] = "GFT"
alg_to_name["tba-simple"] = "TBA-S"
alg_to_name["rmpls"] = "RMPLS"


variable_to_label = dict()
variable_to_label["max_congestion"] = "Weighted Max Single Link Utilization"
variable_to_label["max_congestion_normalized"] = "Weighted Max Single Link Utilization*"
variable_to_label["delivered_packet_rate"] = "Weighted Connectivity"
variable_to_label["path_stretch"] = "Weighted Path Stretch"
variable_to_label["fortz_thorup_sum"] = "Weighted Ftz Thrp"
variable_to_label["clean_packets"] = "Clean Packet Ratio"
variable_to_label["util_poly_score"] = "Polynomic congestion u*u"
variable_to_label["util_exp_score_2"] = "Exponential congestion 2^u"
variable_to_label["util_exp_score_4"] = "Exponential congestion 4^u"




# Plot options generator
line_options = [
    "mark=none, color=orange, loosely dashed, thick",
    "mark=none, color=orange, densely dashed, thick",
    "mark=none, color=orange, solid, thick",
    "mark=none, color=orange, dash dot, thick",
    "mark=none, color=orange, dotted, thick",
    "mark=none, color=magenta, loosely dashed, thick",
    "mark=none, color=magenta, densely dashed, thick",
    "mark=none, color=magenta, solid, thick",
    "mark=none, color=magenta, dash dot, thick",
    "mark=none, color=magenta, dotted, thick",
    "mark=none, color=red, loosely dashed, thick",
    "mark=none, color=red, densely dashed, thick",
    "mark=none, color=red, solid, thick",
    "mark=none, color=red, dash dot, thick",
    "mark=none, color=red, dotted, thick",
    "mark=none, color=gray, loosely dashed, thick",
    "mark=none, color=gray, densely dashed, thick",
    "mark=none, color=gray, solid, thick",
    "mark=none, color=gray, dash dot, thick",
    "mark=none, color=gray, dotted, thick",
    "mark=none, color=green, loosely dashed, thick",
    "mark=none, color=green, densely dashed, thick",
    "mark=none, color=green, solid, thick",
    "mark=none, color=green, dash dot, thick",
    "mark=none, color=green, dotted, thick",
    "mark=none, color=blue, loosely dashed, thick",
    "mark=none, color=blue, densely dashed, thick",
    "mark=none, color=blue, solid, thick",
    "mark=none, color=blue, dashed, thick",
    "mark=none, color=blue, dotted, thick",
    "mark=none, color=black, loosely dashed, thick",
    "mark=none, color=black, densely dashed, thick",
    "mark=none, color=black, solid, thick",
    "mark=none, color=black, dash dot, thick",
    "mark=none, color=black, dotted, thick",
    "mark=none, color=yellow, loosely dashed, thick",
    "mark=none, color=yellow, densely dashed, thick",
    "mark=none, color=yellow, solid, thick",
    "mark=none, color=yellow, dash dot, thick",
    "mark=none, color=yellow, dotted, thick",
    "mark=none, color=purple, loosely dashed, thick"
    "mark=none, color=purple, densely dashed, thick"
    "mark=none, color=purple, solid, thick"
    "mark=none, color=purple, dash dot, thick"
    "mark=none, color=purple, dotted, thick"
    "mark=none, color=orange, loosely dashed, thick",
    "mark=none, color=orange, densely dashed, thick",
    "mark=none, color=orange, solid, thick",
    "mark=none, color=orange, dash dot, thick",
    "mark=none, color=orange, dotted, thick",
    "mark=none, color=magenta, loosely dashed, thick",
    "mark=none, color=magenta, densely dashed, thick",
    "mark=none, color=magenta, solid, thick",
    "mark=none, color=magenta, dash dot, thick",
    "mark=none, color=magenta, dotted, thick",
    "mark=none, color=red, loosely dashed, thick",
    "mark=none, color=red, densely dashed, thick",
    "mark=none, color=red, solid, thick",
    "mark=none, color=red, dash dot, thick",
    "mark=none, color=red, dotted, thick",
    "mark=none, color=gray, loosely dashed, thick",
    "mark=none, color=gray, densely dashed, thick",
    "mark=none, color=gray, solid, thick",
    "mark=none, color=gray, dash dot, thick",
    "mark=none, color=gray, dotted, thick",
    "mark=none, color=green, loosely dashed, thick",
    "mark=none, color=green, densely dashed, thick",
    "mark=none, color=green, solid, thick",
    "mark=none, color=green, dash dot, thick",
    "mark=none, color=green, dotted, thick",
    "mark=none, color=blue, loosely dashed, thick",
    "mark=none, color=blue, densely dashed, thick",
    "mark=none, color=blue, solid, thick",
    "mark=none, color=blue, dashed, thick",
    "mark=none, color=blue, dotted, thick",
    "mark=none, color=black, loosely dashed, thick",
    "mark=none, color=black, densely dashed, thick",
    "mark=none, color=black, solid, thick",
    "mark=none, color=black, dash dot, thick",
    "mark=none, color=black, dotted, thick",
    "mark=none, color=yellow, loosely dashed, thick",
    "mark=none, color=yellow, densely dashed, thick",
    "mark=none, color=yellow, solid, thick",
    "mark=none, color=yellow, dash dot, thick",
    "mark=none, color=yellow, dotted, thick",
    "mark=none, color=purple, loosely dashed, thick"
    "mark=none, color=purple, densely dashed, thick"
    "mark=none, color=purple, solid, thick"
    "mark=none, color=purple, dash dot, thick"
    "mark=none, color=purple, dotted, thick"
    "mark=none, color=orange, loosely dashed, thick",
    "mark=none, color=orange, densely dashed, thick",
    "mark=none, color=orange, solid, thick",
    "mark=none, color=orange, dash dot, thick",
    "mark=none, color=orange, dotted, thick",
    "mark=none, color=magenta, loosely dashed, thick",
    "mark=none, color=magenta, densely dashed, thick",
    "mark=none, color=magenta, solid, thick",
    "mark=none, color=magenta, dash dot, thick",
    "mark=none, color=magenta, dotted, thick",
    "mark=none, color=red, loosely dashed, thick",
    "mark=none, color=red, densely dashed, thick",
    "mark=none, color=red, solid, thick",
    "mark=none, color=red, dash dot, thick",
    "mark=none, color=red, dotted, thick",
    "mark=none, color=gray, loosely dashed, thick",
    "mark=none, color=gray, densely dashed, thick",
    "mark=none, color=gray, solid, thick",
    "mark=none, color=gray, dash dot, thick",
    "mark=none, color=gray, dotted, thick",
    "mark=none, color=green, loosely dashed, thick",
    "mark=none, color=green, densely dashed, thick",
    "mark=none, color=green, solid, thick",
    "mark=none, color=green, dash dot, thick",
    "mark=none, color=green, dotted, thick",
    "mark=none, color=blue, loosely dashed, thick",
    "mark=none, color=blue, densely dashed, thick",
    "mark=none, color=blue, solid, thick",
    "mark=none, color=blue, dashed, thick",
    "mark=none, color=blue, dotted, thick",
    "mark=none, color=black, loosely dashed, thick",
    "mark=none, color=black, densely dashed, thick",
    "mark=none, color=black, solid, thick",
    "mark=none, color=black, dash dot, thick",
    "mark=none, color=black, dotted, thick",
    "mark=none, color=yellow, loosely dashed, thick",
    "mark=none, color=yellow, densely dashed, thick",
    "mark=none, color=yellow, solid, thick",
    "mark=none, color=yellow, dash dot, thick",
    "mark=none, color=yellow, dotted, thick",
    "mark=none, color=purple, loosely dashed, thick"
    "mark=none, color=purple, densely dashed, thick"
    "mark=none, color=purple, solid, thick"
    "mark=none, color=purple, dash dot, thick"
    "mark=none, color=purple, dotted, thick"
]

axis_option = {
    "delivered_packet_rate": r"ylabel={Weighted Average Connectivity}, legend pos= {south east}, legend style = {legend cell align=left}",
    "max_congestion": r"ylabel={Weighted Average Max Utilization}, legend pos= {north west}, legend style = {legend cell align=left}",
    "max_congestion_normalized": r"ylabel={Weighted Average Max Utilization*}, legend pos= {north west}, legend style = {legend cell align=left}",
    "path_stretch": r"ylabel={Weighted Average Path Stretch}, legend pos= {north west}, legend style = {legend cell align=left}",
    "fortz_thorup_sum": r"ylabel={Weighted Average Fortz Thorup Sum}, legend pos= {north west}, legend style = {legend cell align=left}",
    "clean_packets": r"ylabel={Clean Packet Ratio}, legend pos= {north west}, legend style = {legend cell align=left}",
    "util_poly_score": r"ylabel={Polynomic congestion u*u}, legend pos= {north west}, legend style = {legend cell align=left}",
    "util_exp_score_2": r"ylabel={Exponential congestion 2^u}, legend pos= {north west}, legend style = {legend cell align=left}",
    "util_exp_score_4": r"ylabel={Exponential congestion 4^u}, legend pos= {north west}, legend style = {legend cell align=left}",

}

def default_packages():
    return ['inputenc', 'amsmath', 'amsfonts', 'amssymb', 'xcolor', 'tikz', 'pgfplots']

get_latex_packages = defaultdict(default_packages)


def tex_string(variable, data_points, args, topologies):

    latex_packages = get_latex_packages[variable]
    legend_order = sorted([alg_to_name.get(alg, "null") for alg in data_points.keys()])

    # Add all graph styling
    output = ""
    output += r"\documentclass[margin=10,varwidth]{standalone}"
    for package in latex_packages:
        output += rf"\usepackage{{{package}}}"
    output += r"\begin{document}" + \
              r"\begin{tikzpicture}" + \
              r"\begin{axis}"
    output += rf"[{axis_option[variable]}]"
    output += rf"\legend{{{', '.join(legend_order)}}}"

    # Add each line
    shuffle(line_options)
    line_options_gen = (x for x in line_options)
    alphabetical_order = sorted(data_points.keys(), key=lambda x: alg_to_name.get(x, "null"))
    for alg in alphabetical_order:
        output += rf"\addplot[{next(line_options_gen)}] coordinates{{" + \
                  r"".join(data_points[alg]) + \
                  r"};"

    output += r"\end{axis}" + \
              r"\end{tikzpicture}" + \
              r"\end{document}"

    return output

def scenario_probability(num_failed_links, num_edges, fp=0.001):
    if num_failed_links < num_edges:
        return (fp ** num_failed_links) * ((1 - fp) ** (num_edges - num_failed_links))
    elif num_failed_links == num_edges:
        return (fp ** num_failed_links)
    else:
        raise Exception(f"Error: There was {num_failed_links} failed links, but only {num_edges} in the network")

def compute_normalization_sum(runs, num_edges):
    sum = 0
    for run in runs:
        sum += scenario_probability(run["failed_links#"], num_edges)
    return sum


def generate_data_points(variable, data, topology_info):
    # Map an algorithm to a set of data points

    alg_to_data_points = {}
    for alg, topologies in data.items():
        connectivity_unsorted = []
        for topology, topology_data in topologies.items():
            connectivity = 0
            norm_sum = 0
            for run in topology_data["runs"]:
                _scenario_probability = scenario_probability(run["failed_links#"], topology_info[topology]["num_edges"])
                connectivity += (run[variable] * _scenario_probability)
                norm_sum += _scenario_probability
            connectivity /= norm_sum
            connectivity_unsorted.append(connectivity)
        for i in range(len(connectivity_unsorted)):
            print(f"{i}: {connectivity_unsorted[i]} = {list(topologies)[i]}")
        data_points = [f"({i}, {con})" for i, con in enumerate(connectivity_unsorted)]
        alg_to_data_points[alg] = data_points

    return alg_to_data_points

def max_congestion_normalized_data(data, topology_info):
    # Map an algorithm to a set of data points
    # First compute normalization sum for each topology
    alg_to_topo_to_norm_sum = {}
    for alg, topologies in data.items():
        topo_to_norm_sum = {}
        for topology, topology_data in topologies.items():
            topo_to_norm_sum[topology] = compute_normalization_sum(topology_data["runs"], topology_info[topology]["num_edges"])
        alg_to_topo_to_norm_sum[alg] = topo_to_norm_sum

    alg_to_data_points = {}
    for alg, topologies in data.items():
        values_unsorted = []
        for topology, topology_data in topologies.items():
            value = 0
            for run in topology_data["runs"]:
                if run["delivered_packet_rate"] > 0:
                    value += ((run["max_congestion"] / run["delivered_packet_rate"]) * scenario_probability(run["failed_links#"], topology_info[topology]["num_edges"])) / alg_to_topo_to_norm_sum[alg][topology]
                else:
                    value += (10 * scenario_probability(run["failed_links#"], topology_info[topology]["num_edges"])) / alg_to_topo_to_norm_sum[alg][topology]

            values_unsorted.append(value)
        for i in range(len(values_unsorted)):
            print(f"{i}: {values_unsorted[i]} = {list(topologies)[i]}")
        data_points = [f"({i}, {con})" for i, con in enumerate(values_unsorted)]
        alg_to_data_points[alg] = data_points

    return alg_to_data_points

def data_subset(topologies, input_dir, algorithms):
    data = {}
    # Jargon line below simply says to only use subset of algorithms, unless "all" is specified
    for alg_dir in ([alg for alg in os.listdir(input_dir) if alg in algorithms] if algorithms != "all" else [x for x in listdir(input_dir) if isdir(join(input_dir, x))]):
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
    parser.add_argument("--max_congestion_normalized", action="store_true", help="normalized by connectivity")
    parser.add_argument("--delivered_packet_rate", action="store_true")
    parser.add_argument("--fortz_thorup_sum", action="store_true")
    parser.add_argument("--clean_packets", action="store_true")
    parser.add_argument("--util_poly_score", action="store_true")
    parser.add_argument("--util_exp_score_2", action="store_true")
    parser.add_argument("--util_exp_score_4", action="store_true")

    parser.add_argument("--dont_filter_unfinished_topologies", action="store_true", help="Don't Filter a topology from the data if not all algorithms finished")
    parser.add_argument("--topology_info", default="topology_info.json", help="Json file describing the topologies")
    parser.add_argument("--num_failed_links", type=int, help='Results for failure scenarios with exactly <num_failed_links> failed links')
    parser.add_argument("--max_nodes", type=int, help='Results for networks with up to <max_nodes> nodes')
    parser.add_argument("--topologies", help='Provide the path to a list of names of topologies you want results for')
    parser.add_argument("--algorithms", type=str, default="all", help='Subset of algorithms to compute results for. Specify algorithms by the name of its result directory, separate names by comma. If left empty compare all algorithms in directory.')

    args = parser.parse_args()

    input_dir = args.input_dir

    # Remove empty folders
    for d in [x for x in os.listdir(input_dir) if x in alg_to_name]:
        alg_dir = os.path.join(input_dir, d)
        for topo_dir in os.listdir(alg_dir):
            full_path = os.path.join(alg_dir, topo_dir)
            if len(os.listdir(full_path)) == 0:
                os.rmdir(full_path)

    # Load info of topologies
    with open(args.topology_info, "r") as f:
        topology_info = json.load(f)


    # Only load from specific topologies if path provided. By default use all topologies
    if args.topologies:
        with open(args.topologies, "r") as f:
            topologies = yaml.load(f)
    else:
        topologies = topology_info.keys()

    # Load data
    algorithms = [x for x in args.algorithms.split(", ")]
    data = data_subset(topologies, input_dir, args.algorithms)

    # Filter topologies that are too large
    if args.max_nodes:
        pop_list = [topology[0] for topology in topology_info.items() if topology[1]["num_nodes"] > args.max_nodes]
        for topologies in data.values():
            for topology in pop_list:
                topologies.pop(topology, None)

    # Only compute for failure scenarios with <args.num_failed_links> failed links
    if args.num_failed_links:
        for topologies in data.values():
            for topology in topologies.values():
                new_runs = []
                for run in topology["runs"]:
                    if run["failed_links#"] == args.num_failed_links:
                        new_runs.append(run)
                topology["runs"] = new_runs
                topology["failure_scenarios#"] = len(new_runs)

    # Filter a toplogy if simulation failed for at least one algorithm
    if not args.dont_filter_unfinished_topologies:
        filter_topologies = []
        for topology_to_check in topology_info:
            for topologies in data.values():
                if not topology_to_check in topologies:
                    filter_topologies.append(topology_to_check)
                    break
        for topology in filter_topologies:
            for alg in data.keys():
                data[alg].pop(topology, None)

    #Generate plots
    if args.delivered_packet_rate:
        data_points = generate_data_points("delivered_packet_rate", data, topology_info)
        _tex_string = tex_string("delivered_packet_rate", data_points, args, topologies)
        with open(os.path.join(args.output_dir, "delivered_packet_rate.tex"), "w") as f:
            f.write(_tex_string)
    if args.max_congestion:
        data_points = generate_data_points("max_congestion", data, topology_info)
        _tex_string = tex_string("max_congestion", data_points, args, topologies)
        with open(os.path.join(args.output_dir, "max_congestion.tex"), "w") as f:
            f.write(_tex_string)
    if args.max_congestion_normalized:
        data_points = max_congestion_normalized_data(data, topology_info)
        _tex_string = tex_string("max_congestion_normalized", data_points, args, topologies)
        with open(os.path.join(args.output_dir, "max_congestion_normalized.tex"), "w") as f:
            f.write(_tex_string)
    if args.path_stretch:
        data_points = generate_data_points("path_stretch", data, topology_info)
        _tex_string = tex_string("path_stretch", data_points, args, topologies)
        with open(os.path.join(args.output_dir, "path_stretch.tex"), "w") as f:
            f.write(_tex_string)
    if args.fortz_thorup_sum:
        data_points = generate_data_points("fortz_thorup_sum", data, topology_info)
        _tex_string = tex_string("fortz_thorup_sum", data_points, args, topologies)
        with open(os.path.join(args.output_dir, "fortz_thorup_sum.tex"), "w") as f:
            f.write(_tex_string)
    if args.clean_packets:
        data_points = generate_data_points("clean_packets_ratio", data, topology_info)
        _tex_string = tex_string("clean_packets", data_points, args, topologies)
        with open(os.path.join(args.output_dir, "clean_packets.tex"), "w") as f:
            f.write(_tex_string)
    if args.util_poly_score:
        data_points = generate_data_points("util_poly_score", data, topology_info)
        _tex_string = tex_string("util_poly_score", data_points, args, topologies)
        with open(os.path.join(args.output_dir, "util_poly_score.tex"), "w") as f:
            f.write(_tex_string)
    if args.util_exp_score_2:
        data_points = generate_data_points("util_exp_score_2", data, topology_info)
        _tex_string = tex_string("util_exp_score_2", data_points, args, topologies)
        with open(os.path.join(args.output_dir, "util_exp_score_2.tex"), "w") as f:
            f.write(_tex_string)
    if args.util_exp_score_4:
        data_points = generate_data_points("util_exp_score_4", data, topology_info)
        _tex_string = tex_string("util_exp_score_4", data_points, args, topologies)
        with open(os.path.join(args.output_dir, "util_exp_score_4.tex"), "w") as f:
            f.write(_tex_string)



