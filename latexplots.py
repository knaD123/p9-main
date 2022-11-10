import os
import statistics
import json

alg_to_name = dict()
alg_to_name.update({f"inout-disjoint_max-mem={i}_path-heuristic=semi_disjoint_paths": f"FBR SD ({i})" for i in range(50)})
alg_to_name.update({f"inout-disjoint_max-mem={i}_path-heuristic=greedy_min_congestion": f"FBR GC ({i})" for i in range(50)})
alg_to_name.update({f"inout-disjoint_max-mem={i}_path-heuristic=shortest_path": f"FBR SP ({i})" for i in range(50)})
alg_to_name["rsvp-fn"] = "RSVP"

variable_to_label = dict()
variable_to_label["weighted_max_congestion"] = "Max Congestion"
variable_to_label["weighted_delivered_packet_rate"] = "Connectivity"
variable_to_label["weighted_path_stretch"] = "Path Stretch"

def add_data_points(data, variable, f):
    for alg, tops in sorted(data.items(), key=lambda x: x[0]):
        f.writelines([
            f"\\addplot{next(option_gen)} coordinates{{\n"
        ])

        for index, top in enumerate(sorted(tops.values(), key=lambda x: x[variable]), start=0):
            f.write(f"({index}, {top[variable]})\n")

        f.writelines([
            "};\n"
        ])

# Compute Fortz and Thorup score

def create_plot(data, variable):
    with open(f"latex/{variable}.tex", "w") as f:
        alg_joined = ', '.join(alg_names)
        f.writelines([
            r"\documentclass[margin=10,varwidth]{standalone}\usepackage[utf8]{inputenc}\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{xcolor} \usepackage{tikz} \usepackage{pgfplots}",
            "\n",
            "\\usepackage{tikz}\n",
            "\\usepackage{pgfplots}\n",
            "\\begin{document}\n"
            "\\begin{tikzpicture}\n",
            f"\\begin{{axis}}[ylabel={{{variable_to_label[variable]}}}, legend pos= {{north west}}, legend style = {{legend cell align=left}}]\n",
            f"\\legend{{{alg_joined}}}\n"
        ])

        add_data_points(data, variable, f)

        f.writelines([
            "\\end{axis}\n"
            "\\end{tikzpicture}\n"
            "\\end{document}\n"
        ])

if __name__ == "__main__":
    results = dict()
    res_dir = "results/"
    for alg_dir in os.listdir(res_dir):
        alg_res = dict()
        for topo in os.listdir(os.path.join(res_dir, alg_dir)):
            topo_file = os.path.join(res_dir, alg_dir, topo, "results.json")
            with open(topo_file, "r") as f:
                alg_res[topo] = json.load(f)
        results[alg_dir] = alg_res

    alg_names = [alg_to_name[key] for key in sorted(results.keys())]

    # Plot options generator
    options_list = [
        "[mark=none, color=orange, loosely dashed, thick]",
        "[mark=none, color=magenta, solid, thick]",
        "[mark=none, color=red, solid, thick]",
        "[mark=none, color=gray, densely dashed, thick]",
        "[mark=none, color=dgreen, dotted, thick]",
        "[mark=none, color=blue, dashed, thick]",
        "[mark=none, color=black, dash dot, thick]"
    ]

    # Generate aggregated data for each algorithm
    for alg, tops in results.items():
        for top in tops.values():
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
        option_gen = (x for x in options_list)
        create_plot(results, v)

