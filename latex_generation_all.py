#!/usr/bin/env python3
import datetime
import os
import argparse
from get_results import parse_result_data, TopologyResult, FailureScenarioData, compute_connectedness
import overleaf
import time
from datetime import datetime
import re
import math
import functools

from typing import Dict, Tuple, List, Callable

class AlgorithmPlotConfiguration:
    def __init__(self, name: str, color: str, line_style: str, mark: str = "none"):
        self.name: str = name
        self.color: str = color
        self.line_style: str = line_style
        self.mark: str = mark


alg_to_plot_config_dict: {str: AlgorithmPlotConfiguration} = {
    # "cfor-disjoint": AlgorithmPlotConfiguration("Continue Forwarding", "black", "dashed"),
    "tba-simple": AlgorithmPlotConfiguration("B-CA", "black", "dash dot", "+"),
    "tba-complex": AlgorithmPlotConfiguration("E-CA", "blue", "dashed", "diamond*"),
    "rsvp-fn": AlgorithmPlotConfiguration("RSVP-FN", "dgreen", "dotted", "triangle*"),
    # "hd": AlgorithmPlotConfiguration("Hop Distance", "green", "loosely dotted"),
    "kf": AlgorithmPlotConfiguration("KF", "cyan", "densely dotted"),
    "gft": AlgorithmPlotConfiguration("GFT-CA", "orange", "loosely dashed", "x"),
    "inout-disjoint": AlgorithmPlotConfiguration("FBR", "red", "solid", "square*"),
    "inout-disjoint-full": AlgorithmPlotConfiguration('FBR-full', 'magenta', 'solid', 'circle*'),
    "rmpls": AlgorithmPlotConfiguration("R-MPLS", "gray", "densely dashed", "*"),
    "plinko4": AlgorithmPlotConfiguration('Plinko', 'purple', 'loosely dotted')
}

alg_to_bar_config_dict = {
    "tba-simple": "dots",
    "inout-disjoint": "horizontal lines",
    "tba-complex": "north east lines",
    "inout-disjoint-full": 'vertical lines'
}

parser = argparse.ArgumentParser()
parser.add_argument("--max_points", type=int, required=False)
parser.add_argument("--auto_overleaf", action='store_true', required=False)
args = parser.parse_args()

results_folder = os.path.join(os.path.dirname(__file__), "results")
print("Parsing results from results folder")
results_data = parse_result_data(results_folder)

max_points = 1000000
if args.max_points is not None:
    max_points = args.max_points

overleaf_upload = False
if args.auto_overleaf is not None:
    overleaf_upload = args.auto_overleaf


class OutputData:
    def __init__(self, filename: str, content: str, content_type: str):
        self.file_name: str = filename
        self.content: str = content
        self.content_type: str = content_type


def init_connectedness_table_output(data):
    return "\\begin{tabular}{c |" + " c" * len(data.keys()) + "}\n\t" + "$|F|$ & " + " & ".join(
        data.keys()) + "\\\\ \hline \n"


def add_failure_line_connectedness_table(filtered_data, len_f):
    return f"\t {len_f} & " + " & ".join(
        ["{:.6f}".format(sum(c.connectedness for c in filtered_data[algo]) / len(filtered_data[algo])) for algo in
         filtered_data]) + "\\\\ \n"


def compute_average_connectedness_for_algorithm(data, alg) -> float:
    return sum(c.connectedness for c in data[alg]) / len(data[alg])

def compute_average_connectivity_for_algorithm(data, alg) -> float:
    return sum(top.connectivity for top in data[alg]) / len(data[alg])

def end_connectedness_table_output():
    return "\end{tabular}"


def generate_all_latex():
    start_time = time.time()
    if overleaf_upload:
        overleaf.synchronize()

    latex_dir = os.path.join(os.path.dirname(__file__), f"latex")
    if not os.path.exists(latex_dir) or not os.path.isdir(latex_dir):
        os.mkdir(latex_dir)

    max_memories = [3, 5, 8]
    max_memories_filtered_data = dict({alg: results_data[alg] for alg in results_data if (
                "_max-mem=" in alg and int(alg.split("_max-mem=")[1]) in max_memories) or "_max-mem=" not in alg})
    max_memories_filtered_data2 = {}
    for k, v in max_memories_filtered_data.items():
        if "_max-mem=" in k:
            key = k.split("_max-mem=")[0] + "-" + k.split("_max-mem=")[1]
            max_memories_filtered_data2[key] = v
        else:
            max_memories_filtered_data2[k] = v
    connectedness_table_output = init_connectedness_table_output(max_memories_filtered_data2)

    # generate latex code for connectedness plot for each failure scenario cardinality
    print("Creating connectedness table with each failure scenario cardinality")
    for len_f in range(0, 5):
        filtered_data = remove_failure_scenarios_that_are_not_of_correct_failure_cardinality(max_memories_filtered_data, len_f)

        compute_connectedness(filtered_data)

        connectedness_table_output += add_failure_line_connectedness_table(filtered_data, len_f)

    connectedness_table_output += end_connectedness_table_output()
    output_latex_content("connectedness_table.tex", connectedness_table_output, "connectedness table")
    output_latex_content("connectedness_plot_data.tex", latex_connectedness_plot(results_data, max_points),
                         "connectedness plot")

    output_latex_content("memory_plot_data.tex", latex_memory_plot(results_data, max_points), "memory plot")
    no_keep_forwarding_data = dict({alg: results_data[alg] for alg in results_data.keys() if alg != 'kf' or alg != 'plinko4'})
    output_latex_content("memory_plot_data_no-kf.tex", latex_memory_plot(no_keep_forwarding_data, max_points), "memory plot without keep forwarding")
    output_latex_content("memory_failure_data.tex", latex_memory_failure_rate_plot(results_data), "memory bar chart")
    output_latex_content("loop_table_data.tex", latex_loop_table(results_data), "loop table")
    # output_latex_content('scatter_tba_vs_inout_data.tex', latex_scatter_plot(results_data, 'tba-complex_max-mem=5', 'inout-disjoint_max-mem=5'), 'scatter plot')
    # output_latex_content("latency_average_max_data.tex", latex_average_max_latency_plot(results_data), "average max number of hops plot (latency)")
    # output_latex_content("latency_average_mean_data.tex", latex_average_mean_latency__plot(results_data), "average mean number of hops plot (latency)")
    output_latex_content("fwd_gen_time_data.tex", latex_gen_time_plot(results_data), "forwarding table generation time")
    output_latex_content('latency_full_median.tex', latex_full_latency_plot(results_data), 'median failure scenario latency')


    if overleaf_upload:
        overleaf.push()
    print(f"Time taken to generate latex: {time.time() - start_time}")


def output_latex_content(file_name: str, content: str, content_type: str):
    content = "% timestamp:" + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n" + content
    print(f"Writing {content_type} to file: 'latex/{file_name}'")
    latex_file_table = open(os.path.join(os.path.dirname(__file__), f"latex/{file_name}"), "w")
    latex_file_table.write(content)
    if overleaf_upload:
        try:
            overleaf.set_file_text(content, f"figures/results_auto_generated/{file_name}")
        except:
            print(f"ERROR: Failed uploading {content_type} at 'figures/results_auto_generated/{file_name}'")


def latex_average_max_latency_plot(data: Dict[str, List[TopologyResult]]) -> str:
    latex_plot_legend = r"\legend{"
    algs = set()
    skip_algs = set()
    skip_algs.add("kf")
    skip_algs.add("plinko4")
    memories = ["4"]

    for alg in data.keys():
        if alg in skip_algs:
            continue

        if "max-mem" in alg:
            (alg_proper_name, memory_group) = re.split("_max-mem=", alg)
            if memory_group not in memories:
                continue
            latex_plot_legend += f"{alg_to_plot_config_dict[alg_proper_name].name + memory_group}, "
            algs.add(alg)
        else:
            algs.add(alg)
            latex_plot_legend += f"{alg_to_plot_config_dict[alg].name}, "
    latex_plot_legend += "}\n"

    full_connected_failure_scenarios: set[(TopologyResult, int)] = set()
    for topology in list(data.items())[0][1]:
        topology: TopologyResult
        for i in range(len(topology.failure_scenarios)):
            full_connected_failure_scenarios.add((topology.topology_name, i))

    for (alg, topologies) in data.items():
        for topology in topologies:
            topology: TopologyResult
            for i, fs in enumerate(topology.failure_scenarios):
                fs: FailureScenarioData
                if fs.successful_flows != fs.connected_flows:
                    full_connected_failure_scenarios.discard((topology.topology_name, i))

    filtered_data = {alg: data[alg] for alg in algs}

    latex_plot_data = ""
    for (alg, topologies) in filtered_data.items():
        alg: str
        topologies: list[TopologyResult]
        if alg not in algs:
            continue

        latex_plot_data += r"\addplot[mark=none" + \
                           ", color=" + alg_to_plot_config_dict[re.split("_max-mem=", alg)[0]].color + \
                           ", " + alg_to_plot_config_dict[re.split("_max-mem=", alg)[0]].line_style + \
                           ", thick] coordinates{" + "\n"

        average_hops_max: list[(TopologyResult, float)] = \
            [(top, sum([failure_scenario.hops_max for failure_scenario in top.failure_scenarios if (top.topology_name, top.failure_scenarios.index(failure_scenario)) in full_connected_failure_scenarios], 0) / len(list(filter(lambda x: x[0] == top.topology_name, full_connected_failure_scenarios))), full_connected_failure_scenarios) for top in topologies]

        cactus_data = sorted(average_hops_max, key=lambda x: x[1])

        latex_plot_data += ''.join(map(lambda data: f"({filtered_data[0]}, {filtered_data[1][1]}) % {data[1][0].topology_name}\n", list(enumerate(cactus_data, 1)))) + "};\n"

    return latex_plot_legend + latex_plot_data

def latex_full_latency_plot(data: Dict[str, List[TopologyResult]]) -> str:
    latex_plot_legend = r"\legend{"
    algs = list()
    skip_algs = set()
    skip_algs.add("kf")
    skip_algs.add("plinko4")
    memories = ["4"]
    inf_hops = 100

    for alg in sorted(data.keys()):
        if alg in skip_algs:
            continue

        if "max-mem" in alg:
            (alg_proper_name, memory_group) = re.split("_max-mem=", alg)
            if memory_group not in memories:
                continue
            latex_plot_legend += f"{alg_to_plot_config_dict[alg_proper_name].name + memory_group}, "
            algs.append(alg)
        else:
            algs.append(alg)
            latex_plot_legend += f"{alg_to_plot_config_dict[alg].name}, "
    latex_plot_legend += "}\n"

    filtered_data = {alg: data[alg] for alg in algs}

    latex_plot_data = ""
    for (alg, topologies) in sorted(filtered_data.items(), key=lambda x: x[0]):
        alg: str
        topologies: list[TopologyResult]
        if alg not in algs:
            continue

        latex_plot_data += r"\addplot[mark=none" + \
                           ", color=" + alg_to_plot_config_dict[re.split("_max-mem=", alg)[0]].color + \
                           ", " + alg_to_plot_config_dict[re.split("_max-mem=", alg)[0]].line_style + \
                           ", thick] coordinates{" + "\n"

        points = []
        from get_results import inf
        from statistics import median_low as median  # Use median_low to avoid avg over number and infinity
        for t in topologies:
            med_hops = [median(fs.hops[i] for fs in t.failure_scenarios if fs.hops[i] != -1) for i in range(t.num_flows)]
            med_hops = map(lambda x: inf_hops if x == inf else x, med_hops)
            for i, mh in enumerate(med_hops):
                points.append((t.topology_name + str(i), mh))
        points.sort(key=lambda x: x[1])

        latex_plot_data += '\n'.join(map(lambda p: f'({p[0]}, {p[1][1]}) % {p[1][0]}', enumerate(points))) + "\n};\n"

    return latex_plot_legend + latex_plot_data


def latex_average_mean_latency__plot(data: Dict[str, List[TopologyResult]]) -> str:
    latex_plot_legend = r"\legend{"
    algs = set()
    skip_algs = set()
    skip_algs.add("kf")
    skip_algs.add("plinko4")
    memories = ["4"]

    for alg in data.keys():
        if alg in skip_algs:
            continue

        if "max-mem" in alg:
            (alg_proper_name, memory_group) = re.split("_max-mem=", alg)
            if memory_group not in memories:
                continue
            latex_plot_legend += f"{alg_to_plot_config_dict[alg_proper_name].name + memory_group}, "
            algs.add(alg)
        else:
            algs.add(alg)
            latex_plot_legend += f"{alg_to_plot_config_dict[alg].name}, "
    latex_plot_legend += "}\n"

    full_connected_failure_scenarios: set[(TopologyResult, int)] = set()
    for topology in list(data.items())[0][1]:
        topology: TopologyResult
        for i in range(len(topology.failure_scenarios)):
            full_connected_failure_scenarios.add((topology.topology_name, i))

    for (alg, topologies) in data.items():
        if alg in skip_algs:
            continue
        for topology in topologies:
            topology: TopologyResult
            for i, fs in enumerate(topology.failure_scenarios):
                fs: FailureScenarioData
                if fs.successful_flows != fs.connected_flows or fs.connected_flows == 0:
                    full_connected_failure_scenarios.discard((topology.topology_name, i))

    filtered_data = {alg: data[alg] for alg in algs}

    latex_plot_data = ""
    for (alg, topologies) in filtered_data.items():
        alg: str
        topologies: list[TopologyResult]
        if alg not in algs:
            continue

        latex_plot_data += r"\addplot[mark=none" + \
                           ", color=" + alg_to_plot_config_dict[re.split("_max-mem=", alg)[0]].color + \
                           ", " + alg_to_plot_config_dict[re.split("_max-mem=", alg)[0]].line_style + \
                           ", thick] coordinates{" + "\n"

        average_hops_mean: list[(TopologyResult, float)] = \
            [(top,
              sum(failure_scenario.hops_mean for j, failure_scenario in enumerate(top.failure_scenarios) if (top.topology_name, j) in full_connected_failure_scenarios)
                  / len(list(filter(lambda x: x[0] == top.topology_name, full_connected_failure_scenarios)))) for top in topologies]

        cactus_data = sorted(average_hops_mean, key=lambda x: x[1])

        latex_plot_data += ''.join(map(lambda data: f"({data[0]}, {data[1][1]}) % {data[1][0].topology_name}\n", list(enumerate(cactus_data, 1)))) + "};\n"

    return latex_plot_legend + latex_plot_data


def latex_gen_time_plot(data: Dict[str, List[TopologyResult]]) -> str:
    latex_plot_legend = r"\legend{"
    algs = list()
    skip_algs = set()
    skip_algs.add("kf")
    skip_algs.add("plinko4")
    memories = ["4"]

    for alg in sorted(data.keys()):
        if alg in skip_algs:
            continue

        if "max-mem" in alg:
            (alg_proper_name, memory_group) = re.split("_max-mem=", alg)
            if memory_group not in memories:
                continue
            latex_plot_legend += f"{alg_to_plot_config_dict[alg_proper_name].name + memory_group}, "
            algs.append(alg)
        else:
            algs.append(alg)
            latex_plot_legend += f"{alg_to_plot_config_dict[alg].name}, "
    latex_plot_legend += "}\n"

    latex_plot_data = ""

    for (alg, topologies) in sorted(data.items(), key=lambda x: x[0]):
        if alg not in algs:
            continue

        cactus_data = sorted(topologies, key=lambda topology: topology.fwd_gen_time / topology.num_flows)

        latex_plot_data += r"\addplot[mark=none" + \
                           ", color=" + alg_to_plot_config_dict[re.split("_max-mem=", alg)[0]].color + \
                           ", " + alg_to_plot_config_dict[re.split("_max-mem=", alg)[0]].line_style + \
                           ", thick] coordinates{" + "\n"

        counter = 0
        for i in range(0, len(cactus_data)):
            latex_plot_data += f"({counter}, {(cactus_data[i].fwd_gen_time / cactus_data[i].num_flows) / 1000000000}) %{cactus_data[i].topology_name}\n"
            counter += 1
        latex_plot_data += r"};" + "\n"

    return latex_plot_legend + latex_plot_data


def latex_memory_failure_rate_plot(data: Dict[str, List[TopologyResult]]) -> str:
    latex_plot_legend = r"\legend{"
    skip_algs = set()
    skip_algs.add("Keep Forwarding")
    skip_algs.add("Plinko-4")
    alg_longname_to_proper_alg_name = {}
    algs = set()
    mem_plot_cap = 25
    memories = [i for i in range(2,mem_plot_cap+1)]
    memory_to_alg_dict = {i: [] for i in range(2, mem_plot_cap+1)}

    for alg in sorted(data.keys()):
        alg: str
        if "max-mem" in alg:
            (alg_proper_name, memory_group) = re.split("_max-mem=", alg)
            alg_longname_to_proper_alg_name[alg] = alg_proper_name
            memory_to_alg_dict[int(memory_group)].append(alg)
            algs.add(alg_proper_name)
        else:
            alg_longname_to_proper_alg_name[alg] = alg
            max_memory_topology = max(data[alg], key=lambda it: it.max_memory / it.num_flows)
            max_memory = int(math.ceil((max_memory_topology.max_memory / max_memory_topology.num_flows)))
            filtered_memories = list(filter(lambda it: it >= max_memory, memories))
            for memory in filtered_memories:
                memory_to_alg_dict[memory].append(alg)
            algs.add(alg)

    alg_to_coordinates = {}
    for alg in algs:
        alg_to_coordinates[
            alg] = r"\addplot[" + f"color={alg_to_plot_config_dict[alg].color}, {alg_to_plot_config_dict[alg].line_style}, thick" + r", every mark/.append style={solid}] coordinates{"
        latex_plot_legend += f"{alg_to_plot_config_dict[alg].name}, "

    latex_plot_legend += "}\n"

    sorted_dict = sorted(list(memory_to_alg_dict.items()), key=lambda x: int(x[0]))
    for (memory_group, algs) in sorted_dict:
        for alg_longname in algs:
            connectedness = compute_average_connectedness_for_algorithm(data, alg_longname)
            failedness = 1.0 - connectedness
            #failedness = 1.0 - compute_average_connectivity_for_algorithm(data, alg_longname)

            alg_to_coordinates[alg_longname_to_proper_alg_name[alg_longname]] += f"({memory_group}, {failedness}) "

    for alg in alg_to_coordinates.keys():
        alg_to_coordinates[alg] += "};\n"

    return latex_plot_legend + ''.join(map(lambda x: x[1], sorted(alg_to_coordinates.items(), key=lambda x: x[0])))

def latex_scatter_plot(data: Dict[str, List[TopologyResult]], alg1: str, alg2: str) -> str:
    def get_connectedness(r: List[TopologyResult]):
        return map(lambda tr: tr.connectedness, r)

    datapoints = zip(get_connectedness(data[alg1]), get_connectedness(data[alg2]))

    return ''.join(map(lambda dp: str(dp), datapoints))


def remove_failure_scenarios_that_are_not_of_correct_failure_cardinality(data: {str: TopologyResult}, lenf: int) -> {
    str: TopologyResult}:
    filtered_data = {}
    for (conf, topologies) in data.items():
        conf: str
        filtered_data[conf] = []
        for topology in topologies:
            topology: TopologyResult
            failure_scenarios = list(filter(lambda scenario: scenario.failed_links == lenf, topology.failure_scenarios))
            filtered_data[conf].append(
                TopologyResult(topology.topology_name, topology.total_links, topology.num_flows, failure_scenarios, -1,
                               topology.fwd_gen_time, topology.max_memory, topology.within_memory_limit, -1))

    return filtered_data


def latex_connectedness_plot(data: dict, _max_points) -> str:
    latex_plot_legend = r"\legend{"
    skip_algs = set()
    for alg in data.keys():
        if not alg_to_plot_config_dict.keys().__contains__(alg):
            skip_algs.add(alg)
            continue
        latex_plot_legend += f"{alg_to_plot_config_dict[alg].name}, "
    latex_plot_legend += "}\n"

    latex_plot_data = ""
    for (alg, topologies) in data.items():
        alg: str
        if skip_algs.__contains__(alg):
            continue

        cactus_data = sorted(topologies, key=lambda topology: topology.connectedness)

        skip_number = len(cactus_data) / _max_points
        if skip_number < 1:
            skip_number = 1

        latex_plot_data += r"\addplot[mark=none" + \
                           ", color=" + alg_to_plot_config_dict[alg].color + \
                           ", " + alg_to_plot_config_dict[alg].line_style + \
                           ", thick] coordinates{" + "\n"

        counter = 0
        for i in range(0, len(cactus_data), int(skip_number)):
            if counter > _max_points:
                break
            latex_plot_data += f"({counter}, {cactus_data[i].connectedness}) %{cactus_data[i].topology_name}\n"
            counter += 1
        latex_plot_data += r"};" + "\n"

    return latex_plot_legend + latex_plot_data


def latex_loop_table(data) -> str:
    alg_to_res_dict = {}

    for alg in data.keys():
        num_links = 0
        num_looping_links = 0
        for topology in data[alg]:
            topology: TopologyResult
            num_links += topology.total_links

            for failure_scenario in topology.failure_scenarios:
                failure_scenario: FailureScenarioData
                num_looping_links += failure_scenario.looping_links

        alg_to_res_dict[alg] = num_looping_links

    latex_tabular_header = r"\begin{tabular}{c |"
    latex_algs = r"     "
    latex_numbers = r"    loop ratio "

    for (alg, alg_num) in alg_to_res_dict.items():
        latex_tabular_header += " c"
        latex_algs += f"& {alg} "
        latex_numbers += f"& {alg_num}"

    latex_algs += r"\\\hline" + "\n"
    latex_tabular_header += "}\n"
    latex_numbers += r"\\" + "\n"
    latex_tabular_end = r"\end{tabular}"

    return latex_tabular_header + latex_algs + latex_numbers + latex_tabular_end


def latex_memory_plot(data, _max_points) -> str:
    latex_plot_legend = r"\legend{"
    skip_algs = set()
    for alg in data.keys():
        if not alg_to_plot_config_dict.keys().__contains__(alg):
            skip_algs.add(alg)
            continue
        latex_plot_legend += f"{alg_to_plot_config_dict[alg].name}, "
    latex_plot_legend += "}\n"

    latex_plot_data = ""
    for (alg, topologies) in data.items():
        if skip_algs.__contains__(alg):
            continue
        cactus_data = sorted(topologies, key=lambda topology: topology.max_memory / topology.num_flows)

        skip_number = len(cactus_data) / _max_points
        if skip_number < 1:
            skip_number = 1

        latex_plot_data += r"\addplot[mark=none" + \
                           ", color=" + alg_to_plot_config_dict[alg].color + \
                           ", " + alg_to_plot_config_dict[alg].line_style + \
                           ", thick] coordinates{" + "\n"

        counter = 0
        for i in range(0, len(cactus_data), int(skip_number)):
            if counter > _max_points:
                break
            latex_plot_data += f"({counter}, {cactus_data[i].max_memory / cactus_data[i].num_flows}) %{cactus_data[i].topology_name}\n"
            counter += 1
        latex_plot_data += r"};" + "\n"

    return latex_plot_legend + latex_plot_data


generate_all_latex()
