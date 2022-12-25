from os import listdir
from os.path import isdir, join
import json

def scenario_probability(num_failed_links, num_edges, fp=0.001):
    if num_failed_links < num_edges:
        return (fp ** num_failed_links) * ((1 - fp) ** (num_edges - num_failed_links))
    elif num_failed_links == num_edges:
        return (num_failed_links ** fp)
    else:
        raise Exception(f"Error: There was {num_failed_links} failed links, but only {num_edges} in the network")

def compute_all_data_points(data, num_edges):
    ft_gen_time = data["ft_gen_time"]
    path_stretch = 0
    max_congestion = 0
    connectedness = 0
    util_poly_score = 0
    util_exp_score_2 = 0
    util_exp_score_4 = 0
    clean_packets = 0
    normalization_sum = 0
    for run in data["runs"]:
        _scenario_probability = scenario_probability(run["failed_links#"], num_edges)
        normalization_sum += _scenario_probability

        path_stretch += run["path_stretch"] * _scenario_probability
        max_congestion += run["max_congestion"] * _scenario_probability
        connectedness += run["delivered_packet_rate"] * _scenario_probability
        util_poly_score += run["util_poly_score"] * _scenario_probability
        util_exp_score_2 += run["util_exp_score_2"] * _scenario_probability
        util_exp_score_4 += run["util_exp_score_4"] * _scenario_probability
        clean_packets += run["clean_packets_ratio"] * _scenario_probability
    
    path_stretch /= normalization_sum
    max_congestion /= normalization_sum
    connectedness /= normalization_sum
    util_poly_score /= normalization_sum
    util_exp_score_2 /= normalization_sum
    util_exp_score_4 /= normalization_sum
    clean_packets /= normalization_sum
    return path_stretch, max_congestion, connectedness, util_poly_score, util_exp_score_2, util_exp_score_4, clean_packets, ft_gen_time


with open("topology_info.json", "r") as f:
    all_topologies = json.load(f)

res_dir = "test_results"

experiments = [x for x in listdir(res_dir) if isdir(join(res_dir, x))]

unfinished_topologies = []
for topology in all_topologies.keys():
    break_topo = False
    for experiment in experiments:
        for alg in [x for x in listdir(join(res_dir, experiment)) if isdir(join(res_dir, experiment, x))]:
            if not topology in listdir(join(res_dir, experiment, alg)):
                unfinished_topologies.append(topology)
                break_topo = True
                break
        if break_topo:
            break

finished_topologies = [x for x in all_topologies if x not in unfinished_topologies]

## Compare values


all_data_points = {}
for topology in finished_topologies:
    for experiment in experiments:
        for algo in [x for x in listdir(join(res_dir, experiment)) if isdir(join(res_dir, experiment, x))]:
            with open(join(res_dir, experiment, algo, topology, "results.json")) as f:
                data = json.load(f)

            path_stretch, max_congestion, connectedness, util_poly_score, util_exp_score_2, util_exp_score_4, clean_packets, gen_time = compute_all_data_points(data, all_topologies[topology]["num_edges"])

