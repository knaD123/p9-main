import os
from functools import reduce

from tqdm import tqdm
from ast import literal_eval
from typing import Dict, Tuple, List, Callable

inf = 1_000_000

def compute_probability(f, e, pf=0.001):
    return (pf ** f) * (1 - pf) ** (e - f)

class FailureScenarioData:
    def __init__(self, failed_links, looping_links, successful_flows, connected_flows, hops):
        self.failed_links = failed_links
        self.looping_links = looping_links
        self.successful_flows = successful_flows
        self.connected_flows = connected_flows
        self.hops: List[int] = hops
        self.probability = -1.0
        self.normalised_probability = -1.0


class CommonResultData:
    def __init__(self, total_links, num_flows, fwd_gen_time, max_memory):
        self.total_links = total_links
        self.num_flows = num_flows
        self.fwd_gen_time = fwd_gen_time
        self.max_memory = max_memory


class TopologyResult:
    def __init__(self, topology_name, total_links, num_flows, failure_scenarios, connectedness, fwd_gen_time, max_memory, within_memory_limit, connectivity):
        self.topology_name = topology_name
        self.total_links = total_links
        self.num_flows = num_flows
        self.failure_scenarios: List[FailureScenarioData] = failure_scenarios
        self.connectedness = connectedness
        self.connectivity = connectivity
        self.fwd_gen_time = fwd_gen_time
        self.max_memory = max_memory
        self.within_memory_limit = within_memory_limit

        # self.hops: Dict[int, int] = {}
        # for fs in self.failure_scenarios:
        #     for hop in fs.hops:
        #         self.hops[hop] = self.hops.get(hop, 0) + 1

        self.total_prob_mass = 0.0
        for fs in self.failure_scenarios:
            fs.probability = compute_probability(fs.failed_links, self.total_links)
            self.total_prob_mass += fs.probability

        for fs in self.failure_scenarios:
            fs.normalised_probability = fs.probability / self.total_prob_mass


def __parse_line_in_common(line: str):
    parts = line.split(' ')
    for part in parts:
        prop_name, value = part.split(':')

        if (prop_name == 'len(E)'):
            total_links = int(value)
            continue
        if (prop_name == 'num_flows'):
            num_flows = int(value)
            continue
        if (prop_name == 'fwd_gen_time'):
            fwd_gen_time = int(value)
            continue
        if (prop_name == 'memory'):
            max_memory = int(max(literal_eval(value)))
            continue

    return CommonResultData(total_links, num_flows, fwd_gen_time, max_memory)

def __parse_single_line_in_failure_scenario(line: str):
    # remove spaces in memory list
    # line = line.replace(", ", ",")

    parts = line.split(' ')
    for part in parts:
        prop_name, value = part.split(':')

        if (prop_name == 'len(F)'): #No match/case in this version :(
            failed_links = int(value)
            continue
        if (prop_name == 'looping_links'):
            looping_links = int(value)
            continue
        if (prop_name == 'successful_flows'):
            successful_flows = int(value)
            continue
        if (prop_name == 'connected_flows'):
            connected_flows = int(value)
            continue
        if (prop_name == 'hops'):
            hops = [int(n) for n in value[1:-2].split(',')]
            continue

    return FailureScenarioData(failed_links, looping_links, successful_flows, connected_flows, hops)


def parse_result_data(result_folder) -> Dict[str, List[TopologyResult]]:
    result_dict: dict[str, list[TopologyResult]] = {}
    conf_progress = 1
    for conf_name in os.listdir(result_folder):
        print(f"\nParsing results from algorithm {conf_name} - {conf_progress}/{len(os.listdir(result_folder))}")
        conf_progress += 1
        result_dict[conf_name] = []
        for topology in tqdm(os.listdir(f"{result_folder}/{conf_name}")):
            failure_scenarios = []
            total_links, num_flows, fwd_gen_time, max_memory = 0, 0, 0, 0
            res_dir = f"{result_folder}/{conf_name}/{topology}"
            for failure_chunk_file in os.listdir(res_dir):
                with open(f"{res_dir}/{failure_chunk_file}", "r") as t:
                    lines = t.readlines()

                    if str(failure_chunk_file) == "common":
                        common_data = __parse_line_in_common(lines[0])
                        total_links = common_data.total_links
                        num_flows = common_data.num_flows
                        fwd_gen_time = common_data.fwd_gen_time
                        max_memory = common_data.max_memory
                    else:
                        for line in lines:
                            failure_data = __parse_single_line_in_failure_scenario(line)
                            failure_scenarios.append(failure_data)
                    within_memory_limit = True
                    if conf_name.__contains__("max-mem="):
                        memory_cap = int(conf_name.split("max-mem=")[1])
                        within_memory_limit = max_memory <= num_flows * memory_cap

            result_dict[conf_name].append(TopologyResult(topology, total_links, num_flows, failure_scenarios, -1, fwd_gen_time, max_memory, within_memory_limit, -1))

    compute_connectedness(result_dict)

    check_within_memory_limit(result_dict)

    return result_dict

def check_within_memory_limit(result_data: dict):
    for alg, topology_results in result_data.items():
        for topology in topology_results:
            if not topology.within_memory_limit:
                print(f"get_results: USING TOO MUCH MEMORY on topology {topology.topology_name} - Flaw in algorithm {alg}!")


def compute_connectedness(result_data: dict) -> {}:
    for conf_name in result_data.keys():
        conf_name: str
        for topology in result_data[conf_name]:
            topology: TopologyResult
            connectedness = 0
            tot_successful = 0
            tot_connected = 0
            for failure_scenario in topology.failure_scenarios:
                failure_scenario: FailureScenarioData

                tot_connected += failure_scenario.connected_flows
                tot_successful += failure_scenario.successful_flows

                p = failure_scenario.normalised_probability

                if failure_scenario.connected_flows != 0:
                    connectedness += p * (failure_scenario.successful_flows / failure_scenario.connected_flows)
                else:
                    connectedness += p

            if(tot_connected == 0):
                topology.connectivity = 1
            else:
                topology.connectivity = tot_successful / tot_connected

            if len(topology.failure_scenarios) == 0:
                connectedness = 1.2
                # this should never happen
                # raise Exception("Topology had connectivity of 0.. very likely bug")
                print(f"{conf_name} No failure scenarios for {topology.topology_name}")

            topology.connectedness = connectedness

if __name__ == '__main__':
    data = parse_result_data('results')

    print("hello")