import mpls_classes
import mpls_fwd_gen
import argparse
import yaml
import re
import math
import json
def main(conf):
    with open(conf["demands"],"r") as file:
        flows_with_load = [[x,y, int(z)] for [x,y,z] in yaml.load(file, Loader=yaml.BaseLoader)]
    total_packets = num_packets(flows_with_load)
    flows_with_load = flows_take(sorted(flows_with_load, key=lambda x: x[2], reverse=True), take_percent=conf['take_percent'])
    flows = [flow[:2] for flow in flows_with_load]
    conf["loads"] = flows_with_load

    # Load link capacities
    with open(conf["topology"]) as f:
        topo_data = json.load(f)

    link_caps = {}
    for f in topo_data["network"]["links"]:
        src = f["from_router"]
        tgt = f["to_router"]
        if src != tgt:
            link_caps[(src, tgt)] = f.get("bandwidth", 0)
            if f["bidirectional"]:
                link_caps[(tgt, src)] = f.get("bandwidth", 0)

    conf["link_caps"] = link_caps

    print(f'Total number of packets over a second: {total_packets}')

    # Load topology
    G = mpls_fwd_gen.topology_from_aalwines_json(conf["topology"], visualize=False)
    print("*****************************")
    print(G.graph["name"])

    # Generate MPLS forwarding rules
    network = mpls_fwd_gen.generate_fwd_rules(G, conf,
                                 enable_PHP=conf["php"],
                                 numeric_labels=False,
                                 enable_LDP=conf["ldp"],
                                 enable_RMPLS=conf["enable_RMPLS"],
                                 num_lsps=flows,
                                 tunnels_per_pair=conf["rsvp_tunnels_per_pair"],
                                 enable_services=conf["vpn"],
                                 num_services=conf["vpn_num_services"],
                                 PE_s_per_service=conf["vpn_pes_per_services"],
                                 CEs_per_PE=conf["vpn_ces_per_pe"],
                                 random_seed=conf["random_seed"]
                                 )
    if conf["method_name"]:
        method = conf["method_name"]
    else:
        if conf["method"] == "rsvp":
            if conf["enable_RMPLS"]:
                method = "rmpls"
            else:
                method = "rsvp_fn"
        elif conf["method"] == "fbr":
            if conf["path_heuristic"] == "greedy_min_congestion":
                method = "fbr_gmc"
            elif conf["path_heuristic"] == "essence":
                method = "fbr_essence"
            else:
                method = "fbr"

        else:
            method = conf["method"]

    # Omnet
    network.flows_for_omnet = network.build_flow_table(flows_with_load)

    network.to_omnetpp(name=conf["topo_name"], output_dir=f"{conf['output_dir']}/{conf['topo_name']}/{method}", scaler=conf['scaler'], packet_size=conf["packet_size"], zero_latency=conf["zero_latency"], package_name=conf["package_name"], algorithm=method, latency_scaler=conf["latency_scaler"])

    # Add package.ned
    if conf["generate_package"]:
        with open(f"{conf['output_dir']}/package.ned", "w") as f:
            f.write(f"package {conf['package_name']};")
def num_packets(flows_with_load):
    sum = 0
    for (x,y,z) in flows_with_load:
        sum += z

    return sum / 64

def flows_take(flows_with_load, take_percent):
    load_sum = sum(flow[2] for flow in flows_with_load)
    target_sum = take_percent * load_sum
    current_sum = 0
    result_flows = []

    for flow in flows_with_load:
        current_sum += flow[2]
        if current_sum >= target_sum:
            break
        result_flows.append(flow)

    return result_flows

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conf", type=str, required=True)
    p.add_argument("--enable_RMPLS", action="store_true",
                   help="Use experimental RMPLS recursive protection (LFIB post processing). Defaults False")
    p.add_argument("--vpn", action="store_true", help="Enable MPLS VPN generic services. Defaults False. ")
    p.add_argument("--vpn_num_services", type=int, default=1,
                   help="Number of (random) MPLS VPN services, if enabled. Defaults to 1")
    p.add_argument("--vpn_pes_per_services", type=int, default=3,
                   help="Number of PE routers allocated to each MPLS VPN service, if enabled. Defaults to 3")
    p.add_argument("--vpn_ces_per_pe", type=int, default=1,
                   help="Number of CE to attach to each PE serving a VPN, if enabled. Defaults to 1")
    p.add_argument("--take_percent", type=float, default=1, help="What percentage of biggest flows to take")
    p.add_argument("--scaler", type=float, default=1, help="Multiplies the send interval by the scaler value and divides the link bandwidth by the same value")
    p.add_argument("--packet_size", type=int, default=64, help="Size in bytes")
    p.add_argument("--zero_latency", action="store_true", help="Set latency to 0 for all links")
    p.add_argument("--output_dir", default="./omnet_files")
    p.add_argument("--package_name", default="inet.zoo_topology")
    p.add_argument("--generate_package", action="store_true")
    p.add_argument("--topo_name", type=str, required=True)
    p.add_argument("--method_name", type=str, default="", help="Name of the algorithm that is used")
    p.add_argument("--latency_scaler", type=float, default=1)

    conf = vars(p.parse_args())

    with open(conf["conf"], "r") as f:
        conf.update(yaml.safe_load(f))

    main(conf)