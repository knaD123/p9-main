import mpls_classes
import mpls_fwd_gen
import argparse
import yaml
import re
import math
def main(conf):
    # Load topology
    G = mpls_fwd_gen.topology_from_aalwines_json(conf["topology"], visualize=False)
    print("*****************************")
    print(G.graph["name"])

    with open(conf["demands"],"r") as file:
        flows_with_load = [[x,y, int(z)] for [x,y,z] in yaml.load(file, Loader=yaml.BaseLoader)]
        flows_with_load = sorted(flows_with_load, key=lambda x: x[2], reverse=True)[
                          :math.ceil(len(flows_with_load) * conf["take_percent"])]
        flows = [flow[:2] for flow in flows_with_load]

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

    # Omnet
    name = re.search(r".*zoo_(.*)\.json", conf["topology"]).group(1).lower()
    network.flows_for_omnet = network.build_flow_table(flows_with_load)

    network.to_omnetpp(name=name, output_dir=f"./omnet_files/{name}")

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

    conf = vars(p.parse_args())

    with open(conf["conf"], "r") as f:
        conf.update(yaml.safe_load(f))

    main(conf)