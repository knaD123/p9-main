from typing import Dict, Tuple
import random

import networkx as nx
from matplotlib import pyplot as plt

import mpls_classes
import mpls_fwd_gen
import argparse
import yaml
import re
import math
import json


# To be looped
def communicator(network):

    with open("demand1.json", "r") as f:
        demands = json.load(f)
    import_demands(demands)



    x = 1



def import_demands(demands: Dict[str, Dict[str, float]]):
    for src in demands.keys():
        for tgt in demands[src]:
            demands[src][tgt] = sendinterval_to_load(demands[src][tgt])

    return demands

def sendinterval_to_load(send_interval):
    return int(64 / send_interval)

def get_router_rules(router, path):
    return router.get_routing_entries_for_path(path)