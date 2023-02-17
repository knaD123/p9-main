import csv
import json
from collections import defaultdict


dump_dict = {}

network_dict = {}
network_dict["name"] = "deutsche_telekom"
links_list = []
routers_list = []

with open ("links.csv", "r") as f:
    csv_reader = csv.reader(f)

    # Skip the column names
    next(csv_reader)
    raw_links = []
    for row in csv_reader:
        # load the links in a list
        raw_links.append(tuple(row))

def first_interface():
    return 0

router_to_interface = defaultdict(first_interface)

# Create the json formatted links
# Note that links are directed..
for link in raw_links:
    src, tgt, capacity = link
    # Check if equivalent link in opposite direction is already added to the links_dict
    # If this is the case, we can continue
    skip_link = False
    for added_link in links_list:
        if src == added_link["to_router"] and tgt == added_link["from_router"] and capacity == added_link["bandwidth"]:
            skip_link = True
    if skip_link:
        continue
    link_dict = {}
    link_dict["bandwidth"] = int(capacity)
    link_dict["from_router"] = src
    link_dict["to_router"] = tgt
    link_dict["from_interface"] = f"i{router_to_interface[src]}"
    link_dict["to_interface"] = f"i{router_to_interface[tgt]}"
    router_to_interface[src] += 1
    router_to_interface[tgt] += 1
    link_dict["latency"] = 0
    link_dict["weight"] = 0
    if (tgt, src, capacity) in raw_links:
        link_dict["bidirectional"] = True
    else:
        link_dict["bidirectional"] = False

    links_list.append(link_dict)

#Create json formatted routers
for link in raw_links:
    src, tgt, capacity = link
    for router in routers_list:
        if router["name"] == src:
            break
    else:
        new_router = {}
        new_router["name"] = src
        new_router["location"] = {"latitude": 0.0, "longitude": 0.0}
        new_router["interfaces"] = []
        i_dict = {}
        i_dict["names"] = []
        for i in range(router_to_interface[src]):
            i_dict["names"].append(f"i{i}")
        i_dict["routing_table"] = {}

        new_router["interfaces"].append(i_dict)
        routers_list.append(new_router)

    for router in routers_list:
        if router["name"] == tgt:
            break
    else:
        new_router = {}
        new_router["name"] = tgt
        new_router["location"] = {"latitude": 0.0, "longitude": 0.0}
        new_router["interfaces"] = []
        i_dict = {}
        i_dict["names"] = []
        for i in range(router_to_interface[tgt]):
            i_dict["names"].append(f"i{i}")
        i_dict["routing_table"] = {}

        new_router["interfaces"].append(i_dict)
        routers_list.append(new_router)

network_dict["links"] = links_list
network_dict["routers"] = routers_list

dump_dict["network"] = network_dict
with open("deutsche_telekom.json", "w") as f:
    json.dump(dump_dict, f, indent=2)


