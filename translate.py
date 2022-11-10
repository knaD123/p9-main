import json
import os
import re
import shutil

import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


if os.path.exists("demands"):
    shutil.rmtree("demands")

os.mkdir("demands")

for topo in os.listdir("zoo"):
    f = open("zoo/" + topo, "r")
    data = f.read()
    f.close()
    data_into_list = data.replace(" ", "").split("\n")
    filtered_list = list(dict.fromkeys(data_into_list))
    i = 0
    for line in filtered_list:
        if line.__contains__("],"):
            break
        i += 1
    filtered_list_sliced = filtered_list[2:i]
    templist = []
    for i in filtered_list_sliced:
        result = re.sub(r'[^_A-Za-z0-9]', '', i)
        result = result.split("_")
        # result = re.sub(r'[0-9]*[^A-Za-z0-9]', '', result)
        result = result[1:]
        result = '_'.join(result)
        templist.append(result)

    kvplaces = {}
    for idx in enumerate(templist):
        kvplaces.update({idx[0]: idx[1]})

    for demand in os.listdir("zoo_demands"):
        if demand.__contains__(os.path.splitext(topo)[0]):
            f = open("zoo_demands/" + demand, "r")
            file = f.readlines()
            f.close()
            finallst = []
            file = file[2:]

            for line in file:
                temptriple = []
                line = re.sub(r'[A-Za-z]*', '', line)
                line = re.sub(r'\n', '', line)
                line = re.split(" ", line)
                line = line[1:]
                line = [int(i) for i in line]

                temptriple.append(kvplaces.get(line[0]))
                temptriple.append(kvplaces.get(line[1]))
                temptriple.append(line[2])
                finallst.append(temptriple)

            demandsplit = demand.split(".")
            filename = demandsplit[0] + "_" + demandsplit[1] + ".yml"

            with open("demands/" + filename, "a") as file:
                yaml.dump(finallst, file, default_flow_style=True, Dumper=NoAliasDumper)
