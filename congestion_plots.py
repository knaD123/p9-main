import os
import re
import statistics
import subprocess

path = os.path.join(os.path.dirname(__file__), "results")
data = dict()

for alg in os.listdir(path):
    alg_dir = os.path.join(path, alg)
    alg_data = list()
    for topo in os.listdir(alg_dir):
        topo_file = os.path.join(alg_dir, topo, "0")
        with open(topo_file, "r") as t:
            lines = t.readlines()
            topo_data = [float(re.search("median_congestion:([0-9]+.[0-9]+) ", line)[1]) for line in lines]
            alg_data.append(statistics.mean(topo_data))
    alg_data.sort()
    data[alg] = alg_data


with open("congestion_plot.tex", "w") as f:
    def add_alg_data(data):
        f.writelines([
            "\\addplot coordinates{\n"
        ])

        for index, cong in enumerate(data, start=0):
            f.write(f"({index}, {cong})\n")

        f.writelines([
            "};\n"
        ])

    f.writelines([
        r"\documentclass[margin=10,varwidth]{standalone}\usepackage[utf8]{inputenc}\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{xcolor} \usepackage{tikz} \usepackage{pgfplots}", "\n",
        "\\usepackage{tikz}\n",
        "\\usepackage{pgfplots}\n",
        "\\begin{document}\n"
        "\\begin{tikzpicture}\n",
        "\\begin{axis}[ylabel={Congestion}]\n"
        "\\legend{FBR3, RSVP-FN}\n"
    ])

    add_alg_data(data["inout-disjoint-full_max-mem=3"])
    add_alg_data(data["rsvp-fn"])


    f.writelines([
        "\\end{axis}\n"
        "\\end{tikzpicture}\n"
        "\\end{document}\n"
    ])

