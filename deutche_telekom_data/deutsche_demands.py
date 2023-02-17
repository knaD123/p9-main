import csv
from collections import defaultdict

with open("flows-no-date.csv", "r") as r:
    csv_reader = csv.reader(r)

    #skip headers
    next(csv_reader)

    # map demand to timestamp. Disregard demands with less than 1 packet
    time_stamp_to_demands = defaultdict(list)
    for line in csv_reader:
        time_stamp, src, tgt, load = line
        if int(float(load)) > 0 and src != tgt:
            time_stamp_to_demands[time_stamp].append([src,tgt,int(float(load))])

    for time_stamp, demands in time_stamp_to_demands.items():
        output = "["
        output += ", ".join(f"[{x[0]}, {x[1]}, {x[2]}]" for x in demands)
        output += "]"
        with open(f"deutsche_demand-{time_stamp}.yml", "w") as f:
            f.write(output)






