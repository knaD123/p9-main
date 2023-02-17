
Network topology and flows of a large european ISP (internet service provider).

Router names are anonymized to random four-digit numbers starting from 1000.
Link capacities (bytes per timeunit) and flow traffic (bytes per timeunit, averaged over an hour) are scaled to fit between 0 and 1000000 (for anonymization).
Flow data is aggregated over an hour. The file contains data for 24 hours of one day.

The files are gzip compressed CSV files. 
(Hint: on Linux you can use 'less' to inspect .gz files)

Example of data import in Python:
    import pandas as pd
    import networkx as nx
    links = pd.read_csv('links.csv.gz', compression='gzip')
    flows = pd.read_csv('flows.csv.gz', compression='gzip', parse_dates=['timestamp'])
    graph = nx.from_pandas_edgelist(links, source='linkStart', target='linkEnd', edge_attr='capacity')

