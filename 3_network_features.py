#coding:utf-8
import os
import networkx as nx
import pandas as pd
from networkx.algorithms import bipartite
from tqdm import tqdm
import json

# technical nets are unweighted
def cal_tech_net(path):
    # check if file does not exist or empty
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return {'t_num_dev_nodes':0,\
                't_num_file_nodes':0,\
                't_num_dev_per_file':0,\
                't_num_file_per_dev':0,\
                't_graph_density':0,\
                't_dev_nodes': set()}

    bipartite_G = nx.Graph()
    df = pd.read_csv(path, header=None, sep='##', engine='python')
    df.columns = ['file', 'dev', 'weight']

    ## Logic to add nodes and edges to graph with their metadata
    for _, row in df.iterrows():
        dev_node = row['dev']
        file_node = row['file'].replace('   (with props)', '')
        bipartite_G.add_node(dev_node, bipartite='dev')
        bipartite_G.add_node(file_node, bipartite='file')
        bipartite_G.add_edge(dev_node, file_node)

    dev_nodes = {n for n, d in bipartite_G.nodes(data=True) if d["bipartite"] == 'dev'}
    file_nodes = {n for n, d in bipartite_G.nodes(data=True) if d["bipartite"] == 'file'}

    graph_density = bipartite.density(bipartite_G, dev_nodes)
    file_degrees, dev_degrees = bipartite.degrees(bipartite_G, dev_nodes)

    num_file_nodes = len(file_degrees)
    num_dev_nodes = len(dev_degrees)
    file_node_degree = sum([degree for node, degree in file_degrees])/len(file_degrees)
    dev_node_degree = sum([degree for node, degree in dev_degrees])/len(dev_degrees)

    # return the features of tech net
    return {'t_num_dev_nodes':num_dev_nodes,\
            't_num_file_nodes':num_file_nodes,\
            't_num_dev_per_file':file_node_degree,\
            't_num_file_per_dev':dev_node_degree,\
            't_graph_density':graph_density,\
            't_dev_nodes': set(dev_nodes)}

# social nets are weighted
def cal_social_net(path):
    # if no network data
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return {'s_num_nodes':0, \
                's_dev_nodes':set(),\
                's_weighted_mean_degree':0,\
                's_num_component':0,\
                's_avg_clustering_coef':0,\
                's_largest_component':0,\
                's_graph_density':0}

    # Processing features in social networks
    G = nx.read_edgelist(path, create_using=nx.DiGraph(), nodetype=str, comments='*', delimiter='##', data=(('weight', int),))
    # all dev nodes
    dev_nodes = set(G.nodes)
    # num. of total nodes
    num_nodes = len(dev_nodes)
    # weighted mean degree
    degrees = G.degree(weight='weight')
    weighted_mean_degree = sum([degree for node, degree in degrees])/num_nodes
    # average clustering coefficient
    avg_clustering_coef = nx.average_clustering(G)
    # betweenness = nx.betweenness_centrality(G, weight='weight')
    graph_density = nx.density(G)

    G = nx.read_edgelist(path, create_using=nx.Graph(), nodetype=str, comments='*', delimiter='##', data=(('weight', int),))
    # num. of dis-connected components
    num_component = nx.number_connected_components(G)
    # largest connected component
    largest_component = len(max(nx.connected_components(G), key=len))
    # num. of nodes in each component
    # num_nodes_component = [list(c) for c in list(nx.connected_components(G))]

    # return the features of the 
    return {'s_num_nodes': num_nodes,\
            's_dev_nodes': dev_nodes,\
            's_weighted_mean_degree':weighted_mean_degree,\
            's_num_component':num_component,\
            's_avg_clustering_coef':avg_clustering_coef,\
            's_largest_component':largest_component,\
            's_graph_density':graph_density}


def get_net_overlap(net1, net2):
	net1_set = set()
	with open(net1, 'r') as f:
		lines = f.read().splitlines()
	# print([net1, net2])
	for line in lines:
		sender, recivier, weight = line.split('##')
		net1_set.add((sender, recivier))
	net2_set = set()
	with open(net2, 'r') as f:
		lines = f.read().splitlines()
	for line in lines:
		sender, recivier, weight = line.split('##')		
		net2_set.add((sender, recivier))
	if len(net1_set) == len(net2_set) == 0:
		return 0
	intersection_edges = net1_set.intersection(net2_set)
	return len(intersection_edges) / (len(net1_set) + len(net2_set)) 

s_path = './network_data/emails/'
t_path = './network_data/commits/'
s_nets = set(os.listdir(s_path))
t_nets = set(os.listdir(t_path))
# have either socio or technical netowrks
nets = s_nets.union(t_nets)

projects = {}
for net_file in nets:
    project_name, period = net_file.split('__')
    if project_name not in projects:
        projects[project_name] = set()
    projects[project_name].add(int(period.replace('.edgelist', '')))

for project_name in projects:
    projects[project_name] = sorted(list(projects[project_name]))

with open('project_incubation_time.json', 'r') as f:
	project_incubation_dict = json.load(f)

df = pd.DataFrame()

s_total = []
t_total = []

for project_name in tqdm(sorted(projects.keys())):
    # may not have the network data
    for month in range(project_incubation_dict[project_name]):
        net_file = '{}__{}.edgelist'.format(project_name, month)
        social_net_path = s_path + net_file
        tech_net_path = t_path + net_file
        # remove extension
        s_net_features = cal_social_net(social_net_path)
        t_net_features = cal_tech_net(tech_net_path)
        # calculating network overlap
        last_t_network = t_path + '{}__{}.edgelist'.format(project_name, month-1)
        if not os.path.exists(last_t_network) or not os.path.exists(tech_net_path):
        	t_net_overlap = 0
        else:
        	t_net_overlap = get_net_overlap(tech_net_path, last_t_network)
        	t_total.append(t_net_overlap)

        last_s_network = s_path + '{}__{}.edgelist'.format(project_name, month-1)
        if not os.path.exists(last_s_network) or not os.path.exists(social_net_path):
        	s_net_overlap = 0
        else:
        	s_net_overlap = get_net_overlap(social_net_path, last_s_network)
        	s_total.append(s_net_overlap)

        # num. of developers that presence in both social net and technical net (not using)
        num_st_dev = len(s_net_features['s_dev_nodes'].intersection(t_net_features['t_dev_nodes']))
        project_features = {'proj_name':project_name,'month':month,'st_num_dev':num_st_dev, \
        't_net_overlap':t_net_overlap, 's_net_overlap': s_net_overlap}
        # remove keys
        s_net_features.pop('s_dev_nodes', None)
        t_net_features.pop('t_dev_nodes', None)
        all_features = {**s_net_features, **t_net_features, **project_features}
        df = df.append(all_features, ignore_index=True)

#to_path = './R/'
#if not os.path.exists(to_path):
#    os.makedirs(to_path)
df.to_csv('./network_data.csv', index=False)

#print(sum(s_total)/len(s_total))
#print(sum(t_total)/len(t_total))