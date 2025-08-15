import networkx as nx
import matplotlib.pyplot as plt
from forward.models import ForWard as fwd
import numpy as np
import copy
from typing import Callable
from forward.utils import longest_path_algorithm

def show_layered_graph(G, label=None, save=False, show=False, filename="Graph.png"):
    plt.clf() 
    ffnn = fwd(G)
    L = ffnn.L

    G = G.to_undirected()
    for l, nodes in enumerate(L):
        for v in nodes:
            G.nodes[v]['subset'] = l
    pos = nx.multipartite_layout(G)
    print('# of input features:', ffnn.in_features)
    print('# of output features:', ffnn.out_features)
    for layer in L:
        print(len(layer))
    print()
    for preds in ffnn.layer_preds:
        print(len(preds))
    labels = nx.get_node_attributes(G, label)
    nx.draw_networkx(G, pos, with_labels=False, node_size=50, width=0.1)
    if save:
        plt.savefig(filename, format="PNG")
    if show:
        plt.show()


def mlp_graph(net_arch = [27, 8, 8, 5], direct=False):
    # Initialize directed graph
    G = nx.DiGraph()
    layers = []
    # add nodes to graphs
    s1 = 0
    for i, layer_size in enumerate(net_arch):
        s2 = s1 + layer_size
        l = list(range(s1, s2))
        layers.append(l)
        name = 'layer_' + str(1)
        G.add_nodes_from(l, layer=name)
        s1 = s2

    if not direct:
        # add connections
        for l1, l2 in zip(layers[:-1], layers[1:]):
            for i in l1:
                for j in l2:
                    G.add_edge(i, j)


    def connect_layers(l1, l2):
        for i in l1:
            for j in l2:
                G.add_edge(i,j)

    if direct:
        for i,l1 in enumerate(layers[:-1]):
            print(i)
            for l2 in layers[i+1:]:
                connect_layers(l1,l2)
            

    return G

def ffnn2graph(G, params):
    ffnn = fwd(G, bias=False)
    G_copy = copy.deepcopy(G)

    last_layer = False
    for W_, layer, preds in zip(params.values(), ffnn.L[1:], ffnn.layer_preds):
        W = W_["kernel"]
        b = W_["bias"]

        for j in range(len(layer)):
            v = layer[j]
            G_copy.nodes[v]["bias"] = b[j].item()
            for i in range(len(preds)):
                u = preds[i]
                weight = W[i,j].item()
                if weight != 0:
                    #G_new.add_edge(u, v)#, weight=weight)
                    if not G_copy.has_edge(u, v):
                        print("G has no edge: ", u, v)
                    G_copy[u][v]['weight'] = weight

    #show_layered_graph(G_copy)
    return G_copy

def mod_eff(G):
    Gu = G.to_undirected()
    mods = []
    seeds = [120,121,122,123,124,125,126,127,128,129]
    for seed in seeds:
        mod = nx.community.modularity(Gu, nx.community.louvain_communities(Gu, seed=seed, weight=None), weight=None)
        mods.append(mod)
    mod = np.mean(mods)
    glob_eff = nx.global_efficiency(Gu)
    return mod, glob_eff

def get_nc(G):
    # Iterate over the nodes and remove those with no incoming or outgoing edges
    nodes_to_remove = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
    # Remove the nodes from the graph
    G.remove_nodes_from(nodes_to_remove)
    mod, glob_eff = mod_eff(G)
    nc = min(mod,glob_eff)/max(mod,glob_eff)
    return nc, mod, glob_eff

def get_ns(G, input_len=243):
    return G.number_of_nodes()-input_len+G.number_of_edges()

def make_graph(genome, config, input_size):
    G = nx.DiGraph()

    for node in genome.nodes.keys():
        bias = genome.nodes[node].bias
        activation = genome.nodes[node].activation
        G.add_node(node, bias=bias, activation=activation)
    for k in genome.connections.keys():
        enabled = genome.connections[k].enabled
        if enabled:
            weight = genome.connections[k].weight
            G.add_edge(k[0], k[1], weight=weight)

    # ensure correct input size
    for n in range(-1*input_size,0):
        if n not in (G.nodes):
            G.add_node(n)

    return G

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def generate_neat_config(
    filename="neat_config",
    fitness_criterion="max",
    fitness_threshold=3.9,
    pop_size=20,
    reset_on_extinction="true",
    no_fitness_termination="true",
    
    # CustomGenome parameters
    activation_default="sigmoid",
    activation_mutate_rate=0.0,
    aggregation_default="sum",
    aggregation_mutate_rate=0.0,
    bias_init_mean=0.0,
    bias_init_stdev=1.0,
    bias_max_value=30.0,
    bias_min_value=-30.0,
    bias_mutate_power=0.4,
    bias_mutate_rate=0.0,
    bias_replace_rate=0.02,
    compatibility_disjoint_coefficient=1.0,
    compatibility_weight_coefficient=0.5,
    conn_add_prob=0.5,
    conn_delete_prob=0.5,
    enabled_default="true",
    enabled_mutate_rate=0.0,
    feed_forward="true",
    initial_connection="full_nodirect",
    node_add_prob=0.5,
    node_delete_prob=0.5,
    num_hidden=128,
    num_inputs=243,
    num_outputs=5,
    response_init_mean=1.0,
    response_init_stdev=0.0,
    response_max_value=30.0,
    response_min_value=-30.0,
    response_mutate_power=0.0,
    response_mutate_rate=0.0,
    response_replace_rate=0.0,
    weight_init_mean=0.0,
    weight_init_stdev=1.0,
    weight_max_value=30.0,
    weight_min_value=-30.0,
    weight_mutate_power=0.4,
    weight_mutate_rate=0.0,
    weight_replace_rate=0.02,
    structural_mutation_surer="true",
    single_structural_mutation="false",
    
    # Default species and reproduction parameters
    compatibility_threshold=3.0,
    species_fitness_func="max",
    max_stagnation=15,
    species_elitism=2,
    elitism=2,
    survival_threshold=0.2,

    # RL hyperparameters
    mutate_batch_size=0.3,
    mutate_n_steps = 0.3,
    mutate_gamma = 0.3,
    mutate_learning_rate=0.3,
    mutate_ent_coef=0.3,
    mutate_clip_range=0.3,
    mutate_n_epochs=0.3,
    mutate_gae_lambda=0.3,
    mutate_max_grad_norm=0.3,
    mutate_vf_coef=0.3,
    mutate_activation_fn=0.3,
    mutate_lr_schedule=0.3
):
    with open(filename, "w") as config_file:
        config_file.write(f"[NEAT]\n")
        config_file.write(f"fitness_criterion = {fitness_criterion}\n")
        config_file.write(f"fitness_threshold = {fitness_threshold}\n")
        config_file.write(f"pop_size = {pop_size}\n")
        config_file.write(f"reset_on_extinction = {reset_on_extinction}\n")
        config_file.write(f"no_fitness_termination = {no_fitness_termination}\n\n")

        config_file.write(f"[CustomGenome]\n")
        config_file.write(f"# Node activation options\n")
        config_file.write(f"activation_default = {activation_default}\n")
        config_file.write(f"activation_mutate_rate = {activation_mutate_rate}\n")
        config_file.write(f"activation_options = {activation_default}\n\n")

        config_file.write(f"# Node aggregation options\n")
        config_file.write(f"aggregation_default = {aggregation_default}\n")
        config_file.write(f"aggregation_mutate_rate = {aggregation_mutate_rate}\n")
        config_file.write(f"aggregation_options = {aggregation_default}\n\n")

        config_file.write(f"# Node bias options\n")
        config_file.write(f"bias_init_mean = {bias_init_mean}\n")
        config_file.write(f"bias_init_stdev = {bias_init_stdev}\n")
        config_file.write(f"bias_max_value = {bias_max_value}\n")
        config_file.write(f"bias_min_value = {bias_min_value}\n")
        config_file.write(f"bias_mutate_power = {bias_mutate_power}\n")
        config_file.write(f"bias_mutate_rate = {bias_mutate_rate}\n")
        config_file.write(f"bias_replace_rate = {bias_replace_rate}\n\n")

        config_file.write(f"# Genome compatibility options\n")
        config_file.write(f"compatibility_disjoint_coefficient = {compatibility_disjoint_coefficient}\n")
        config_file.write(f"compatibility_weight_coefficient = {compatibility_weight_coefficient}\n\n")

        config_file.write(f"# Connection add/remove rates\n")
        config_file.write(f"conn_add_prob = {conn_add_prob}\n")
        config_file.write(f"conn_delete_prob = {conn_delete_prob}\n\n")

        config_file.write(f"# Connection enable options\n")
        config_file.write(f"enabled_default = {enabled_default}\n")
        config_file.write(f"enabled_mutate_rate = {enabled_mutate_rate}\n\n")

        config_file.write(f"feed_forward = {feed_forward}\n")
        config_file.write(f"initial_connection = {initial_connection}\n\n")

        config_file.write(f"# Node add/remove rates\n")
        config_file.write(f"node_add_prob = {node_add_prob}\n")
        config_file.write(f"node_delete_prob = {node_delete_prob}\n\n")

        config_file.write(f"# Network parameters\n")
        config_file.write(f"num_hidden = {num_hidden}\n")
        config_file.write(f"num_inputs = {num_inputs}\n")
        config_file.write(f"num_outputs = {num_outputs}\n\n")

        config_file.write(f"# Node response options\n")
        config_file.write(f"response_init_mean = {response_init_mean}\n")
        config_file.write(f"response_init_stdev = {response_init_stdev}\n")
        config_file.write(f"response_max_value = {response_max_value}\n")
        config_file.write(f"response_min_value = {response_min_value}\n")
        config_file.write(f"response_mutate_power = {response_mutate_power}\n")
        config_file.write(f"response_mutate_rate = {response_mutate_rate}\n")
        config_file.write(f"response_replace_rate = {response_replace_rate}\n\n")

        config_file.write(f"# Connection weight options\n")
        config_file.write(f"weight_init_mean = {weight_init_mean}\n")
        config_file.write(f"weight_init_stdev = {weight_init_stdev}\n")
        config_file.write(f"weight_max_value = {weight_max_value}\n")
        config_file.write(f"weight_min_value = {weight_min_value}\n")
        config_file.write(f"weight_mutate_power = {weight_mutate_power}\n")
        config_file.write(f"weight_mutate_rate = {weight_mutate_rate}\n")
        config_file.write(f"weight_replace_rate = {weight_replace_rate}\n\n")

        config_file.write(f"# RL hyperparameter options\n")
        config_file.write(f"mutate_batch_size = {mutate_batch_size}\n")
        config_file.write(f"mutate_n_steps = {mutate_n_steps}\n")
        config_file.write(f"mutate_gamma = {mutate_gamma}\n")
        config_file.write(f"mutate_learning_rate = {mutate_learning_rate}\n")
        config_file.write(f"mutate_ent_coef = {mutate_ent_coef}\n")
        config_file.write(f"mutate_clip_range = {mutate_clip_range}\n")
        config_file.write(f"mutate_n_epochs = {mutate_n_epochs}\n")
        config_file.write(f"mutate_gae_lambda = {mutate_gae_lambda}\n")
        config_file.write(f"mutate_max_grad_norm = {mutate_max_grad_norm}\n")
        config_file.write(f"mutate_vf_coef = {mutate_vf_coef}\n")
        config_file.write(f"mutate_activation_fn = {mutate_activation_fn}\n")
        config_file.write(f"mutate_lr_schedule = {mutate_lr_schedule}\n")


        config_file.write(f"structural_mutation_surer = {structural_mutation_surer}\n")
        config_file.write(f"single_structural_mutation = {single_structural_mutation}\n\n")

        config_file.write(f"[DefaultSpeciesSet]\n")
        config_file.write(f"compatibility_threshold = {compatibility_threshold}\n\n")

        config_file.write(f"[DefaultStagnation]\n")
        config_file.write(f"species_fitness_func = {species_fitness_func}\n")
        config_file.write(f"max_stagnation = {max_stagnation}\n")
        config_file.write(f"species_elitism = {species_elitism}\n\n")

        config_file.write(f"[DefaultReproduction]\n")
        config_file.write(f"elitism = {elitism}\n")
        config_file.write(f"survival_threshold = {survival_threshold}\n")

    print(f"Configuration file '{filename}' has been generated.")
    return filename

def push_sources_and_sinks(G, L):
    """
    The function places all nodes with values < 0 in the first layer (L[0]) and
    all nodes with values >= 0 and < 5 in the last layer of layering L.
    """

    # Retrieve nodes with values < 0 (to be placed in the first layer)
    sources = [node for node in G.nodes if node < 0]

    # Retrieve nodes with values >= 0 and < 5 (to be placed in the last layer)
    sinks = [node for node in G.nodes if 0 <= node < 5]

    # Update the first layer
    L[0] = sorted(sources)

    # Update the last layer
    L[-1] = sorted(sinks)

    # Remove sources and sinks from all intermediate layers
    for i in range(1, len(L) - 1):
        L[i] = sorted(list(set(L[i]) - set(sources) - set(sinks)))

    return L

class Graph2ffnn:
#def graph2ffnn(dag, bias=False):
    def __init__(self, dag):

        # assign layers to nodes
        L = longest_path_algorithm(dag)

        # push source nodes into the first layer and sink nodes into last layer
        L = push_sources_and_sinks(dag, L)

        # retrieve source and sink nodes
        sources = L[0]
        sinks = L[-1]

        layer_preds = []
        weights = []
        masks = []

        for i, layer in enumerate(L[1:]):
            # store predecessors of nodes in the current layer
            preds = set()
            for v in layer:
                preds |= set(dag.predecessors(v))
            preds = sorted(list(preds)) # NOTE: integer node IDs are expected

            # allocate memory for the layer weights
            W = np.zeros((len(preds), len(layer)))

            # build a mask to restore the complex topology
            M = np.zeros_like(W)
            for j, v in enumerate(layer):
                for pred in dag.predecessors(v):
                    M[preds.index(pred),j] = 1.

            layer_preds.append(preds)
            weights.append(W)
            masks.append(M)

        self.in_features = len(sources) #
        self.out_features = len(sinks) #
        self.dag = dag #
        self.L = L #
        self.sources = sources #
        self.sinks = sinks #
        #bias = bias
        self.layer_preds = layer_preds #
        self.weights = weights #
        self.masks = masks #

        #return dag, in_features, out_features, L, sources, sinks, bias, layer_preds, weights, masks


