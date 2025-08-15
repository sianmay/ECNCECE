import random
import neat
import math
import copy

from neat.genome import DefaultGenomeConfig
from neat.config import ConfigParameter
from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.graphs import creates_cycle

class CustomGenomeConfig(DefaultGenomeConfig):
    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('num_hidden', int),
                        ConfigParameter('feed_forward', bool),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('conn_delete_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('single_structural_mutation', bool, 'false'),
                        ConfigParameter('structural_mutation_surer', str, 'default'),
                        ConfigParameter('initial_connection', str, 'unconnected'),
                        ConfigParameter('mutate_batch_size', float),
                        ConfigParameter('mutate_n_steps', float),
                        ConfigParameter('mutate_gamma', float),
                        ConfigParameter('mutate_learning_rate', float),
                        ConfigParameter('mutate_ent_coef', float),
                        ConfigParameter('mutate_clip_range', float),
                        ConfigParameter('mutate_n_epochs', float),
                        ConfigParameter('mutate_gae_lambda', float),
                        ConfigParameter('mutate_max_grad_norm', float),
                        ConfigParameter('mutate_vf_coef', float),
                        ConfigParameter('mutate_activation_fn', float),
                        ConfigParameter('mutate_lr_schedule', float),
        ]
        
        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        self.node_gene_type.validate_attributes(self)
        self.connection_gene_type.validate_attributes(self)

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ['1', 'yes', 'true', 'on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0', 'no', 'false', 'off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)

        self.node_indexer = None

class CustomGenome(neat.DefaultGenome):

    def set_seed(seed):
        random.seed(seed)
    
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return CustomGenomeConfig(param_dict)

    def __init__(self, key):
        super().__init__(key)
        self.batch_size = None
        self.n_steps = None
        self.gamma = None
        self.learning_rate = None
        self.ent_coef = None
        self.clip_range = None
        self.n_epochs = None
        self.gae_lambda = None
        self.max_grad_norm = None
        self.vf_coef = None
        #self.graph = None
        self.num_hidden = None
        self.activation_fn = None
        self.lr_schedule = None


    def configure_new(self, config):
        super().configure_new(config)

        self.batch_size = random.choice([8, 16, 32, 64, 128, 256, 512])
        self.n_steps = random.choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        self.gamma = random.choice([0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        self.learning_rate = random.choice([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,
                                                            0.0009,0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01])
        self.ent_coef = 10**random.uniform(math.log10(1e-8), math.log10(0.1))
        self.clip_range = random.choice([0.1, 0.2, 0.3, 0.4])
        self.n_epochs = random.choice([1, 5, 10, 20])
        self.gae_lambda = random.choice([0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        self.max_grad_norm = random.choice([0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        self.vf_coef = random.uniform(0, 1)

        if self.batch_size > self.n_steps:
            self.batch_size = self.n_steps

        self.activation_fn = random.choice(["tanh", "relu"])
        self.lr_schedule =random.choice(['linear', 'constant'])
        


    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.batch_size = random.choice((genome1.batch_size, genome2.batch_size))
        self.n_steps = random.choice((genome1.n_steps, genome2.n_steps))
        self.gamma = random.choice((genome1.gamma, genome2.gamma))
        self.learning_rate = random.choice((genome1.learning_rate, genome2.learning_rate))
        self.ent_coef = random.choice((genome1.ent_coef, genome2.ent_coef))
        self.clip_range = random.choice((genome1.clip_range, genome2.clip_range))
        self.n_epochs = random.choice((genome1.n_epochs, genome2.n_epochs))
        self.gae_lambda = random.choice((genome1.gae_lambda, genome2.gae_lambda))
        self.max_grad_norm = random.choice((genome1.max_grad_norm, genome2.max_grad_norm))
        self.vf_coef = random.choice((genome1.vf_coef, genome2.vf_coef))
        self.activation_fn = random.choice((genome1.activation_fn, genome2.activation_fn))
        self.lr_schedule = random.choice((genome1.lr_schedule, genome2.lr_schedule))


    def mutate(self, config):
        super().mutate(config)

        if random.random() < config.mutate_batch_size:
            self.batch_size = random.choice([8, 16, 32, 64, 128, 256, 512])
        if random.random() < config.mutate_n_steps:
            self.n_steps = random.choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        if random.random() < config.mutate_gamma:
            self.gamma = random.choice([0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        if random.random() < config.mutate_learning_rate:
            self.learning_rate = random.choice([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,
                                                            0.0009,0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 
                                                            0.008, 0.009, 0.01])
        if random.random() < config.mutate_ent_coef:
            self.ent_coef = 10**random.uniform(math.log10(1e-8), math.log10(0.1))
        if random.random() < config.mutate_clip_range:
            self.clip_range = random.choice([0.1, 0.2, 0.3, 0.4])
        if random.random() < config.mutate_n_epochs:
            self.n_epochs = random.choice([1, 5, 10, 20])
        if random.random() < config.mutate_gae_lambda:
            self.gae_lambda = random.choice([0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        if random.random() < config.mutate_max_grad_norm:
            self.max_grad_norm = random.choice([0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        if random.random() < config.mutate_vf_coef:
            self.vf_coef = random.uniform(0, 1)

        if self.batch_size > self.n_steps:
            self.batch_size = self.n_steps

        if random.random() < config.mutate_activation_fn:
            self.activation_fn = random.choice(["tanh", "relu"])

        if random.random() < config.mutate_lr_schedule:
            random.choice(['linear', 'constant'])

    def mutate_add_node(self, config):
        
        # new node adds two new connections to it, instead of splitting existing connection
        conn_to_split = self.mutate_add_connection(config)

        if not conn_to_split:
            conn_to_split = random.choice(list(self.connections.values()))

        #print("conn_to_split: ", conn_to_split)

        # Choose a random connection to split
        #conn_to_split = choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id, 1.0, True)
        self.add_connection(config, new_node_id, o, conn_to_split.weight, True)

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        
        possible_outputs = list(self.nodes)
        out_node = random.choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = random.choice(possible_inputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            # TODO: Should this be using mutation to/from rates? Hairy to configure...
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return self.connections[key]

        # Don't allow connections between two output nodes
        if in_node in config.output_keys: #and out_node in config.output_keys:
            return

        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(self.connections), key):
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg
        return cg

    def get_pruned_copy(self, genome_config):
        used_node_genes, used_connection_genes = get_pruned_genes(self.nodes, self.connections,
                                                                  genome_config.input_keys, genome_config.output_keys)
        new_genome = neat.DefaultGenome(None)
        new_genome.nodes = used_node_genes
        new_genome.connections = used_connection_genes
        return new_genome

def get_pruned_genes(node_genes, connection_genes, input_keys, output_keys):

    used_nodes = required_for_output(input_keys, output_keys, connection_genes)
    used_pins = used_nodes.union(input_keys)

    # Copy used nodes into a new genome.
    used_node_genes = {}
    for n in used_nodes:
        used_node_genes[n] = copy.deepcopy(node_genes[n])

    # Copy enabled and used connections into the new genome.
    used_connection_genes = {}
    for key, cg in connection_genes.items():
        in_node_id, out_node_id = key
        if cg.enabled and in_node_id in used_pins and out_node_id in used_pins:
            used_connection_genes[key] = copy.deepcopy(cg)
        #elif cg.enabled:
        #    connection_genes[key].enabled = False
    #used_nodes2 = required_for_output(input_keys, output_keys, used_connection_genes)

    return used_node_genes, used_connection_genes

def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s),
    but only if they are reachable from the input nodes.

    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a set of identifiers of required nodes.
    """
    assert not set(inputs).intersection(outputs)

    # Filter connections to only include enabled ones
    enabled_connections = [key for key, value in connections.items() if value.enabled]

    # Step 1: Identify all nodes reachable from inputs
    reachable_from_inputs = set(inputs)
    s = set(inputs)
    while True:
        t = set(b for (a, b) in enabled_connections if a in s and b not in s)
        if not t:
            break
        reachable_from_inputs.update(t)
        s.update(t)

    # Step 2: Identify nodes required for outputs
    required = set(outputs)
    s = set(outputs)
    while True:
        # Find nodes not in `s` whose output is consumed by a node in `s`
        t = set(a for (a, b) in enabled_connections if b in s and a not in s)
        if not t:
            break
        layer_nodes = set(x for x in t if x not in inputs)
        required.update(layer_nodes)
        s.update(t)

    # Step 3: Filter required nodes to include only those reachable from inputs
    required = required.intersection(reachable_from_inputs)
    #required = required.intersection(reachable_from_inputs).union(outputs)
    return required

