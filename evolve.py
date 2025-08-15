
import neat
import networkx as nx
import RLagent as agent
import utils
from custom_genome import CustomGenome
from flax import linen as nn
import multiprocessing
import gc
import jax
from CustomReporter import CustomReporter
from parallel import ParallelEvaluator
import register
from stable_baselines3.common.env_util import make_vec_env
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium')

def log_complexity(genome, config, wandb_run, save=False, winner=False, gen=""):
        pruned = genome.get_pruned_copy(config.genome_config)

        G = utils.make_graph(pruned, config, 243)

        if save:
            run_name = wandb_run.id
            filename = run_name + "_gen" + str(gen) + "_Graph.adjlist"
            nx.write_adjlist(G, filename)
            wandb_run.save(filename)
            filename = run_name + "_gen" + str(gen) + "_Graph.png"
            utils.show_layered_graph(G, save=True, filename=filename)
            wandb_run.save(filename)

        ns = utils.get_ns(G)
        nc, mod, glob_eff = utils.get_nc(G)
        num_nodes = G.number_of_nodes()
        num_conn = G.number_of_edges()

        return ns, nc, num_nodes, num_conn, mod, glob_eff


def eval_genome(genome, config, learn=True, total_timesteps=100_000, skip_evaluated=True, n_seasons=2, 
                seed=11108, energy_costs=False):
    if skip_evaluated and genome.fitness is not None:
        return genome.fitness

    input_size = 243 #env.observation_space.shape[0]
    pruned = genome.get_pruned_copy(config.genome_config)
    G = utils.make_graph(pruned, config, input_size)

    if energy_costs:
        ns = utils.get_ns(G)

        connections = G.number_of_edges()
        if connections == 0:
            return -(ns*0.01) - 1

        energy_coef = 0.01/254
        env_args = {'n_seasons': n_seasons, 'col_dist': True, 'v': 4, 'size': 20, 'ns': ns, 
                    'col_seed':seed, 'col_var':0.2, 'energy_coef':energy_coef}
    else:
        env_args = {'n_seasons': n_seasons, 'col_dist': True, 'v': 4, 'size': 20, 'col_seed':seed}
    env = make_vec_env("Env-energy-v2", n_envs=5, env_kwargs=env_args)

    if genome.activation_fn == "relu":
        activation_fn = nn.relu
    else:
        activation_fn = nn.tanh

    learning_rate = genome.learning_rate
    if genome.lr_schedule == "linear":
         learning_rate = utils.linear_schedule(learning_rate)
   
    if True:
        model = agent.build_model(env, G, seed=seed, lr=learning_rate, gamma=genome.gamma, batch_size=genome.batch_size,
                                n_steps=genome.n_steps, ent_coef=genome.ent_coef, clip_range=genome.clip_range, 
                                n_epochs=genome.n_epochs, gae_lambda=genome.gae_lambda, max_grad_norm=genome.max_grad_norm, 
                                vf_coef=genome.vf_coef, activation_fn=activation_fn)
        if learn:
            model.learn(total_timesteps=total_timesteps, progress_bar=False)

        mean_reward, std_reward = agent.evaluate(model,deterministic=True, n_episodes=100, print_out=False)
    
    env.close()
    del model
    del G
    del env
    return mean_reward

def eval_genomes(genomes, config, learn=True, total_timesteps=100_000, skip_evaluated=True, n_seasons=2, seed=11108):
    learn = True
    for genome_id, genome in genomes:
        print("genome ", genome_id)
        if genome.fitness is None:
            genome.fitness = eval_genome(genome, config, learn, total_timesteps, skip_evaluated, args.n_seasons, args.seed)


def run(config_file='neat_config_exp_v2', parallel=True, wandb_run=None, restore=False, checkpoint_file=None, n_seasons=4, 
        seed=1361, n_gens=1, energy_costs=False, nprocs=3):

    if restore:
        checkpoint = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        p = neat.Population(checkpoint.config)
        p.population = checkpoint.population
        p.species = checkpoint.species
        p.generation = checkpoint.generation
        config = checkpoint.config

    else:
        CustomGenome.set_seed(seed)
        # Load configuration.
        config = neat.Config(CustomGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)
    
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = CustomReporter()
    p.add_reporter(stats)

    if wandb_run:
        filename_prefix = wandb_run.name + '-'
    else:
        filename_prefix = 'neat-checkpoint-'
    checkpointer = neat.Checkpointer(generation_interval=50, time_interval_seconds=None, filename_prefix=filename_prefix)
    p.add_reporter(checkpointer)

    if parallel:
        pe = ParallelEvaluator(nprocs, eval_genome, timeout=600, n_seasons=n_seasons, seed=seed, energy_costs=energy_costs)
        gen_start = p.generation
        print("gen start: ", gen_start)

    for generation in range(gen_start, n_gens):
        if parallel:
            try:
                gen_best = p.run(pe.evaluate, 1)
            except Exception as e:
                print(f"Error during evaluation: {e}")
                if isinstance(e, multiprocessing.context.TimeoutError):
                    print("Timeout occurred during evaluation.")
                print("Saving checkpoint...")
                checkpointer.save_checkpoint(
                    config=config,
                    population=p.population,
                    species_set=p.species,
                    generation=p.generation,
                )
                print("checkpoint saved")
                raise
        else: # non parallel version incomplete: Doesn't parse eval genomes arguments
            try:
                gen_best = p.run(eval_genomes, 1)
            except Exception as e:
                print(f"Error during evaluation: {e}")      
        if wandb_run:
            gen_mean = stats.get_fitness_mean()
            #wandb_run.log({"gen": p.generation-1, "gen_best_fitness": gen_best.fitness, "gen_mean_fitness": gen_mean})
            if (p.generation-1) % 100 == 0:
                ns, nc, num_nodes, num_conn, mod, glob_eff = log_complexity(gen_best, config, wandb_run, save=True, gen = p.generation-1)
            else:
                ns, nc, num_nodes, num_conn, mod, glob_eff = log_complexity(gen_best, config, wandb_run)
            if energy_costs:
                gen_best_task_perf = gen_best.fitness + (ns/254)
            else: 
                gen_best_task_perf = gen_best.fitness + 1
            wandb_run.log({"gen": p.generation-1, "gen_best_fitness": gen_best.fitness, "gen_mean_fitness": gen_mean, 
                "gen_best_ns_v2": ns, "gen_best_nc_v2": nc, "gen_best_num_nodes": num_nodes, "gen_best_num_conn": num_conn,
                "gen_best_modularity": mod, "gen_best_global_efficiency": glob_eff,
                "gen_best_batch_size": gen_best.batch_size, "gen_best_n_steps": gen_best.n_steps, "gen_best_gamma": gen_best.gamma,
                "gen_best_learning_rate": gen_best.learning_rate, "gen_best_ent_coef": gen_best.ent_coef, "gen_best_clip_range": gen_best.clip_range,
                "gen_best_n_epochs": gen_best.n_epochs, "gen_best_gae_lambda": gen_best.gae_lambda, "gen_best_max_grad_norm": gen_best.max_grad_norm,
                "gen_best_vf_coef": gen_best.vf_coef, "gen_best_activation_fn": gen_best.activation_fn, "gen_best_lr_schedule": gen_best.lr_schedule,
                "gen_best_task_perf": gen_best_task_perf})
        jax.clear_caches()
        gc.collect()
    winner = gen_best
    checkpointer.save_checkpoint(
        config=config,
        population=p.population,
        species_set=p.species,
        generation=p.generation,
        )        

    if wandb_run:
        ns, nc, num_nodes, num_conn, mod, glob_eff = log_complexity(winner, config, wandb_run, save=True, winner=True, gen = p.generation-1)
        wandb_run.log({"winner_ns_v2": ns, "winner_nc_v2": nc})

    return winner.fitness

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Optional app description')

    # seed argument
    parser.add_argument('--seed', type=int, default=1361,
                        help='Random seed for reproducibility (default: 1361)')

    # n_seasons argument
    parser.add_argument('--n_seasons', type=int, default=4,
                        help='Number of seasons in the environment (default: 4)')

    # n_gens argument
    parser.add_argument('--n_gens', type=int, default=1,
                        help='Number of generations for evolution (default: 1)')

    # EC argument
    parser.add_argument('--EC', action='store_true',
                        help='Enable energy costs that scale with ANN size')

    # n_procs argument
    parser.add_argument('--n_procs', type=int, default=3, 
                        help='Number of parallel processes for evaluating population fitness (default: 3)')

    args = parser.parse_args()
    print("Argument values:")
    print("seed: ", args.seed)
    print("n_seasons: ", args.n_seasons)
    print("Generations: ", args.n_gens)
    print("Energy Costs: ", args.EC)
    print("n_procs: ", args.n_procs)

                        
    run(n_seasons=args.n_seasons, seed=args.seed, n_gens=args.n_gens, energy_costs=args.EC, n_procs=args.n_procs)
