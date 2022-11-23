import os
import time
import logging
import json
logging.basicConfig(level=logging.INFO)
import pickle
import argparse
from functools import partial

import numpy as np
from distributed import Client
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant

import torch

torch.cuda.empty_cache()

import ConfigSpace as CS

from optimizer import run_model
from DEHB.dehb import DEHB, AsyncDE


def save_incumbent(inc_config, inc_score, inc_info, file_name):
    try:
        # res = dict()
        res = inc_config.get_dictionary()
        # res["score"] = inc_score
        # res["info"] = inc_info
        with open(os.path.join("{}.json".format(file_name)), 'w') as f:
            json.dump(res, f)
    except Exception as e:
        logging.warning("Incumbent not saved: {}".format(repr(e)))


if __name__ == '__main__':

    cmdline_parser = argparse.ArgumentParser('AutoML SS21 final project')

    cmdline_parser.add_argument('-e', '--epochs',
                                default=50,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-m', '--model_path',
                                default=None,
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-s', "--constraint_max_model_size",
                                default=2e7,
                                help="maximal model size constraint",
                                type=int)
    cmdline_parser.add_argument('-p', "--constraint_min_precision",
                                default=0.39,
                                help='minimal constraint constraint',
                                type=float)
    cmdline_parser.add_argument('-r', "--run_id",
                                default='0',
                                help='run id ',
                                type=str)
    cmdline_parser.add_argument('-min', '--min_budget',
                                type=float,
                                default=10,
                                help='Minimum budget (epoch length)')
    cmdline_parser.add_argument('-max', '--max_budget',
                                type=float,
                                default=50,
                                help='Maximum budget (epoch length)')
    cmdline_parser.add_argument('-eta', '--eta',
                                type=int,
                                default=3,
                                help='Parameter for Hyperband controlling early stopping aggressiveness')
    cmdline_parser.add_argument('-out', '--output_path',
                                type=str,
                                default="./dehb_results",
                                help='Directory for DEHB to write logs and outputs')
    cmdline_parser.add_argument('-o', '--optimizer',
                                type=str,
                                default="DE")
    cmdline_parser.add_argument('-ru', '--runtime',
                                type=float,
                                default=70000,
                                help='Total time in seconds as budget to run DEHB')
    cmdline_parser.add_argument('-po', '--popsize',
                                type=int,
                                default=20,
                                help='Population size for DE')

    args, unknowns = cmdline_parser.parse_known_args()
    # HERE ARE THE CONSTRAINTS!
    # HERE ARE THE CONSTRAINTS!
    # HERE ARE THE CONSTRAINTS!
    constraint_model_size = args.constraint_max_model_size
    constraint_precision = args.constraint_min_precision

    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = CS.ConfigurationSpace()

    # We can add multiple hyperparameters at once:
    n_conf_layer = UniformIntegerHyperparameter("n_conv_layers", 1, 3, default_value=3)
    n_conf_layer_0 = UniformIntegerHyperparameter("n_channels_conv_0", 512, 2048, default_value=512)
    n_conf_layer_1 = UniformIntegerHyperparameter("n_channels_conv_1", 512, 2048, default_value=512)
    n_conf_layer_2 = UniformIntegerHyperparameter("n_channels_conv_2", 512, 2048, default_value=512)
    n_fc_layers = UniformIntegerHyperparameter("n_fc_layers", 1, 3, default_value=2)
    n_fc_layer_0 = UniformIntegerHyperparameter("n_channels_fc_0", 64, 256, default_value=256)
    n_fc_layer_1 = UniformIntegerHyperparameter("n_channels_fc_1", 64, 256, default_value=128)
    n_fc_layer_2 = UniformIntegerHyperparameter("n_channels_fc_2", 64, 256, default_value=64)
    kernel_size = CategoricalHyperparameter("kernel_size", choices=[2, 3, 4, 5])
    learning_rate_init = UniformFloatHyperparameter('learning_rate_init',
                                                    0.00001, 1.0, default_value=0.001, log=True)
    batch_size = UniformIntegerHyperparameter("batch_size", 128, 512, default_value=128, log=True)
    glob_av_pool = CategoricalHyperparameter("global_avg_pooling", choices=[False, True])
    use_BN = CategoricalHyperparameter("use_BN", choices=[False, True])
    dropout_rate = UniformFloatHyperparameter("dropout_rate", 0.0, 1.0, default_value=0.2)
    optimizer = CategoricalHyperparameter('optimizer', choices=['SGD', 'Adam', 'AdamW'])
    sgd_momentum = UniformFloatHyperparameter('sgd_momentum', 0.0, 0.99, default_value=0.9)
    weight_decay = UniformFloatHyperparameter('weight_decay', 0.00001, 0.1, default_value=0.0001)
    cs.add_hyperparameters([n_conf_layer, n_conf_layer_0, n_conf_layer_1, n_conf_layer_2, kernel_size,
                            learning_rate_init, batch_size, glob_av_pool, use_BN, dropout_rate, optimizer,
                            sgd_momentum, n_fc_layers, n_fc_layer_2, n_fc_layer_1, n_fc_layer_0,
                            weight_decay])

    # Add conditions to restrict the hyperparameter space
    use_sgd_momentum = CS.conditions.InCondition(sgd_momentum, optimizer, ["SGD"])
    use_conf_layer_2 = CS.conditions.InCondition(n_conf_layer_2, n_conf_layer, [3])
    use_conf_layer_1 = CS.conditions.InCondition(n_conf_layer_1, n_conf_layer, [2, 3])
    use_fc_layer_2 = CS.conditions.InCondition(n_fc_layer_2, n_fc_layers, [3])
    use_fc_layer_1 = CS.conditions.InCondition(n_fc_layer_1, n_fc_layers, [2, 3])
    # Add  multiple conditions on hyperparameters at once:
    cs.add_conditions([use_conf_layer_2, use_conf_layer_1, use_sgd_momentum, use_fc_layer_2, use_fc_layer_1])

    dimensions = len(cs.get_hyperparameters())

    if args.optimizer == "DE":
        run = "de_popsize_{}_precision_{}_modelsize_{}".format(args.popsize, constraint_precision,
                                                               constraint_model_size)
        de = AsyncDE(f=partial(run_model, seed=123), cs=cs, dimensions=dimensions, output_path=args.output_path,
                     client=None, n_workers=1,
                     budget=args.max_budget, pop_size=args.popsize, mutation_factor=0.5, crossover_prob=0.5,
                     max_model_params=constraint_model_size)
        traj, runtime, history = de.run(generations=1, budget=args.max_budget, verbose=args.verbose,
                                        popsize=args.popsize, constraint_precision=constraint_precision,
                                        constraint_model_size=constraint_model_size, out_file=run)
        inc_config = de.vector_to_configspace(de.inc_config)
        inc_info = de.inc_info
        inc_score = de.inc_score
        if inc_info["constraints"] == 0:
            save_incumbent(inc_config=inc_config, inc_score=inc_score, inc_info=inc_info,
                           file_name="de_incumbent_precision_{}_model_size_{}"
                           .format(constraint_precision, constraint_model_size))
        else:
            logging.info("No configuration found that satisfies the constraints")
        name = time.strftime("%x %X %Z", time.localtime(time.time()))
        name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
        logging.info("Saving optimisation trace history...")
        with open(os.path.join(args.output_path, "history_{}.pkl".format(name)), "wb") as f:
            pickle.dump(history, f)
        with open(os.path.join(args.output_path, "traj_{}.pkl".format(name)), "wb") as f:
            pickle.dump(traj, f)

    else:
        run = "dehb_eta_{}_precison_{}_modelsize_{}".format(args.eta, constraint_precision, constraint_model_size)
        dehb = DEHB(f=run_model, cs=cs, dimensions=dimensions, min_budget=args.min_budget,
                    max_budget=args.max_budget, eta=args.eta, output_path=args.output_path, client=None, n_workers=1,
                    max_model_params=constraint_model_size)
        traj, runtime, history = dehb.run(total_cost=args.runtime, verbose=args.verbose, seed=123,
                                          constraint_precision=constraint_precision,
                                          constraint_model_size=constraint_model_size, out_file=run)
        inc_config = dehb.vector_to_configspace(dehb.inc_config)
        inc_info = dehb.inc_info
        inc_score = dehb.inc_score
        if inc_info["constraints"] == 0:
            save_incumbent(inc_config=inc_config, inc_score=inc_score, inc_info=inc_info,
                           file_name="dehb_incumbent_eta_{}_precision_{}_model_size_{}"
                           .format(args.eta, constraint_precision, constraint_model_size))
        else:
            logging.info("No configuration found that satisfies the constraints")
        name = time.strftime("%x %X %Z", time.localtime(dehb.start))
        name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
        dehb.logger.info("Saving optimisation trace history...")
        with open(os.path.join(args.output_path, "history_{}.pkl".format(name)), "wb") as f:
            pickle.dump(history, f)
