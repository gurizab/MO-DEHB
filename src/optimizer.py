import os
import time
import logging

logging.basicConfig(level=logging.INFO)
import pickle
import argparse
import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from sklearn.model_selection import StratifiedKFold
import json
import torch
from torchsummary import summary
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import ConfigSpace as CS
from DEHB.dehb import DEHB
from cnn import torchModel
from torchvision import transforms


def get_optimizer_and_crit(cfg):
    if cfg['optimizer'] == 'AdamW':
        model_optimizer = torch.optim.AdamW
    elif cfg['optimizer'] == 'Adam':
        model_optimizer = torch.optim.Adam
    else:
        model_optimizer = torch.optim.SGD

    train_criterion = torch.nn.CrossEntropyLoss

    return model_optimizer, train_criterion


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
def run_model(cfg, budget, seed, constraint_precision=0.39, constraint_model_size=2e7,
             data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower'),
             out_file='', **kwargs):
    """
        Creates an instance of the torch_model and fits the given data on it.
        This is the function-call we try to optimize. Chosen values are stored in
        the configuration (cfg).

        Parameters
        ----------
        eta: Number of Successive Halving brackets
        constraint_precision: Precision constraint
        data_dir: Directory of the data
        cfg: Configuration (basically a dictionary)
            configuration chosen by smac
        seed: int or RandomState
            used to initialize the rf's random generator
        instance: str
            used to represent the instance to use (just a placeholder for this example)
        budget: float
            used to set max iterations for the MLP

        Returns
        -------
        float
    """
    logging.info("Constraints to be satisfied: Max Model Size {}, Min Precision {}".format(constraint_model_size,
                                                                                           constraint_precision))
    logging.info("Running configuration %s" % cfg)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_augmentations = transforms.ToTensor()

    data = ImageFolder(os.path.join(data_dir, "train"), transform=data_augmentations)
    test_data = ImageFolder(os.path.join(data_dir, "test"), transform=data_augmentations)
    targets = data.targets

    # image size
    channels, img_height, img_width = data[0][0].shape
    input_shape = (channels, img_height, img_width)
    model = torchModel(cfg,
                       input_shape=input_shape,
                       num_classes=len(data.classes)).to(device)
    # instantiate optimizer
    model_optimizer, train_criterion = get_optimizer_and_crit(cfg)
    if cfg['optimizer'] == 'SGD':
        optimizer = model_optimizer(model.parameters(),
                                    lr=cfg['learning_rate_init'], momentum=cfg['sgd_momentum'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = model_optimizer(model.parameters(),
                                    lr=cfg['learning_rate_init'], weight_decay=cfg['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=0.00001)
    total_model_params = np.sum(p.numel() for p in model.parameters())
    train_criterion = train_criterion().to(device)
    logging.info('Generated Network:')
    summary(model, input_shape, device=device)
    num_epochs = int(np.ceil(budget))

    # Train the model
    score = []
    score_top5 = []
    score_precision = []

    # returns the cross validation accuracy
    # to make CV splits consistent
    start = time.time()
    cv = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)  # to make CV splits consistent
    for train_idx, valid_idx in cv.split(data, targets):
        train_data = Subset(data, train_idx)
        val_dataset = Subset(data, valid_idx)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=cfg["batch_size"],
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=cfg["batch_size"],
                                shuffle=False)
        for epoch in range(num_epochs):
            logging.info('#' * 50)
            logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, device)
            val_score, val_score_top5, val_score_precision = model.eval_fn(
                val_loader, device)
            lr_scheduler.step()
            logging.info('Train accuracy %f', train_score)
            logging.info('Test accuracy %f', val_score)
            logging.info('Test Precision %f', np.mean(val_score_precision))
        score.append(val_score)
        score_top5.append(val_score_top5)
        score_precision.append(np.mean(val_score_precision))
        # reset model, optimizer and crit
        model = torchModel(cfg,
                           input_shape=input_shape,
                           num_classes=len(data.classes)).to(device)
        # if total_model_params < 1e5 or total_model_params == 1e5:
        # instantiate optimizer
        model_optimizer, train_criterion = get_optimizer_and_crit(cfg)
        if cfg['optimizer'] == 'SGD':
            optimizer = model_optimizer(model.parameters(),
                                        lr=cfg['learning_rate_init'], momentum=cfg['sgd_momentum'],
                                        weight_decay=cfg['weight_decay'])
        else:
            optimizer = model_optimizer(model.parameters(),
                                        lr=cfg['learning_rate_init'], weight_decay=cfg['weight_decay'])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=0.00001)
        total_model_params = np.sum(p.numel() for p in model.parameters())
        train_criterion = train_criterion().to(device)
    cost = time.time() - start
    acc = 1 - np.mean(score)
    acc_top5 = np.mean(score_top5)
    precision = np.mean(score_precision)
    constraint_function = 0
    if precision < constraint_precision:
        constraint_function += (constraint_precision - precision)

    with open('%s.json' % out_file, 'a+') as f:
        json.dump({'configuration': dict(cfg), 'top3': np.mean(score), 'top5': acc_top5, 'precision': precision,
                   'n_params': total_model_params, 'budget': budget}, f)
        f.write("\n")
    res = {
        "fitness": acc,
        "cost": cost,
        "info": {"budget": budget, "constraints": constraint_function, "precision": precision,
                 "num_model_params": total_model_params}
    }
    return res


if __name__ == '__main__':
    """
    This is just an example of how to implement BOHB as an optimizer!
    Here we do not consider any of the constraints and 
    """

    cmdline_parser = argparse.ArgumentParser('AutoML SS21 final project BOHB example')

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
    cmdline_parser.add_argument('-ru', '--runtime',
                                type=float,
                                default=70000,
                                help='Total time in seconds as budget to run DEHB')

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
    batch_size = UniformIntegerHyperparameter("batch_size", 128, 512, default_value=256, log=True)
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
    dehb = DEHB(f=run_model, cs=cs, dimensions=dimensions, min_budget=args.min_budget,
                max_budget=args.max_budget, eta=args.eta, output_path=args.output_path, client=None, n_workers=1,
                max_model_params=constraint_model_size)
    traj, runtime, history = dehb.run(total_cost=args.runtime, verbose=args.verbose, seed=123, eta=args.eta,
                                      constraint_precision=constraint_precision)
    name = time.strftime("%x %X %Z", time.localtime(dehb.start))
    name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
    dehb.logger.info("Saving optimisation trace history...")
    with open(os.path.join(args.output_path, "history_{}.pkl".format(name)), "wb") as f:
        pickle.dump(history, f)
