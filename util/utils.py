import random
import numpy as np
import torch
import wandb
import socket
from collections import OrderedDict
import os
import hashlib

def set_random_seed():
    seed_value = 42
    random.seed(seed_value)  # set the seed for Python's built-in random module
    np.random.seed(seed_value)  # set the seed for NumPy's random module
    torch.manual_seed(seed_value)  # set the seed for torch's random module


def set_wandb(args, project_name):
    """ Establish wandb connection to monitor the training """
    config_dict = OrderedDict()
    for key, item in vars(args).items():
        config_dict[key] = item

    run = None
    if args.log_wb and not args.evaluate:
        # wandb.login(key=wandb_key)
        # start a new wandb run to track this script
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            # track hyperparameters and run metadata
            config=config_dict)
    return run, config_dict


def pop_irrelevant_keys(hyperparameters):
    # Create a copy to not modify the original, and remove irrelevant keys if they exist
    hyperparameters_copy = hyperparameters.copy()
    hyperparameters_copy.pop('evaluate', None)
    hyperparameters_copy.pop('job_num', None)

    return hyperparameters_copy


def hash_model_architecture(model, hyperparameters):
    """Compute a deterministic hash for the model architecture and hyperparameters."""
    # Convert model architecture to string
    model_str = str(model)
    hyperparameters_copy = pop_irrelevant_keys(hyperparameters)

    # Convert hyperparameters to string
    hyper_str = str(sorted(hyperparameters_copy.items()))

    # Combine both strings
    full_str = model_str + hyper_str

    # Compute the hash
    return hashlib.md5(full_str.encode()).hexdigest()


def compare_model_and_dict(model, state_dict):
    # Create an iterator over the model's parameters (excluding certain types if desired)
    model_params_iter = iter(model.parameters())

    # Iterate through the state dictionary
    for saved_param_name, saved_param_value in state_dict.items():
        # Exclude certain parameter names or types if desired (e.g., dropout)
        # Modify this condition as needed
        if 'dropout' not in saved_param_name:
            # Get the next parameter from the model
            model_param = next(model_params_iter)

            # Compare the shapes of the saved parameter and the model parameter
            if model_param.shape != saved_param_value.shape:
                print(
                    f"Mismatch in layer {saved_param_name}: model shape {model_param.shape} != saved shape {saved_param_value.shape}")
                return False
    return True


def load_weights_from_dict(model, state_dict):
    # Create an iterator over the saved weights
    saved_weights_iter = iter(state_dict.values())

    # Iterate over the parameters of the model
    for param in model.parameters():
        # Get the next saved weight
        saved_weight = next(saved_weights_iter)

        # Check that the shapes match
        if param.shape != saved_weight.shape:
            raise ValueError(f'Shapes do not match: model shape {param.shape} != saved shape {saved_weight.shape}')

        # Copy the saved weight into the parameter
        param.data = saved_weight


def compare_model_architecture(state_dict_1, state_dict_2):
    """
    Compare the architecture of two model by comparing the shapes of their state dicts.

    Args:
        state_dict_1: The state_dict of the first model.
        state_dict_2: The state_dict of the second model.

    Returns:
        bool: True if both architectures are the same, False otherwise.
    """
    if set(state_dict_1.keys()) != set(state_dict_2.keys()):
        # The parameter names do not match
        return False

    for key in state_dict_1:
        if state_dict_1[key].shape != state_dict_2[key].shape:
            # The shapes of the corresponding parameters do not match
            return False

    return True


def load_dict_to_model(model, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(path, map_location=device)  # ['networks']['feature_extractor']
    if compare_model_and_dict(model, state_dict):
        load_weights_from_dict(model, state_dict)
    elif compare_model_architecture(model.state_dict(), state_dict):
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f'Model and state dictionary do not have the same architecture:\n{path}')


def load_model_by_architecture_hash(model, hyperparameters, cache_dir="."):
    """Load the model from a cache directory based on its architecture hash."""
    model_hash = hash_model_architecture(model, hyperparameters)
    checkpoint_path = os.path.join(cache_dir, f"model_{model_hash}.pt")

    if os.path.exists(checkpoint_path):
        load_dict_to_model(model, checkpoint_path)
        print(f"Model loaded from {checkpoint_path}")
        return True
    else:
        print(f"No saved model found with hash {model_hash} in {cache_dir}")
        return False


def add_parameters(parser):
    parser.add_argument("--job_num", type=int, help="job number", default=0)
    parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
    parser.add_argument("--n_epoch", type=int, help="number of training epoches", default=5000)
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--decay_rate", type=float, help="lr decay factor", default=0.7)
    parser.add_argument("--decay_every", type=int, help="lr decay iterations", default=700)

    parser.add_argument("--log_wb", action="store_true", help="apply W&B logging")

    # model config parameters
    parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default='xyz')
    parser.add_argument("--k_eig", type=int, help="number of eigenvectors used for diffusion", default=500)

    parser.add_argument("--dropout", action="store_true", help="dropout layers in diffusion")
    parser.add_argument("--n_block", type=int, help="number of diffusion blocks", default=2)
    parser.add_argument("--diffusion_hidden", type=int, help="c width diffusion", default=256)
    parser.add_argument("--diffusion_out", type=int, help="diffusion out dimension", default=2)
    parser.add_argument("--base_diffusion", type=float, help="base diffusion time", default=0.002)
    parser.add_argument("--exp_diffusion", type=float, help="exp factor for diffusion time", default=0.8)

    parser.add_argument("--n_layer_net", type=int, help="number of layers in sine activated network", default=6)
    parser.add_argument("--hidden_net", type=int, help="dim of hidden layers sine activated network", default=256)

    parser.add_argument("--base_sigma", type=float, help="base sigma fourier mapping", default=10.0)
    parser.add_argument("--exp_sigma", type=float, help="exp factor for sigma fourier mapping", default=1.2)
    parser.add_argument("--siren_w0", type=float, help="w0 parameter siren", default=220.0)

    parser.add_argument("--n_layer_back", type=int, help="number of layers in backbone", default=2)
    parser.add_argument("--hidden_back", type=int, help="dim of hidden layers backbone network", default=64)
    parser.add_argument("--backbone_activation", type=str, help="backbone activation", default="relu")

    parser.add_argument("--rand_totation", action="store_true", help="random rotation augmentation")


def get_activation_func(activation_name, leaky_slope=None, elu_alpha=None, prelu_init=None):
    if activation_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation_name == "tanh":
        return torch.nn.Tanh()
    elif activation_name == "relu":
        return torch.nn.ReLU(inplace=True)
    elif activation_name == "leaky_relu":
        return torch.nn.LeakyReLU(negative_slope=leaky_slope)
    elif activation_name == "elu":
        return torch.nn.ELU(elu_alpha)
    elif activation_name == "prelu":
        return torch.nn.PReLU(init=prelu_init)
    elif activation_name == "siren":
        return "siren"  # SineLayer()
    elif activation_name == "sinh":
        return "sinh"
    elif activation_name == "None":
        return None
    else:
        raise ValueError(f"Activation function {activation_name} not found")


def set_config(config, args):
    config["encoding"]["n_block"] = args.n_block
    config["encoding"]["diffusion_hidden"] = args.diffusion_hidden
    config["encoding"]["base_diffusion"] = args.base_diffusion
    config["encoding"]["exp_diffusion"] = args.exp_diffusion
    config["encoding"]["vertex_feat_dim"] = args.diffusion_out

    config["encoding"]["base_sigma"] = args.base_sigma
    config["encoding"]["exp_sigma"] = args.exp_sigma

    config["SIREN"]["dims"] = [args.hidden_net] * args.n_layer_net
    config["SIREN"]["w0"] = args.siren_w0

    config["Backbone"]["activation"] = get_activation_func(args.backbone_activation)
    config["Backbone"]["dims"] = [args.hidden_back] * args.n_layer_back


def hash_model_architecture(model, hyperparameters):
    """Compute a deterministic hash for the model architecture and hyperparameters."""
    # Convert model architecture to string
    model_str = str(model)
    hyperparameters.pop('evaluate', None)
    hyperparameters.pop('job_num', None)
    # Convert hyperparameters to string
    hyper_str = str(sorted(hyperparameters.items()))
    # Combine both strings
    full_str = model_str + hyper_str
    # Compute the hash
    return hashlib.md5(full_str.encode()).hexdigest()
