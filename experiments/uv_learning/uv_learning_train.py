import torch
from torch.utils.data import DataLoader
import gc
import json
import os
import argparse
from tqdm import tqdm
import wandb

import diffusion_net
from diffusion_net.utils import toNP
import util.utils as utils
from experiments.uv_learning.uv_learning_model import UVLearningModel
from experiments.uv_learning.uv_learning_dataset import UVLearningDataset
from experiments.uv_learning.uv_learning_utils import visualize_uv


# === Options
utils.set_random_seed()
# Parse a few args
parser = argparse.ArgumentParser()
utils.add_parameters(parser)
args = parser.parse_args()
args.dataset = "dennis"

# wandblog
run, config_dict = utils.set_wandb(args, "MDNF_UVLearning")
# system things
device = torch.device('cuda:0')
dtype = torch.float32

# model
input_features = args.input_features  # one of ['xyz', 'hks']
k_eig = args.k_eig

# training settings
train = not args.evaluate
n_epoch = args.n_epoch
lr = args.lr
decay_every = args.decay_every
decay_rate = args.decay_rate
augment_random_rotate = (input_features == 'xyz')

# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data", "op_cache")
model_save_path = os.path.join(base_path, "data", "saved_models/")
dataset_path = os.path.join(base_path, "data", f"{args.dataset}")

# === Load datasets
# Load the test dataset
test_dataset = UVLearningDataset(dataset_path, args.dataset, train=False,
                                 k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)
# Load the train dataset
if train:
    train_dataset = UVLearningDataset(dataset_path, args.dataset, train=True,
                                      k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None)

# === Create the model
C_in = {'xyz': 3, 'hks': 16}[input_features]  # dimension of input features
with open(os.path.join(base_path, "config", "config.json")) as config_file:
    config = json.load(config_file)
utils.set_config(config, args)

model = UVLearningModel(config,
                        n_input_dims=C_in,
                        out_dims=2,  # uv coordinates
                        diffusion_dropout=args.dropout)

model = model.to(device)
model_hash = utils.hash_model_architecture(model, config_dict)  # compute model hash for ckpt name
if args.log_wb and train:
    wandb.config.update({'model_hash': model_hash}, allow_val_change=True)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(model_save_path))
    ret = utils.load_model_by_architecture_hash(model, hyperparameters=config_dict, cache_dir=model_save_path)
    if not ret:
        raise ValueError(f"The corresponding model was not found")

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = torch.nn.MSELoss()
texture_filename = os.path.join(dataset_path, 'meshes', f'{args.dataset}_texture.png')


def forward_pass(data):
    # Get data
    verts, faces, frames, mass, L, evals, evecs, gradX, gradY, f_uv = [d.to(device) for d in data]
    # Randomly rotate positions
    if args.rand_totation and augment_random_rotate:
        verts = diffusion_net.utils.random_rotate_points(verts)
    # Construct features
    if input_features == 'xyz':
        features = verts
    elif input_features == 'hks':
        features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

    # Apply the model and evaluate loss
    model_data = {"verts": verts, "faces": faces, 'feats': features, 'mass': mass, 'L': L,
                 'evals': evals, 'evecs': evecs, 'gradX': gradX, 'gradY': gradY}
    preds = model(model_data).float()

    loss = loss_function(preds, f_uv.float())
    # -------------------------------------------------------------------------------------------------
    if not train:
        visualize_uv(toNP(f_uv), toNP(preds), toNP(faces))

    return loss


def train_epoch(epoch):
    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    for data in tqdm(train_loader):
        loss = forward_pass(data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)

    if args.log_wb and train:  # Logging epoch metrics
        wandb.log({"Train_Epoch_Loss": epoch_loss})
    if epoch % 1000 == 0:
        p = os.path.join(model_save_path, f"model_{model_hash}.pt")
        print(" ==> saving last model to " + p)
        torch.save(model.state_dict(), p)

    return epoch_loss


def test():
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            loss = forward_pass(data)
            running_loss += loss.item()

    # Logging epoch metrics
    epoch_loss = running_loss / len(test_loader)
    if args.log_wb and train:
        wandb.log({"Test_Epoch_Loss": epoch_loss})
    return epoch_loss


if train:
    os.makedirs(model_save_path, exist_ok=True)
    print("Training...")
    for epoch in range(n_epoch):
        train_loss = train_epoch(epoch)
        test_loss = test()
        torch.cuda.empty_cache()
        gc.collect()
        print("Epoch {} - Train Loss: {:06.5f}  Test Loss: {:06.5f}".format(epoch, train_loss, test_loss))

    p = os.path.join(model_save_path, f"model_{model_hash}.pt")
    print(" ==> saving last model to " + p)
    torch.save(model.state_dict(), p)


# Test
test_loss = test()
print("Test loss: {:06.5f}".format(test_loss))
