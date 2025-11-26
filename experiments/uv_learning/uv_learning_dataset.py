import trimesh
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import diffusion_net


class UVLearningDataset(Dataset):
    """ Dataset for learning neural field representing UV function on a mesh """
    def __init__(self, root_dir, mesh_name, train, k_eig=128, use_cache=True, op_cache_dir=None):
        self.train = train  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.mesh_name = mesh_name
        # store in memory
        self.verts_list = []
        self.faces_list = []

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, f"train.pt")
            test_cache = os.path.join(self.cache_dir, f"test.pt")
            load_cache = train_cache if self.train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.f_texture_uv_list = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        print("loading {} meshes".format(1))

        def load_mesh(obj_filename):
            mesh = trimesh.load(obj_filename, process=False)
            verts = torch.tensor(np.ascontiguousarray(mesh.vertices)).float()
            verts = diffusion_net.geometry.normalize_positions(verts)
            self.verts_list = [verts]
            self.faces_list = [torch.tensor(np.ascontiguousarray(mesh.faces))]
            return mesh

        mesh_dirpath = os.path.join(self.root_dir, "meshes")
        obj_filename = os.path.join(mesh_dirpath, f'{self.mesh_name}.obj')
        mesh = load_mesh(obj_filename)
        uv_coordinates = np.asarray(mesh.visual.uv)  # extract uv coordinates
        self.f_texture_uv_list = [uv_coordinates]

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.f_texture_uv_list), load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.f_texture_uv_list[idx]
