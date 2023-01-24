import copy

import torch
import torch.nn as nn
from typing import Tuple
from functorch import vmap
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src import FEATURE_DIM, RADIUS, splev, N_CLASSES, N_CTPS, P, evaluate, compute_traj, modified_eval, generate_game
import time


class Agent:

    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """
        self.classification_Model = ANNModel(FEATURE_DIM, 128, N_CLASSES)  # 调用模型Model
        self.classification_Model.load_state_dict(torch.load("./data/model_parameter.pkl"))  # 加载模型参数

        # TODO: prepare your agent here

    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the parameters required to fire a projectile. 
        
        Args:
            target_pos: x-y positions of shape `(N, 2)` where `N` is the number of targets. 
            target_features: features of shape `(N, d)`.
            class_scores: scores associated with each class of targets. `(K,)` where `K` is the number of classes.
        Return: Tensor of shape `(N_CTPS-2, 2)`
            the second to the second last control points
        """
        assert len(target_pos) == len(target_features)
        N = len(target_pos)
        target_class = torch.zeros(N)
        soft = nn.Softmax(dim=0)
        for i in range(N):
            cla_P = self.classification_Model.forward(target_features[i])
            target_class[i] = soft(cla_P).argmax(dim=0)

        target_class = target_class.long()

        # TODO: compute the firing speed and angle that would give the best score.
        # Example: return a random configuration

        ctps_inter_best = torch.randn((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
        best_score = evaluate(compute_traj(ctps_inter_best), target_pos, class_scores[target_class], RADIUS)

        vmap_func1 = vmap(compute_traj, in_dims=0, out_dims=0)
        vmap_func2 = vmap(evaluate, in_dims=(0, None, None, None), out_dims=0)

        start=time.time()
        while time.time()-start<0.27:
            a = torch.randn((100, N_CTPS - 2, 2))
            b = torch.mul(a, torch.tensor([N_CTPS - 2, 2.]))
            ctps_inter_batch = b + torch.tensor([1., -1.])
            c_t = vmap_func1(ctps_inter_batch)
            cur_score_batch = vmap_func2(c_t, target_pos, class_scores[target_class], RADIUS)
            idx = torch.argmax(cur_score_batch)
            ctps_inter_cur = ctps_inter_batch[idx]
            cur_score = evaluate(compute_traj(ctps_inter_cur), target_pos, class_scores[target_class], RADIUS)

            if cur_score > best_score:
                ctps_inter_best = ctps_inter_cur
                best_score = cur_score

            ctps_inter_copy = copy.deepcopy(ctps_inter_cur)
            ctps_inter_copy = ctps_inter_copy.expand(300, 3, 2)

            tmp_rand = torch.rand((300, N_CTPS - 2, 2))
            ctps_inter_copy = ctps_inter_copy + tmp_rand
            ctps_inter_copy[:150] = ctps_inter_copy[:150] + torch.tensor([-0.5, -0.5])
            ctps_inter_copy[151:200] = ctps_inter_copy[151:200] + torch.tensor([-1, -0.5])
            ctps_inter_copy[201:250] = ctps_inter_copy[201:250] + torch.tensor([0, -0.5])
            ctps_inter_copy[251:] = ctps_inter_copy[251:] + torch.tensor([1, -0.5])

            c_t_copy = vmap_func1(ctps_inter_copy)
            cur_score_copy = vmap_func2(c_t_copy, target_pos, class_scores[target_class], RADIUS)
            idx = torch.argmax(cur_score_copy)
            ctps_inter_copy = ctps_inter_copy[idx]
            cur_score_copy = evaluate(compute_traj(ctps_inter_copy), target_pos, class_scores[target_class], RADIUS)

            if cur_score_copy > best_score:
                ctps_inter_best = ctps_inter_copy
                best_score = cur_score_copy

        return ctps_inter_best


class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)

        out = self.relu2(out)
        out = self.fc3(out)
        return out

