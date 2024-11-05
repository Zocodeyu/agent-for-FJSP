
from agent.Learnable_MLP import *
from agent.O_M_AttentionNetwork import *
import torch
import torch.nn as nn
import torch.nn.functional as F



class LearnableNetwork(nn.Module):
    def __init__(self, config):

        super(LearnableNetwork, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = O_M_AttentionNetwork(config).to(
            device)
        self.actor = LearnableActor(config.num_mlp_layers_actor, 4 * self.embedding_output_dim + self.pair_input_dim,
                           config.hidden_dim_actor, 1).to(device)
        self.critic = LearnableCritic(config.num_mlp_layers_critic, 2 * self.embedding_output_dim, config.hidden_dim_critic,
                             1).to(device)


    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):


        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx)

        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)

        candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                       Fea_Gm_input, fea_pairs), dim=-1)

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')

        pi = F.softmax(candidate_scores, dim=1)


        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)

        return pi, v
