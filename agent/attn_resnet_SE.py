import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y.expand_as(x)

class LightweightResidualBlock(nn.Module):
    """A simple lightweight residual block with SE attention mechanism."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(LightweightResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Adding SE Block after the first convolution and before the second one
        self.se_block = SEBlock(out_channels)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.se_block(out)  # Applying SE Block
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class OP_AttnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob, num_resnet_blocks=1):
        super(OP_AttnBlock, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.alpha = 0.2

        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.dropout = nn.Dropout(p=dropout_prob)

        # Add ResNet blocks
        self.resnet_blocks = nn.Sequential(*[
            LightweightResidualBlock(output_dim, output_dim) for _ in range(num_resnet_blocks)
        ])

    def forward(self, h, op_mask):
        Wh = torch.matmul(h, self.W)
        sz_b, N, _ = Wh.size()

        Wh_concat = torch.stack([Wh.roll(1, dims=1), Wh, Wh.roll(-1, dims=1)], dim=-2)

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        Wh2_concat = torch.stack([Wh2.roll(1, dims=1), Wh2, Wh2.roll(-1, dims=1)], dim=-1)

        e = Wh1.unsqueeze(-1) + Wh2_concat
        e = self.leaky_relu(e)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(op_mask.unsqueeze(-2) > 0, zero_vec, e)

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_new = torch.matmul(attention, Wh_concat).squeeze(-2)

        # Apply ResNet blocks after attention mechanism
        x_processed = self.resnet_blocks(h_new.permute(0, 2, 1))  # Permute for Conv1d compatibility
        x_final = x_processed.permute(0, 2, 1)  # Permute back

        return x_final


class OP_MultiHeadAttnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob, num_heads, activation, concat=True):
        super(OP_MultiHeadAttnBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.num_heads = num_heads
        self.concat = concat
        self.activation = activation
        self.attentions = [
            OP_AttnBlock(input_dim, output_dim, dropout_prob) for
            _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, h, op_mask):

        h = self.dropout(h)

        h_heads = [att(h, op_mask) for att in self.attentions]

        if self.concat:
            h = torch.cat(h_heads, dim=-1)
        else:

            h = torch.stack(h_heads, dim=-1)

            h = h.mean(dim=-1)

        return h if self.activation is None else self.activation(h)


class MCH_AttnBlock(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim, dropout_prob, num_resnet_blocks=1):
        super(MCH_AttnBlock, self).__init__()
        self.node_in_features = node_input_dim
        self.edge_in_features = edge_input_dim
        self.out_features = output_dim
        self.alpha = 0.2
        self.W = nn.Parameter(torch.empty(size=(node_input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.W_edge = nn.Parameter(torch.empty(size=(edge_input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W_edge.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(3 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.dropout = nn.Dropout(p=dropout_prob)

        # Add ResNet blocks
        self.resnet_blocks = nn.Sequential(*[
            LightweightResidualBlock(output_dim, output_dim) for _ in range(num_resnet_blocks)
        ])

    def forward(self, h, mch_mask, comp_val):

        Wh = torch.matmul(h, self.W)
        W_edge = torch.matmul(comp_val, self.W_edge)

        # compute attention matrix

        e = self.get_attention_coef(Wh, W_edge)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(mch_mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)

        # Pass through ResNet blocks
        x_prime_permuted = h_prime.permute(0, 2, 1)  # Reshape for 1D convolution
        for res_block in self.resnet_blocks:
            x_prime_permuted = res_block(x_prime_permuted)
        x_final = x_prime_permuted.permute(0, 2, 1)  # Reshape back

        return x_final

    def get_attention_coef(self, Wh, W_edge):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:2 * self.out_features, :])
        edge_feas = torch.matmul(W_edge, self.a[2 * self.out_features:, :])

        e = Wh1 + Wh2.transpose(-1, -2) + edge_feas.squeeze(-1)

        return self.leaky_relu(e)


class MCH_MultiHeadAttnBlock(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim, dropout_prob, num_heads, activation, concat=True):

        super(MCH_MultiHeadAttnBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.concat = concat
        self.activation = activation
        self.num_heads = num_heads

        self.attentions = [MCH_AttnBlock
                           (node_input_dim, edge_input_dim, output_dim, dropout_prob) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, h, mch_mask, comp_val):

        h = self.dropout(h)
        h_heads = [att(h, mch_mask, comp_val) for att in self.attentions]
        if self.concat:

            h = torch.cat(h_heads, dim=-1)
        else:

            h = torch.stack(h_heads, dim=-1)
            h = h.mean(dim=-1)

        return h if self.activation is None else self.activation(h)
