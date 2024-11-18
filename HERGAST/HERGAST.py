import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.nn.conv.rgat_conv import RGATConv

class HERGAST(torch.nn.Module):

    def __init__(self, hidden_dims, att_drop, dim_reduce='PCA'):

        """
        HERGAST model.

        Parameters
        ----------
        hidden_dims
            The dimension of the 1st and 2nd layer of HERGAST encoder.
        att_drop
            Dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training.
        dim_reduce
            Dimentional reduction profile used as the input of HERGAST, can be 'PCA' or 'HVG'. Default is 'PCA'.
        """

        super(HERGAST, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = RGATConv(in_dim, num_hidden, num_relations=2, heads=1, concat=False,
                              dropout=att_drop, bias=False)
        self.conv2 = RGATConv(num_hidden, out_dim, num_relations=2, heads=1, concat=False,
                              dropout=att_drop, bias=False)
        if dim_reduce=='PCA':
            self.decoder = nn.Sequential(
                nn.Linear(out_dim, num_hidden),
                nn.Linear(num_hidden, in_dim),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(out_dim, num_hidden),
                nn.Linear(num_hidden, in_dim),
                nn.ReLU()
            )                  

    def forward(self, features, edge_index, edge_type):

        """
        Run the forward pass of HERGAST.

        Parameters
        ----------
        features
            The input node features. A [num_nodes, node_dim] node feature matrix.
        edge_index
            The edge indices.
        edge_type
            The one-dimensional relation type/index for each edge in `edge_index`.
        """

        h1 = self.conv1(features, edge_index, edge_type)
        h1 = F.elu(h1)
        h2 = self.conv2(h1, edge_index, edge_type)
        h2 = F.elu(h2)
        h3 = self.decoder(h2)
        return h2, h3