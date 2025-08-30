# Import PyTorch related libraries
import torch  # PyTorch main library
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Activation functions and other functionalities
from torch_geometric.nn import GATConv # Different layers for GNN
import torch_geometric.transforms as T  # Graph data transformations
from torch_geometric.utils import to_dense_adj  # Convert sparse graph to dense adjacency matrix

from layer import *

class ECO_GROW(torch.nn.Module):
    def __init__(self, 
                 layer_name='TKGCN', 
                 graph_cnt=5, 
                 in_channels=2, 
                 hidden_channels=128, 
                 rnn_hidden_size=16, 
                 out_channels=16, 
                 num_layers=3, 
                 with_empty=False,
                 lstm_k=15,
                 top_k=5,
                 node_feat_dim=3,
                 dropout_rate=0.15,
                 device='cuda'
                ):
        super().__init__()
        # Initialize model parameters
        self.graph_cnt = graph_cnt
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.with_empty = with_empty
        self.lstm_k = lstm_k
        self.top_k = torch.nn.Parameter(torch.randint(1, 297, (self.graph_cnt + 1,)).float(), requires_grad=True) 
        self.node_feat_dim = node_feat_dim
        self.device = device
        
        
        # Initialize module lists
        self.conv1_list = torch.nn.ModuleList()
        self.conv2_list = torch.nn.ModuleList()
        self.batch_norm_list = torch.nn.ModuleList()
        
        # LSTM for dynamic graphs
        self.lstm = torch.nn.LSTM(out_channels, rnn_hidden_size, num_layers, batch_first=True)
        
        # static graphs
        if self.with_empty:
            for _ in range(self.graph_cnt - 1):
                self.conv1_list.append(eval(layer_name)(in_channels, hidden_channels))
                self.conv2_list.append(eval(layer_name)(hidden_channels, out_channels))
                self.batch_norm_list.append(torch.nn.BatchNorm1d(out_channels))
            # empty graph
            self.conv1_list.append(GATConv(in_channels, hidden_channels))
            self.conv2_list.append(GATConv(hidden_channels, out_channels))
            self.batch_norm_list.append(torch.nn.BatchNorm1d(out_channels))
        else:
            for _ in range(self.graph_cnt):
                self.conv1_list.append(eval(layer_name)(in_channels, hidden_channels))
                self.conv2_list.append(eval(layer_name)(hidden_channels, out_channels))
                self.batch_norm_list.append(torch.nn.BatchNorm1d(out_channels))
                
        # dynamic graphs
        for _ in range(self.lstm_k):
            self.conv1_list.append(eval(layer_name)(in_channels, hidden_channels))
            self.conv2_list.append(eval(layer_name)(hidden_channels, out_channels))
            self.batch_norm_list.append(torch.nn.BatchNorm1d(out_channels))

        self.att_layer = AttentionLayer(out_channels * self.graph_cnt + self.rnn_hidden_size)
        
        self.linear0 = nn.Linear(node_feat_dim, in_channels)
        self.linear1 = nn.Linear(in_channels * graph_cnt + rnn_hidden_size + node_feat_dim, out_channels * 2)
        self.linear2 = nn.Linear(out_channels * 2, out_channels)
        
        self.graph_scores = nn.Parameter(torch.randn(graph_cnt, 1), requires_grad=True)
        #self.topk_list = [self.top_k for _ in range(graph_cnt)]
        self.reg_linear1 = nn.Linear(out_channels, 1)
        self.reg_linear2 = nn.Linear(out_channels, 1)
        self.reg_linear3 = nn.Linear(out_channels, 1) 

        self.dropout = torch.nn.Dropout(dropout_rate)  # Initialize Dropout layer
        
    def encode(self, x, edge_index, edge_weight, conv1, conv2, batch_norm, top_k):
        if top_k > 0:
            adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight, max_num_nodes=297)[0]
            x = conv1(x, adj).relu()
            x = conv2(x, adj)
            x = self.dropout(x)  # Dropout after first convolution
            x = batch_norm(x) 
        else:  # for empty graph
            x = conv1(x, edge_index).relu()
            x = conv2(x, edge_index)
            x = self.dropout(x)  # Dropout after first convolution
            x = batch_norm(x) 
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return torch.sigmoid(self.reg_linear1(z)), r, torch.sigmoid(self.reg_linear3(z))

    def forward(self, node_feature, graphs, time_graphs, edge_label_index):
        gnn_outputs = []
        weighted_outputs = []
        gnn4time_outputs = []
        softmax_scores = F.softmax(self.graph_scores, dim=0)
        
        # ---------------------- Static Embedding ---------------------- #
        if self.with_empty:
            for i, graph in enumerate(graphs):
                x = graph.x
                x = self.linear0(x)
                residual = x  # Save input feature as residual
                edge_index = graph.edge_index
                
                top_k_value = torch.round(self.top_k[i]).int() 
                top_k_value = torch.clamp(top_k_value, min=1, max=297) 
                
                if i < self.graph_cnt - 1:
                    edge_weight = graph.edge_weight
                    z = self.encode(x, edge_index, edge_weight, self.conv1_list[i], self.conv2_list[i], self.batch_norm_list[i], top_k=top_k_value)
                else:  # empty graph
                    z = self.encode(x, edge_index, edge_weight, self.conv1_list[i], self.conv2_list[i], self.batch_norm_list[i], top_k=0)
                
                gnn_outputs.append(z + residual)  # Residual connection
                weighted_output = softmax_scores[i] * z
                weighted_outputs.append(weighted_output)
                
        else:  # without the empty graph
            for i, graph in enumerate(graphs):
                x = graph.x
                x = self.linear0(x)
                edge_index = graph.edge_index
                edge_weight = graph.edge_weight
                residual = x  # Save input feature as residual
                
                top_k_value = torch.round(self.top_k[i]).int() 
                top_k_value = torch.clamp(top_k_value, min=1, max=297) 
                
                z = self.encode(x, edge_index, edge_weight, self.conv1_list[i], self.conv2_list[i], self.batch_norm_list[i], top_k=top_k_value)
                gnn4time_outputs.append(z + residual)  # Residual connection
                weighted_output = softmax_scores[i] * z
                weighted_outputs.append(weighted_output)

        combined_output = torch.cat(weighted_outputs, dim=1)
        
        # ---------------------- Dynamic Embedding ---------------------- #
        for i, graph in enumerate(time_graphs):
            x = graph.x
            x = self.linear0(x)
            edge_index = graph.edge_index
            edge_weight = graph.edge_weight
            
            top_k_value = torch.round(self.top_k[self.graph_cnt]).int()  
            top_k_value = torch.clamp(top_k_value, min=1, max=297) 
            
            z = self.encode(x, edge_index, edge_weight, 
                            self.conv1_list[self.graph_cnt + i], self.conv2_list[self.graph_cnt + i], 
                            self.batch_norm_list[self.graph_cnt + i], top_k=top_k_value)
            gnn4time_outputs.append(z)

        h0 = torch.zeros(self.num_layers, 297, self.rnn_hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, 297, self.rnn_hidden_size).to(self.device)
        gnn4time_outputs = torch.stack(gnn4time_outputs, dim=1)
        
        # Process stacked outputs through LSTM
        rnn_out, _ = self.lstm(gnn4time_outputs, (h0, c0))
        rnn_out = rnn_out[:, -1, :]
        
        # ---------------------- Concat ---------------------- #
        out = self.linear1(torch.cat([combined_output, rnn_out, node_feature], dim=1)).relu()
        out = self.linear2(out)
        
        return self.decode(out, edge_label_index), out