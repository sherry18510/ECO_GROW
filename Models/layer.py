# Import PyTorch related libraries
import torch  # PyTorch main library
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Activation functions and other functionalities

class TKGCN(nn.Module):
    def __init__(self, in_features, out_features, top_k=None):
        super(TKGCN, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.top_k = top_k

    def forward(self, x, adj):
        h = self.linear(x)  # Apply linear transformation
        att_weights = adj  # Edge weights
        top_k_value = self.top_k
        
        # Sort edges by weight and pick top-k
        _, indices = torch.sort(att_weights, descending=True, dim=1)
        topk_indices = indices[:, :top_k_value]
        
        # Max pooling of top-k neighbors
        h_aggregated = torch.zeros_like(h)
        for i in range(x.size(0)):
            h_aggregated[i] = torch.max(h[topk_indices[i]], dim=0)[0]
            
        return h_aggregated
    
    
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(input_dim, 1, bias=False)

    def forward(self, input_tensor):
        # Compute attention weights
        attn_weights = F.softmax(self.W(input_tensor), dim=1)  # Apply softmax along the row dimension
        # Perform weighted summation to obtain the attention vector
        output = torch.sum(attn_weights * input_tensor, dim=1)  # Sum over the feature dimension
        return output