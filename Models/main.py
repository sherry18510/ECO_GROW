import warnings
warnings.filterwarnings('ignore')  # Ignore possible warnings

from utils import *
from layer import *
from model import *

import argparse

# Set random seed
seed = 42
set_seed(seed)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == "__main__":
    # Create parameter parser
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")

    # Add parameters
    parser.add_argument('--file_name', type=str, default='ECO-GROW', help='Name of the file to train on')
    parser.add_argument('--layer_name', type=str, default='TKGCN', help='Layer name used in the model')
    parser.add_argument('--loss_name', type=str, default='BCE', help='Loss function to use')
    parser.add_argument('--alpha', type=float, default=0.9, help='Alpha parameter')
    parser.add_argument('--beta', type=float, default=0.3, help='Beta parameter') 
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--num_layer', type=int, default=3, help='Number of layers in the model')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--return_embed', type=bool, default=False, help='Return embeddings instead of predictions')
    parser.add_argument('--top_k', type=int, default=5, help='Top K parameter for the model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--barabasi_p', type=int, default=6)

    # Parse command line arguments
    args = parser.parse_args()

    # Call main function
    train_pred(
        file_name=args.file_name,
        layer_name=args.layer_name,
        loss_name=args.loss_name,
        alpha=args.alpha,
        beta=args.beta,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        num_layer=args.num_layer,
        lr=args.lr,
        dropout_rate=args.dropout_rate,
        return_embed=args.return_embed,
        top_k=args.top_k,
        device=args.device,
        barabasi_p=args.barabasi_p
    )