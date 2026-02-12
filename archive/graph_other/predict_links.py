# ********************************************************************************
    # 3. BUILD GRAPH
    # ********************************************************************************

    # data = HeteroData()
    # data['argument'].x = arg_embeddings
    # data['issue'].x = issue_embeddings

    # def edge_tensor(edge_list):
    #     return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # data['argument', 'supports', 'issue'].edge_index = edge_tensor(support_edges)
    # data['argument', 'attacks', 'issue'].edge_index = edge_tensor(attack_edges)
    # data['argument', 'attains', 'value'].edge_index = edge_tensor(arg_attains_edges)
    # data['argument', 'constrains', 'value'].edge_index = edge_tensor(arg_constrains_edges)
    # data['issue', 'attains', 'value'].edge_index = edge_tensor(issue_attains_edges)
    # data['issue', 'constrains', 'value'].edge_index = edge_tensor(issue_constrains_edges)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split
import pandas as pd

# --- Config ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# --- Load your graph and embeddings ---
embedding_data = torch.load("training_dataset_education.pt")
arg_embeddings = embedding_data["argument_embeddings"]  # shape: [N_arg, D]
issue_embeddings = embedding_data["issue_embeddings"]   # shape: [N_issue, D]
# Load the saved dictionary
data = torch.load("training_dataset_education.pt")

# Access its contents
arg_indices = data["arg_indices"]        # Tensor of shape [N]
issue_indices = data["issue_indices"]    # Tensor of shape [N]
labels = data["labels"]                  # Tensor of shape [N, 3]


# --- Load dataset ---
df = pd.read_csv("your_dataset.csv")
df['label'] = df['label'].apply(eval) if isinstance(df['label'].iloc[0], str) else df['label']

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# --- Build HeteroData ---
data = HeteroData()
data['argument'].x = arg_embeddings
data['issue'].x = issue_embeddings

# Add training Argument → Issue edges
train_edge_index = torch.tensor([
    train_df['arg_idx'].tolist(),
    train_df['issue_idx'].tolist()
], dtype=torch.long)
data['argument', 'links', 'issue'].edge_index = train_edge_index
data['issue', 'rev_links', 'argument'].edge_index = train_edge_index.flip(0)  # reverse

# --- Model ---
class GNN(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('argument', 'links', 'issue'): SAGEConv((-1, -1), hidden_channels),
            ('issue', 'rev_links', 'argument'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')
        self.lin = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, x_dict, edge_index_dict, arg_idx, issue_idx):
        x_dict = self.conv1(x_dict, edge_index_dict)
        arg_embed = x_dict['argument'][arg_idx]
        issue_embed = x_dict['issue'][issue_idx]
        pair_embed = torch.cat([arg_embed, issue_embed], dim=1)
        return self.lin(pair_embed)

# --- Training setup ---
model = GNN(hidden_channels=64, out_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.KLDivLoss(reduction="batchmean")  # for soft targets

def get_batch_tensor(df_batch):
    return (
        torch.tensor(df_batch['arg_idx'].tolist(), dtype=torch.long).to(device),
        torch.tensor(df_batch['issue_idx'].tolist(), dtype=torch.long).to(device),
        torch.tensor(df_batch['label'].tolist(), dtype=torch.float).to(device)
    )

# --- Train ---
def train():
    model.train()
    total_loss = 0
    for i in range(0, len(train_df), 512):
        batch = train_df.iloc[i:i+512]
        arg_idx, issue_idx, labels = get_batch_tensor(batch)
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict, arg_idx, issue_idx)
        loss = F.kl_div(F.log_softmax(out, dim=1), labels, reduction='batchmean')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_df)

def evaluate(eval_df):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(eval_df), 512):
            batch = eval_df.iloc[i:i+512]
            arg_idx, issue_idx, labels = get_batch_tensor(batch)
            out = model(data.x_dict, data.edge_index_dict, arg_idx, issue_idx)
            preds = F.softmax(out, dim=1)
            loss = F.kl_div(preds.log(), labels, reduction='batchmean')
            total_loss += loss.item()
            total_correct += (preds.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            total += len(batch)
    acc = total_correct / total
    return total_loss / len(eval_df), acc

# --- Run ---
for epoch in range(1, 31):
    loss = train()
    val_loss, val_acc = evaluate(val_df)
    print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# --- Final test ---
test_loss, test_acc = evaluate(test_df)
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

