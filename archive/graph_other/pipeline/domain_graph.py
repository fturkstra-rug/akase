import json
from tqdm import tqdm
import pandas as pd
import argparse
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch_geometric.data import HeteroData
# from torch_geometric.nn import HeteroConv, SAGEConv
# from sentence_transformers import SentenceTransformer
import random
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from constants import DOMAINS, HUMAN_VALUES
from create_embeddings import embed

# For reproducibility
random.seed(42)
# torch.manual_seed(42)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
    

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", help="Extract all nodes related to this domain.", required=True, type=str, choices=DOMAINS)
    parser.add_argument("-i", "--input_file", help="File with preprocessed scraping data.", required=True, type=str)
    parser.add_argument("-v", "--values_file", help="File with predicted human values", required=False, type=str)
    return parser.parse_args()


def main():
    args = create_arg_parser()

    # Read in user-provided data
    with open(args.input_file, "r") as f:
        seed_data = json.load(f)

    values_df = pd.read_csv(args.values_file, sep="\t")
    subgraph = args.domain is not None

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    argument_texts = []
    issue_texts = []
    value_texts = []

    # Edges from argument to issue with labels
    support_edges = []   # label = 1
    attack_edges = []    # label = 2

    # argument --> human_value
    arg_attains_edges = []
    arg_constrains_edges = []

    # issue --> human_value
    issue_attains_edges = []
    issue_constrains_edges = []

    # Map uuids to indices
    uuid_to_issue_idx = {}
    uuid_to_arg_idx = {}
    arg_to_idx = {}

    print("Constructing graph...")
    for row in tqdm(seed_data):

        # Extract only issues related to domain
        if subgraph:
            domain = row.get("domain", [])
            if not (args.domain in domain):
                continue
        
        # ********************************************************************************
        # Create Issue Nodes
        # ********************************************************************************
        issue_text = row.get("issue", "")
        issue_uuid = row.get("uuid", "")
        if not issue_text or not issue_uuid:
            continue

        issue_idx = len(issue_texts)
        uuid_to_issue_idx[issue_uuid] = issue_idx
        issue_texts.append(issue_text)

        # ********************************************************************************
        # Create Argument Nodes / Argument --> Issue Edges
        # ********************************************************************************

        arguments = row.get("arguments", {})
        if not arguments:
            continue

        for stance in ["pro", "con"]:
            stance_arguments = arguments.get(f"{stance}_arguments", [])
            for i, arg in enumerate(stance_arguments):

                # If the argument does not already exist
                if not (arg in arg_to_idx):
                    arg_idx = len(argument_texts)
                    arg_to_idx[arg] = arg_idx
                    argument_texts.append(arg)
                
                # Add edges
                arg_idx = arg_to_idx[arg]
                if stance == "pro":
                    support_edges.append((arg_idx, issue_idx))
                else:
                    attack_edges.append((arg_idx, issue_idx))
                
                # Map idx to uuid
                arg_uuid = f"{issue_uuid}-{stance}-{i + 1}"
                uuid_to_arg_idx[arg_uuid] = arg_idx

    # ********************************************************************************
    # Create Value Nodes / Issue --> Value & Argument --> Value Edges
    # ********************************************************************************

    # Collapse all sentence predictions into one.
    # Text-ID,  Sentence-ID,    Val1    Val2    Val3
    # example   1               0       1       0
    # example   2               0       0       0
    # example   3               1       1       0

    # Becomes

    # Text-ID,  Sentence-ID,    Val1    Val2    Val3
    # example   1               1       1       0
    value_columns = [f"{value} {attainment}" for attainment in ["constrained", "attained"] for value in HUMAN_VALUES]
    grouped = values_df.groupby("Text-ID")[value_columns].max()

    # print(grouped.sum(axis=1).value_counts())

    def add_value_edge(node_idx, value_idx, attainment_type):
        if attainment_type == "attained":
            arg_attains_edges.append((node_idx, value_idx))
        else:
            arg_constrains_edges.append((node_idx, value_idx))

    for value in HUMAN_VALUES:
        value_idx = len(value_texts)
        value_texts.append(value)

        for attainment_type in ["attained", "constrained"]:
            column_name = value + " " + attainment_type
            value_df = grouped[grouped[column_name] == 1]

            if value_df.empty:
                continue

            # Find the nodes that belong to these rows
            for uuid in value_df.index:

                # If it is an issue uuid
                if max(uuid.rfind("-pro"), uuid.rfind("-con")) < 0:
                    issue_idx = uuid_to_issue_idx[uuid] 
                    add_value_edge(issue_idx, value_idx, attainment_type)
                # If is an argument uuid
                else:
                    arg_idx = uuid_to_arg_idx[uuid]
                    add_value_edge(arg_idx, value_idx, attainment_type)


    # random_issue = 14
    # issue_text = issue_texts[14]
    # print(issue_text)

    # con_arguments = [pair[0] for pair in attack_edges if pair[1] == random_issue]
    # pro_arguments = [pair[0] for pair in support_edges if pair[1] == random_issue]
    
    # pro_attain_values = set()
    # pro_constrain_values = set()
    # for arg_idx in pro_arguments:
    #     for pair in arg_attains_edges:
    #         if pair[0] == arg_idx:
    #             pro_attain_values.add(value_texts[pair[1]])
    #     for pair in arg_constrains_edges:
    #         if pair[0] == arg_idx:
    #             pro_constrain_values.add(value_texts[pair[1]])

    # con_attain_values = set()
    # con_constrain_values = set()
    # for arg_idx in con_arguments:
    #     for pair in arg_attains_edges:
    #         if pair[0] == arg_idx:
    #             con_attain_values.add(value_texts[pair[1]])
    #     for pair in arg_constrains_edges:
    #         if pair[0] == arg_idx:
    #             con_constrain_values.add(value_texts[pair[1]])
    
    
    print(len(arg_attains_edges))
    print(len(arg_constrains_edges))
    print(len(support_edges))
    print(len(attack_edges))
    print(len(argument_texts))
    print(len(issue_texts))
    print(len(value_texts))

    exit()

    # ********************************************************************************
    # Encode texts into embeddings
    # ********************************************************************************

    print("Fetching embeddings...")
    try:
        issue_embeddings = torch.load('issue_embeddings.pt')
        arg_embeddings = torch.load('arg_embeddings.pt')
    except Exception:
        print("Could not load existing embeddings, generating new ones.")
        issue_embeddings = embed(issue_texts)
        arg_embeddings = embed(argument_texts)

        torch.save(issue_embeddings, 'issue_embeddings.pt')
        torch.save(arg_embeddings, 'arg_embeddings.pt')
        print("Embeddings saved.")

    # ********************************************************************************
    # Negative sampling
    # ********************************************************************************

    print("Negative sampling...")
    positive_samples = support_edges + attack_edges
    negative_samples = []
    num_samples = int(0.5 * len(positive_samples)) # equal number of samples per class (no-link, supports, attacks)

    max_iterations = 100_000 # Prevent infinite loop
    current_iteration = 0

    while len(negative_samples) < num_samples or current_iteration >= max_iterations:
        arg_idx = random.randint(0, len(argument_texts) - 1)
        issue_idx = random.randint(0, len(issue_texts) - 1)
        pair = (arg_idx, issue_idx)

        if pair not in positive_samples and pair not in negative_samples:
            negative_samples.append(pair)

        current_iteration += 1

    # ********************************************************************************
    # Compute similarity to set label smoothing parameters
    # ********************************************************************************

    # Extract the relevant embeddings (N = the number of negative samples)
    # Normalize the embeddings, important before calculating cosine similarity.
    # Lastly, clip for smoothing purposes (always between 0.1 and 0.9).
    # This way, even if semantic similarity is 0 or 1, there is still a chance a link does (not) exist

    # print("Calculating cosine similarity scores...")
    negative_arg_indices = torch.tensor([pair[0] for pair in negative_samples])
    negative_issue_indices = torch.tensor([pair[1] for pair in negative_samples])
    negative_labels = [[1.0, 0.0, 0.0] for _ in negative_samples]
    negative_labels_tensor = torch.tensor(negative_labels, dtype=torch.float32)

    # arg_vecs = arg_embeddings[negative_arg_indices]      # Shape: [N, emb_dim]
    # issue_vecs = issue_embeddings[negative_issue_indices]  # Shape: [N, emb_dim]

    # arg_norm = F.normalize(arg_vecs, p=2, dim=1)
    # issue_norm = F.normalize(issue_vecs, p=2, dim=1)

    # cos_sim = (arg_norm * issue_norm).sum(dim=1)  # Shape: [N]

    # clipped_sim = torch.clamp(cos_sim, min=0.1, max=0.9)  # Shape: [N]

    # # With the similarity scores, we can calculate the smoothing parameters.
    # # We use existing ratios found in the data.
    # supports_ratio = len(support_edges) / len(positive_samples)
    # attacks_ratio = len(attack_edges) / len(positive_samples)

    # p_no_link = 1 - clipped_sim
    # p_supports = clipped_sim * supports_ratio
    # p_attacks = clipped_sim * attacks_ratio

    # negative_labels_tensor = torch.stack([p_no_link, p_supports, p_attacks], dim=1)  # Shape: [N, 3]

    # ********************************************************************************
    # Prepare training set
    # ********************************************************************************
    
    print("Preparing dataset...")
    # Extract hard labels for all the positive samples [no-link, supports, attacks]
    support_labels = [[0.0, 1.0, 0.0] for _ in support_edges]
    attack_labels = [[0.0, 0.0, 1.0] for _ in attack_edges]
    positive_labels = support_labels + attack_labels

    positive_arg_indices = torch.tensor([a for a, _ in positive_samples], dtype=torch.long)
    positive_issue_indices = torch.tensor([i for _, i in positive_samples], dtype=torch.long)
    positive_labels_tensor = torch.tensor(positive_labels, dtype=torch.float32)

    arg_indices = torch.cat([positive_arg_indices, negative_arg_indices], dim=0)
    issue_indices = torch.cat([positive_issue_indices, negative_issue_indices], dim=0)
    labels = torch.cat([positive_labels_tensor, negative_labels_tensor], dim=0)


    print(arg_indices.shape)   # [total_samples]
    print(labels.shape)        # [total_samples, 3]

    training_data = {
        "arg_indices": arg_indices,      # [N]
        "issue_indices": issue_indices,  # [N]
        "labels": labels                 # [N, 3]
    }


    all_indices = np.arange(len(arg_indices))
    # Stratifying on soft labels means there are no no-link predictions. 
    # If semantic similarity is <66.67, p_no_link can become the argmax. Currently there are 0 samples.
    # labels_for_stratify = labels.argmax(dim=1).cpu().numpy()

    # First split: train+val and test
    train_val_idx, test_idx = train_test_split(
        all_indices, test_size=0.2, random_state=42, stratify=labels
    )

    train_val_labels = labels[train_val_idx]

    # Then split train+val into train and val
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.1, random_state=42, stratify=train_val_labels
    )    

    output_file = "training_dataset_education"
    torch.save(training_data, output_file + '.pt')
    print(f"Saved to {output_file}.pt")

    df = pd.DataFrame({
        "arg_idx": arg_indices.cpu().numpy(),
        "issue_idx": issue_indices.cpu().numpy(),
        "p_no_link": labels[:, 0].cpu().numpy(),
        "p_supports": labels[:, 1].cpu().numpy(),
        "p_attacks": labels[:, 2].cpu().numpy(),
    })

    df.to_csv(output_file + ".csv", index=False)
    print(f"Saved to {output_file}.csv")

    
    # ********************************************************************************
    # FORMAT DATA AND LINKS
    # ********************************************************************************
    print("Creating heterodata...")
    data = HeteroData()

    # Node features
    data['argument'].x = arg_embeddings
    data['issue'].x = issue_embeddings
    data['value'].x = torch.eye(19)  # Assuming 19 values and one-hot encode (can use embeddings later)

    def edge_list_to_tensor(edge_list):
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Argument–Issue edges for message passing (label-agnostic)
    arg_issue_tensor = edge_list_to_tensor(positive_samples)

    data['argument', 'links', 'issue'].edge_index = arg_issue_tensor
    data['issue', 'rev_links', 'argument'].edge_index = arg_issue_tensor.flip(0)

    # Argument–Value
    data['argument', 'attains', 'value'].edge_index = edge_list_to_tensor(arg_attains_edges)
    data['value', 'rev_attains', 'argument'].edge_index = edge_list_to_tensor(arg_attains_edges).flip(0)

    data['argument', 'constrains', 'value'].edge_index = edge_list_to_tensor(arg_constrains_edges)
    data['value', 'rev_constrains', 'argument'].edge_index = edge_list_to_tensor(arg_constrains_edges).flip(0)

    # Issue–Value
    data['issue', 'attains', 'value'].edge_index = edge_list_to_tensor(issue_attains_edges)
    data['value', 'rev_attains_issue', 'issue'].edge_index = edge_list_to_tensor(issue_attains_edges).flip(0)

    data['issue', 'constrains', 'value'].edge_index = edge_list_to_tensor(issue_constrains_edges)
    data['value', 'rev_constrains_issue', 'issue'].edge_index = edge_list_to_tensor(issue_constrains_edges).flip(0)

    torch.save(data, f"{args.domain}_graph.pt")


    # ********************************************************************************
    # DEFINE MODEL
    # ********************************************************************************

    class HeteroGNN(nn.Module):
        def __init__(self, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = HeteroConv({
                ('argument', 'links', 'issue'): SAGEConv((-1, -1), hidden_dim),
                ('issue', 'rev_links', 'argument'): SAGEConv((-1, -1), hidden_dim),

                ('argument', 'attains', 'value'): SAGEConv((-1, -1), hidden_dim),
                ('value', 'rev_attains', 'argument'): SAGEConv((-1, -1), hidden_dim),
                ('argument', 'constrains', 'value'): SAGEConv((-1, -1), hidden_dim),
                ('value', 'rev_constrains', 'argument'): SAGEConv((-1, -1), hidden_dim),

                ('issue', 'attains', 'value'): SAGEConv((-1, -1), hidden_dim),
                ('value', 'rev_attains_issue', 'issue'): SAGEConv((-1, -1), hidden_dim),
                ('issue', 'constrains', 'value'): SAGEConv((-1, -1), hidden_dim),
                ('value', 'rev_constrains_issue', 'issue'): SAGEConv((-1, -1), hidden_dim),
            }, aggr='sum')

            self.lin = nn.Linear(hidden_dim * 2, out_dim)

        def forward(self, x_dict, edge_index_dict, arg_idx, issue_idx):
            x_dict = self.conv1(x_dict, edge_index_dict)
            print(f"x_dict['argument'].shape: {x_dict['argument'].shape}")
            print(f"arg_idx: {arg_idx}")
            print(f"x_dict['issue'].shape: {x_dict['issue'].shape}")
            print(f"issue_idx: {issue_idx}")
            arg_emb = x_dict['argument'][arg_idx]
            issue_emb = x_dict['issue'][issue_idx]
            edge_repr = torch.cat([arg_emb, issue_emb], dim=1)
            return self.lin(edge_repr)

    
    # ********************************************************************************
    # TRAIN AND EVALUATE
    # ********************************************************************************

    model = HeteroGNN(hidden_dim=64, out_dim=3).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = nn.KLDivLoss(reduction='batchmean')  # For soft targets
    loss_fn = nn.CrossEntropyLoss() # For hard targets

    def get_batch(indices):
        a_idx = arg_indices[indices].to(device)
        i_idx = issue_indices[indices].to(device)
        y = labels[indices].to(device)
        return a_idx, i_idx, y

    def train():
        model.train()
        total_loss = 0
        for i in range(0, len(train_idx), 512):
            batch_ids = train_idx[i:i+512]
            arg_idx_batch, issue_idx_batch, label_batch = get_batch(batch_ids)
            print(f"Max arg_idx_batch: {arg_idx_batch.max()}, Shape of x_dict['argument']: {data.x_dict['argument'].shape}")
            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict, arg_idx_batch, issue_idx_batch)
            loss = loss_fn(F.log_softmax(out, dim=1), label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_idx)

    @torch.no_grad()
    def evaluate(indices, show_confusion=False, save_path=None):
        model.eval()
        total_loss = 0
        correct = 0

        all_preds = []
        all_labels = []

        for i in range(0, len(indices), 512):
            arg_idx_batch, issue_idx_batch, label_batch = get_batch(indices[i:i+512])
            out = model(data.x_dict, data.edge_index_dict, arg_idx_batch, issue_idx_batch)
            pred = F.softmax(out, dim=1)

            # Accumulate for confusion matrix
            all_preds.append(pred.argmax(1).cpu())
            all_labels.append(label_batch.argmax(1).cpu())

            # Accumulate loss and accuracy
            total_loss += loss_fn(pred.log(), label_batch).item()
            correct += (pred.argmax(1) == label_batch.argmax(1)).sum().item()

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Optionally show confusion matrix
        class_names = ["no-link", "supports", "attacks"]
        labels = np.arange(len(class_names))  # [0, 1, 2]

        cm = confusion_matrix(all_labels, all_preds, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Confusion Matrix")

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        if show_confusion:
            plt.show()
        else:
            plt.close(disp)  # Avoids displaying in headless environments

        return total_loss / len(indices), correct / len(indices)

    print("Training...")
    # Do the actual training
    for epoch in range(1, 21):
        train_loss = train()
        val_loss, val_acc = evaluate(val_idx, show_confusion=True, save_path="cm.png")
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    test_loss, test_acc = evaluate(test_idx)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")



if __name__ == "__main__":
    main()




