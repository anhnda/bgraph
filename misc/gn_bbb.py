import copy
import os.path as osp
import time
from multiprocessing import freeze_support

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from BBBData import BBBData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


device = torch.device("cpu")
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = BBBData("input_matrix", ifold=1)

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')

kwargs = {'batch_size': 200, 'num_workers': 0, 'persistent_workers': False}
print(data.train_mask, torch.sum(data.train_mask))
train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[25, 10], shuffle=True, **kwargs)

subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)

# No need to maintain these features during evaluation:
del subgraph_loader.data.x, subgraph_loader.data.y
# Add global node index information.
subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.1, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

model = SAGE(dataset.num_features, 300, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Num classes: ", dataset.num_classes)

def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        # print("Batch: ", batch)
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y, weight=torch.tensor([0.001, 1]).to(device))
        # print(y.shape, y_hat.shape)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def test():
    model.eval()
    y_hat_all = model.inference(data.x, subgraph_loader)
    y_hat = y_hat_all[:,1]

    # print(y_hat.shape)
    # y = data.y.to(y_hat.device)


    # print(y.shape)
    accs = []
    score_val_pos = y_hat[data.val_mask]
    score_test_pos = y_hat[data.test_mask]
    score_candidates = y_hat[data.candidate_mask]

    eval_val, smax = evalx(score_val_pos, score_candidates)
    eval_test, _ = evalx(score_test_pos, score_candidates)
    accs.append((eval_val, eval_test, smax))


def evalx(score_val_pos, score_candidates):
    eval_val = torch.concatenate((score_val_pos, score_candidates), dim=0)
    rank = torch.argsort(eval_val, dim=0, descending=True)
    s1 = rank[:len(score_val_pos)]
    s1 = 1 / s1
    s1 = torch.sum(s1) / len(score_val_pos)
    ar = np.arange(1, len(score_val_pos) + 1)
    ar = 1 / ar
    smax = np.sum(ar) / len(score_val_pos)

    return s1, smax
if __name__ == '__main__':
    freeze_support()
    times = []
    for epoch in range(1, 20):
        start = time.time()
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        eval_val, eval_test, emax = test()

        print(f'Epoch: {epoch:02d} Val: {eval_val:.4f}  Test: {eval_test:.4f} Smax: {emax:.4f}')

        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
