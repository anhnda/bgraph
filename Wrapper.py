import copy
import json
import os.path as osp
import random
import time
from multiprocessing import freeze_support
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from BBBData import BBBData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score

from sagex import SAGEX

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TOP = 10000
W_CAND_IGNORE = 1e-6
W_CAND_OTHER = 0.01
W_CAND = W_CAND_IGNORE
GENE_OUT = "top_gen_score_w_%d" % W_CAND
W_TYPE_NAME = {W_CAND_OTHER: 'OtherCat', W_CAND_IGNORE: 'IgnoreCat'}
class BBBGN:
    def __init__(self, data_dir="input_matrix", ifold=1, num_layers=4):

        self.device = torch.device("cpu")
        # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
        dataset = BBBData(data_dir, ifold=ifold)

        # Already send node features/labels to GPU for faster access during sampling:
        self.data = dataset[0].to(self.device, 'x', 'y')

        kwargs = {'batch_size': 200, 'num_workers': 0, 'persistent_workers': False}
        # print(self.data.train_mask, torch.sum(data.train_mask))
        self.train_loader = NeighborLoader(self.data, input_nodes=self.data.train_mask,
                                           num_neighbors=[25, 10], shuffle=True, **kwargs)

        self.subgraph_loader = NeighborLoader(copy.copy(self.data), input_nodes=None,
                                              num_neighbors=[-1], shuffle=False, **kwargs)

        # No need to maintain these features during evaluation:
        del self.subgraph_loader.data.x, self.subgraph_loader.data.y
        # Add global node index information.
        self.subgraph_loader.data.num_nodes = self.data.num_nodes
        self.subgraph_loader.data.n_id = torch.arange(self.data.num_nodes)

        self.model = SAGEX(dataset.num_features, 300, dataset.num_classes, num_layers=num_layers).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # print("Num classes: ", dataset.num_classes)

    def train(self, epoch):

        self.model.train()

        pbar = tqdm(total=int(len(self.train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_correct = total_examples = 0
        for batch in self.train_loader:
            # print("Batch: ", batch.x.shape, batch.y.shape, batch.edge_index.shape)
            self.optimizer.zero_grad()
            y = batch.y[:batch.batch_size]
            y_hat = self.model(batch.x, batch.edge_index.to(self.device))[:batch.batch_size]
            if (torch.isnan(y_hat).any()):
                print("Nan yhat")
                exit(-1)
            if torch.isinf(y_hat).any():
                print("Detected -inf or inf in logits")

            # y_hat = torch.clamp(y_hat, min=-20, max=20)
            assert y.dtype == torch.long
            assert y_hat.shape[0] == y.shape[0]
            # print("Unique y:", y.unique())
            # print("y_hat shape:", y_hat.shape)
            assert y.max().item() < y_hat.shape[1]

            loss = F.cross_entropy(y_hat, y, weight=torch.tensor([0.05, 1, W_CAND]).to(self.device))

            # print(y.shape, y_hat.shape)
            if (torch.isnan(loss)):
                print("Nan loss")
                exit(-1)
            loss.backward()
            # clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN detected in parameter: {name}")
                    exit(-1)



            total_loss += float(loss) * batch.batch_size
            total_correct += int((y_hat.argmax(dim=-1) == y).sum())
            total_examples += batch.batch_size
            pbar.update(batch.batch_size)
        pbar.close()

        return total_loss / total_examples, total_correct / total_examples

    @torch.no_grad()
    def test(self):
        self.model.eval()
        y_hat_all = self.model.inference(self.data.x, self.subgraph_loader, self.device)
        print("Shape yhat: ", y_hat_all.shape)
        scores = torch.nn.Softmax(-1)(y_hat_all)
        print("Score: ", scores[:10])
        print("Score 1: ", scores[:10, 1])
        y_hat = scores[:, 1]
        accs = []
        print("Len: ", len(self.data.val_mask), len(self.data.test_mask), len(self.data.merged_mask))
        score_val_pos = y_hat[self.data.val_mask]
        score_test_pos = y_hat[self.data.test_mask]
        score_merged = y_hat[self.data.merged_mask]
        # assert len(self.data.val_mask) == len(self.data.test_mask) == len(self.data.candidate_mask)
        merged_ids = self.data.merged_mask.numpy().nonzero()[0]
        # candidate_ids = self.data.candidate_mask.numpy().nonzero()[0]
        val_ids = self.data.val_mask.numpy().nonzero()[0]
        test_ids = self.data.test_mask.numpy().nonzero()[0]

        print(torch.sum(score_val_pos), torch.sum(score_test_pos),
              torch.sum(score_merged))
        print("Len: ", len(score_val_pos), len(score_test_pos), len(score_merged))

        # eval_val, smax = evalx(score_val_pos, score_candidates)
        # eval_test, _ = evalx(score_test_pos, score_candidates)
        eval_val, smax, _ = evalx2(score_val_pos, score_merged, val_ids, merged_ids)
        eval_test, _, _ = evalx2(score_test_pos, score_merged, test_ids, merged_ids)
        eval_merged, _, top_gene_names_merged = evalx2([], score_merged, [], merged_ids)

        return eval_val, eval_test, top_gene_names_merged, smax

    def run(self, fold_id = 0):
        times = []
        all_vals = []
        all_tests = []
        all_gene_names = []
        for epoch in range(1, 21):
            start = time.time()
            loss, acc = self.train(epoch)
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
            eval_val, eval_test, top_genes, emax = self.test()
            all_tests.append(eval_test)
            all_vals.append(eval_val)
            all_gene_names.append(top_genes)
            print(eval_val, eval_test, len(top_genes), emax)
            print(f'Epoch: {epoch:02d} Val: {eval_val:.4f}  Test: {eval_test:.4f} Smax: {emax:.4f}')

            times.append(time.time() - start)

        print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
        ar = np.asarray(all_vals)
        id = np.argmax(ar)
        print("Max: ", all_vals[id], all_tests[id])
        # print("Selected Genes: ")
        # print(all_gene_names[id])
        with open(GENE_OUT + "_%d.json" % fold_id, 'w') as f:
            json.dump(all_gene_names[id], f)

        return all_gene_names[id]

def evalx(score_val_pos, score_candidates):
    eval_val = torch.concatenate((score_candidates, score_val_pos), dim=0)
    print("ESum: ", torch.sum(eval_val))
    rs = torch.argsort(eval_val, dim=0, descending=True)
    rank = torch.zeros(len(rs), dtype=int)
    for i in range(len(rs)):
        rank[rs[i]] = i + 1

    print(rank[:len(score_val_pos)])
    print(eval_val[rank[:len(score_val_pos)] - 1])
    print(rs[:20])
    s1 = rank[:len(score_val_pos)]
    s1 = 1 / s1
    s1 = torch.sum(s1) / len(score_val_pos)
    ar = np.arange(1, len(score_val_pos) + 1)
    ar = 1 / ar
    smax = np.sum(ar) / len(score_val_pos)

    return s1, smax


gene_map = {}
gene_file = open("input_matrix/processed/gene_names.txt")
lines = gene_file.readlines()
for i, gene in enumerate(lines):
    gene_map[i] = gene.strip()

@torch.no_grad()
def evalx2(score_val_pos, score_candidates, val_ids, candidate_ids):
    if len(score_val_pos) != 0:
        eval_val = torch.concatenate((score_candidates, score_val_pos), dim=0)
        all_ids = np.concatenate((candidate_ids, val_ids), axis=0)
    else:
        eval_val = score_candidates
        all_ids = candidate_ids
    print("ESum: ", torch.sum(eval_val))
    rs = torch.argsort(eval_val, dim=0, descending=True)
    rank = torch.zeros(len(rs), dtype=int)
    for i in range(len(rs)):
        rank[rs[i]] = i + 1

    print(rank[:len(score_val_pos)])
    print(eval_val[rank[:len(score_val_pos)] - 1])
    print("Top x: ", rs[:20])
    top_scores = eval_val[rs[:N_TOP]].numpy()
    print("Top scores: ", top_scores)
    original_ids = all_ids[rs[:N_TOP]]
    print("Original IDs: ", original_ids)
    top_gene_names = [gene_map[k] for k in original_ids]
    print("Top gene names: ", top_gene_names[:20])
    top_gene_scores = {}
    for ii in range(N_TOP):
        top_gene_scores[top_gene_names[ii]] = float(top_scores[ii])
    if len(score_val_pos) != 0:
        s1 = rank[:len(score_val_pos)]
        s1 = 1 / s1
        s1 = torch.sum(s1) / len(score_val_pos)
        ar = np.arange(1, len(score_val_pos) + 1)
        ar = 1 / ar
        smax = np.sum(ar) / len(score_val_pos)
    else:
        s1 = 0
        smax = 0

    return s1, smax, top_gene_scores

def update_dict(dsource, dtarget):
    for k, v in dsource.items():
        try:
            vt = dtarget[k]
        except:
            vt = 0
        dtarget[k] = v + vt
if __name__ == '__main__':
    freeze_support()
    K_FOLDS = 10
    total_folds = {}
    for k in range(K_FOLDS):
        # if k != 1:
        #     continue
        random.seed(0)
        torch.manual_seed(0)
        print("------Start fold {}------".format(k))
        model = BBBGN(ifold=k, num_layers=5)
        fold_result = model.run(k)
        update_dict(fold_result, total_folds)
        print("________________________")
    total_fold_avg = {}
    for k,v in total_folds.items():
        total_fold_avg[k] = v / K_FOLDS
    sortedDict = OrderedDict(sorted(total_fold_avg.items(), key=lambda item: item[1], reverse=True))
    with open("output_%s.json" % W_TYPE_NAME[W_CAND], "w") as f:
        f.write('{\n')
        for i, (k, v) in enumerate(sortedDict.items()):
            comma = ',' if i < len(sortedDict) - 1 else ''
            f.write(f'  {json.dumps(k)}: {json.dumps(v)}{comma}\n')
        f.write('}\n')