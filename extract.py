import random
import pandas as pd
import numpy as np
import joblib


# ss = pd.read_parquet('input_matrix/string.parquet', engine='pyarrow')
# print(ss.shape)
# d = ss.to_numpy()
# print(d.nonzero()[0].shape)
# print(ss)
# print(np.sum(ss.to_numpy()))
# ss2 = pd.read_parquet('input_matrix/gtex.parquet', engine='pyarrow')
# print(ss2.shape)
# print(ss2)
# print(ss2.index.tolist())
def load_file(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    lines = [l.strip() for l in lines]
    f.close()
    return lines


def export_graph_gt(r=0.8):
    A = pd.read_parquet('input_matrix/string.parquet', engine='pyarrow')
    A = A.to_numpy()
    non_zeros = A.nonzero()
    rows = non_zeros[0]
    cols = non_zeros[1]

    F = pd.read_parquet('input_matrix/gtex.parquet', engine='pyarrow')
    x = F.to_numpy()

    all_genes = F.index.tolist()
    positive_genes = set(load_file("input_matrix/A1.txt"))
    lbs = []
    split = []
    n_neg = len(positive_genes)
    nc = 0
    r_neg = (n_neg * 1.0) / (len(all_genes) - len(positive_genes))
    for gene in all_genes:
        if gene in positive_genes:
            lbs.append(1)
            v = random.uniform(0, 1)
            if (v < r):
                split.append(1)
            else:
                split.append(2)

        else:
            lbs.append(0)
            v = random.uniform(0, 1)
            if v <= r_neg and nc < n_neg:
                split.append(2)
                nc += 1
            else:
                split.append(1)
    print(len(split), len(lbs))
    joblib.dump((rows, cols, x, lbs, split, all_genes, positive_genes), "input_matrix/gene_graph.pkl")


def export_graph_k_fold(k=5):
    A = pd.read_parquet('input_matrix/string.parquet', engine='pyarrow')

    gene_names = A.index.tolist()
    fout = open("input_matrix/processed/gene_names.txt", "w")
    for gene in gene_names:
        fout.write(gene + "\n")
    fout.close()
    A = A.to_numpy().astype(np.float32)
    nonz = A.nonzero()
    print("Original Ratio: ", len(nonz[0]) / (A.shape[0] * A.shape[1]), np.sum(A))
    t = 0  # 100*np.mean(A)

    print("Mean: ", t, np.min(A), np.max(A))
    A[A < t] = 0
    print("S: ", np.sum(A), np.mean(A), np.min(A), np.max(A))
    non_zeros = A.nonzero()
    rows = non_zeros[0]
    cols = non_zeros[1]

    print("After: ", len(rows) / (A.shape[0] * A.shape[1]))

    F = pd.read_parquet('input_matrix/gtex.parquet', engine='pyarrow')
    x = F.to_numpy()

    all_genes = F.index.tolist()
    positive_genes = load_file("input_matrix/A1.txt")
    candidate_genes = load_file("input_matrix/A2.txt")
    random.shuffle(positive_genes)
    random.shuffle(candidate_genes)
    print("Size: ", len(positive_genes), len(candidate_genes), len(all_genes))
    lbs = []
    for gene in all_genes:
        if gene in positive_genes:
            lbs.append(1)
        elif gene in candidate_genes:
            lbs.append(2)
        else:
            lbs.append(0)
    lbs2 = np.asarray(lbs)
    print("Stats: sum: ", np.sum(lbs2 == 2), np.sum(lbs2 == 0), np.sum(lbs2 == 1), np.sum(lbs2))
    n_neg = len(positive_genes)
    positive_seg_size = int(len(positive_genes) / k)
    splits = []
    d_index = {}
    for i, pos_gene in enumerate(positive_genes):
        ix = all_genes.index(pos_gene)
        d_index[i] = ix
    candidates_index = (lbs2 == 2)
    other_index = (lbs2 == 0)

    for ki in range(k):
        split = np.ones(len(all_genes))
        # print("Len candidates: ", len(candidate_genes))
        # split[candidates_index] =
        print(len(split), len(lbs), positive_seg_size, len(positive_genes))
        start_index = ki * positive_seg_size
        end_index = (ki + 1) * positive_seg_size
        if ki == k - 1:
            end_index = len(positive_genes)
        m = int((start_index + end_index) / 2)

        for i in range(start_index, m):
            split[d_index[i]] = 2
        for i in range(m, end_index):
            split[d_index[i]] = 3

        # ones_indices = np.where(split == 1)[0]
        # print(len(ones_indices))
        # print(type(ones_indices))
        # sample_indices = np.random.choice(ones_indices, 2 * n_neg, replace=False)
        # split[sample_indices[:n_neg]] = 2
        # split[sample_indices[n_neg:]] = 3
        # print(ki, split)

        splits.append(split)

    joblib.dump((rows, cols, x, lbs, splits, candidates_index, other_index, all_genes, positive_genes),
                "input_matrix/gene_graph.pkl")


if __name__ == '__main__':
    random.seed(1)
    export_graph_k_fold(10)
    pass
