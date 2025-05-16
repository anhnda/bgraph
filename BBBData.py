import os
import os.path as osp
from typing import Callable, List, Optional

import joblib
import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    Dataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import coalesce


class BBBData(InMemoryDataset):
    r"""The Reddit dataset from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, containing
    Reddit posts belonging to different communities.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 232,965
          - 114,615,892
          - 602
          - 41
    """

    url = 'https://data.dgl.ai/dataset/reddit.zip'

    def __init__(
            self,
            root: str,
            ifold: int,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = False,
    ) -> None:
        self.ifold = ifold
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        print(self.processed_paths)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['reddit_data.npz', 'reddit_graph.npz']

    @property
    def processed_file_names(self) -> str:
        return 'data_%s.pt' % self.ifold

    def download(self) -> None:
        return

    def process(self) -> None:
        import scipy.sparse as sp

        row, col, x, lbs, split, _, _, _, _ = joblib.load(osp.join(self.root, 'gene_graph.pkl'))
        lbs = np.asarray(lbs, dtype=int)
        split = split[self.ifold]
        print(type(lbs), lbs)
        x = torch.from_numpy(x).to(torch.float)
        y = torch.from_numpy(np.asarray(lbs)).to(torch.long)
        split = torch.from_numpy(np.asarray(split))
        print(x.shape, y.shape, split.shape)

        row = torch.from_numpy(row).to(torch.long)
        col = torch.from_numpy(col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        print(edge_index.shape)
        edge_index = coalesce(edge_index, num_nodes=x.size(0))
        print(edge_index.shape)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = split == 1
        data.val_mask = split == 2
        data.test_mask = split == 3
        data.candidate_mask = lbs == 2
        data.other_mask = lbs == 0
        data.merged_mask = lbs != 1

        data.candidate_mask = torch.from_numpy(data.candidate_mask)
        data.merged_mask = torch.from_numpy(data.merged_mask)
        data.other_mask = torch.from_numpy(data.other_mask)

        print(data.train_mask.shape, data.test_mask.shape)
        print(data.train_mask.sum(), data.test_mask.sum())
        print("Candidate: ", data.candidate_mask.sum(), data.candidate_mask.sum())
        print("Other: ", data.other_mask.sum(), data.other_mask.shape)
        print("Merged: ", data.merged_mask.sum(), data.merged_mask.shape)

        data = data if self.pre_transform is None else self.pre_transform(data)
        # self.data = data
        self.save([data], self.processed_paths[0])
