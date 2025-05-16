import pandas as pd
from typing import List, Dict, Set


def build_GTEX_matrix(tissue_expression_file_path:str) -> pd.DataFrame:
    """
    Returns DataFrame with:
    - index: gene symbols
    - columns: tissues
    - values: median expression
    """
    df = pd.read_csv(
        tissue_expression_file_path,
        sep="\t",
        skiprows=2,
        index_col=1,
    )
    df = df.drop(columns="Name")
    df = df.groupby(level=0).sum() # sum expression of genes with same symbol, but different ENSG
    return df
