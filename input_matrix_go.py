import pandas as pd
from typing import Dict, List

def build_GO_matrix(
    data_file:str, max_go_term_size=500, 
):
    """
    Process the raw MSigDB GO file, to produce binary gene-GO matrix.
    """
    # process
    term_geneL_D = _read_term_geneL_D(data_file)
    binary_df = _build_binary_df(term_geneL_D)
    binary_df = _filter_max_term_size(binary_df, max_term_size=max_go_term_size)
    binary_df.name = "go"
    return binary_df


def _read_term_geneL_D(
    data_file,
) -> Dict[str, List[str]]:
    """
    Read the raw GO file and return a dict like:
    ```python
    {"GO_term1": ["gene1", "gene2", ...], }
    ```
    """

    term_geneL_D = dict()

    # Read the file line by line
    with open(data_file, "r") as file:
        for line in file:
            # Strip leading/trailing whitespaces and split by tab
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                # First part is GO term, second is URL, rest are genes
                go_term = parts[0]
                genes = parts[2:]  # Remaining parts are genes
                term_geneL_D[go_term] = genes
            else:
                print(f"Warning: Line skipped due to insufficient data: {line}")
    return term_geneL_D


def _build_binary_df(term_geneL_D: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Based on the GO-genes dict, creates a DataFrame:
    - index: genes
    - columns: GO terms
    - values: 1 if the term contains the gene, else 0
    """
    # unique genes into one flat list
    all_genes_L: List[str] = sorted({g for gs in term_geneL_D.values() for g in gs})
    # Initialize a DataFrame with zeros
    go_term_L: List[str] = list(term_geneL_D.keys())
    binary_df = pd.DataFrame(0, index=all_genes_L, columns=go_term_L, dtype=int)

    # Populate the matrix
    for go_term, genes in term_geneL_D.items():
        binary_df.loc[genes, go_term] = 1
    return binary_df


def _filter_max_term_size(binary_df: pd.DataFrame, max_term_size: int) -> pd.DataFrame:
    """
    Keep only GO terms that are not larger than the given limit.
    """
    # sum genes per term
    count = binary_df.sum().sort_values()
    # filter terms
    count = count.loc[count <= max_term_size]
    return binary_df[count.index]
