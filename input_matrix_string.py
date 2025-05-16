import pandas as pd
import numpy as np


def build_STRING_matrix(
    string_net_file:str,
    string_info_file:str,
) -> pd.DataFrame:
    """
    Process the raw STRING files, to produce gene-gene adj.matrix. Cells hold continous confidence values [0,1]
    """
    # Read the interaction network data
    net = pd.read_csv(string_net_file, delim_whitespace=True)

    # Read the protein information data
    info = pd.read_csv(string_info_file, sep="\t", index_col=0)
    # Create a mapping from protein IDs to preferred names
    protein_mapping = info["preferred_name"].to_dict()

    # Map the protein IDs in 'net' to their preferred names
    net["protein1"] = net["protein1"].map(protein_mapping).fillna(net["protein1"])
    net["protein2"] = net["protein2"].map(protein_mapping).fillna(net["protein2"])

    # Pivot the DataFrame to create the adjacency matrix
    df = (
        net.pivot_table(
            index="protein1",
            columns="protein2",
            values="combined_score",
            aggfunc="mean",  # to handle (nonexistent) duplicates
            fill_value=0
        )
        / 1000
    )  # Scale the combined_score
    df.name = "string"
    return df
