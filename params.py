ORIGINALS = "data_factory/originals"
DATA_FOLDER = "data_factory"
RESULTS = "results"

GARCIA = "data_factory/originals/41586_2022_4521_MOESM3_ESM_PP.xlsx"
PUVOGEL = "data_factory/originals/41380_2022_1796_MOESM3_ESM_PP.xlsx"
GAL = "data_factory/originals/2_osszes_genlista.txt"
DELI = "data_factory/originals/BBB-specific_genes_Deli_M_Porkolab_G_2024_06_23.xlsx"

##puvogel filters
filter_columns_puvogel = ["p_val", "avg_log2FC", "pct.1", "pct.2", "p_val_adj", "cluster", "gene"]
filter_values_puvogel = [None, None, None, None, ["<=", 0.05], None, None]
filter_puvogel = {k:v for k, v in {filter_columns_puvogel[i]: filter_values_puvogel[i] for i in range(len(filter_columns_puvogel))}.items() if v is not None}

##garcia filters
filter_columns_garcia = ["Gene_name","p_val","avg_log2FC","pct.1","pct.2","p_val_adj", "cluster", "gene"]
filter_values_garcia = [None, None, None, None, None, ["<=", 0.05], None, None]
filter_garcia = {k:v for k, v in {filter_columns_garcia[i]: filter_values_garcia[i] for i in range(len(filter_columns_garcia))}.items() if v is not None}

## gal filters
filter_columns_gal = ["GENE","SYMBOL", "CHR", "START", "STOP", "NSNPS", "NPARAM", "N", "ZSTAT", "P"]
filter_values_gal = [None, None, None, None, None, None, None, None, None, ["<=", 1e-5]]
filter_gal = {k:v for k, v in {filter_columns_gal[i]: filter_values_gal[i] for i in range(len(filter_columns_gal))}.items() if v is not None}

## test 'enrichment' -- not relevant
step_size = 0.0000001
upperbound = 0.1

