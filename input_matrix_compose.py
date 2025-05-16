import pandas as pd
from input_matrix_go import build_GO_matrix
from input_matrix_gtex import build_GTEX_matrix
from input_matrix_string import build_STRING_matrix
import housekeeping
import os

MAX_GOTERM_SIZE = 500

outdir = "data/input_matrix"
srcdir = "data/input_matrix_src"
housekeeping.input_file_existence_check(srcdir, outdir)

print("Reading brain-expressed genes...")
brain = pd.read_csv(
    os.path.join(srcdir, "brain_category_rna_Any_Region.tsv"),
    sep="\t",
    index_col=0,
    usecols=[0, 1],
)
brain = brain[~brain.index.duplicated(keep="first")]

# calc
print("Building matrices...")
print("  GO...")
go = build_GO_matrix(
    os.path.join(srcdir, "c5.go.v2024.1.Hs.symbols.gmt"),
    max_go_term_size=MAX_GOTERM_SIZE,
)
print("  GTEX...")
gtex = build_GTEX_matrix(
    os.path.join(
        srcdir, "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct"
    )
)
print("  STRING...")
string = build_STRING_matrix(
    string_net_file=os.path.join(srcdir, "9606.protein.links.detailed.v12.0.txt"),
    string_info_file=os.path.join(srcdir, "9606.protein.info.v12.0.txt"),
)


print("Merging matrices with inner join...")
df = pd.concat([brain, go, gtex, string], axis=1, join="inner")
idx = list(df.index)  # idx == common genes

# keep common genes only
go = go.loc[idx]
gtex = gtex.loc[idx]
string = string.loc[idx, idx]  # gene symbols both in index and cols

print("Saving filtered matrices...")
go.to_parquet(os.path.join(outdir, "go.parquet"))
gtex.to_parquet(os.path.join(outdir, "gtex.parquet"))
string.to_parquet(os.path.join(outdir, "string.parquet"))

fn = os.path.join(outdir, "gtex_go.parquet")
df[[*gtex.columns, *go.columns]].to_parquet(fn)


print("\nINFO")
print("=====")

# Print dimensions of each DataFrame
print("Dimensions:")
print(f"GO Matrix: {go.shape}")  # (rows, columns)
print(f"GTEX Matrix: {gtex.shape}")
print(f"STRING Matrix: {string.shape}")
