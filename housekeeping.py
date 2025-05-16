import os


def input_file_existence_check(srcdir:str, outdir:str):
    assert os.path.exists(
        os.path.join(srcdir, "brain_category_rna_Any_Region.tsv")
    ), "Brain gene expression missing, download from: https://www.proteinatlas.org/search/brain_category_rna%3AAny%3BRegion+enriched%2CGroup+enriched%2CRegion+enhanced%2CLow+region+specificity?format=tsv&download=yes"
    assert os.path.exists(
        os.path.join(srcdir, "c5.go.v2024.1.Hs.symbols.gmt")
    ), "GO data missing, download from: https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2024.1.Hs/c5.go.v2024.1.Hs.symbols.gmt"
    assert os.path.exists(
        os.path.join(srcdir, "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct")
    ), "GTEx data missing, download from: https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
    assert os.path.exists(
        os.path.join(srcdir, "9606.protein.links.detailed.v12.0.txt")
    ), "STRING data missing, download from: https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    assert os.path.exists(
        os.path.join(srcdir, "9606.protein.info.v12.0.txt")
    ), "STRING data missing, download from: https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
    os.makedirs(outdir, exist_ok=True)
    return
