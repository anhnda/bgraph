import pandas as pd
import params

class GeneData:
    def __init__(self):
        self.puvogel = self.read_gene_xls_puvogel()
        self.garcia = self.read_gene_xls_garcia()
        self.gal = self.read_gene_Gal()
        self.deli = self.read_gene_Deli()

    def __str__(self):
        puvogel_genes = self.puvogel.shape[0]
        garcia_genes = self.garcia.shape[0]
        gal_genes = self.gal.shape[0]
        deli_genes = self.deli.shape[0]
        return str(self.__class__) + "\n" + "Puvogel genes: " + str(puvogel_genes) + "\n" + "Garcia genes: " + str(garcia_genes) + "\n" + "Gal genes: " + str(gal_genes) + "\n" + "Deli genes: " + str(deli_genes) + "\n"

## read files
    def read_gene_xls_puvogel(self):
        excl = pd.ExcelFile(params.PUVOGEL)
        cc = 0
        for sheet in excl.sheet_names:
            df = pd.read_excel(excl, sheet_name=sheet)
            if cc == 0:
                df_out = df
                cc = 1
            else:
                df_out = pd.concat([df_out, df])
        return df_out

    def read_gene_xls_garcia(self):
        df_out = pd.read_excel(params.GARCIA, sheet_name="Post Mortem Vascular Subcluster")
        return df_out

    def read_gene_Gal(self):
        df_out = pd.read_csv(params.GAL, header = 0, sep = "\t", index_col = None)
        return df_out

    def read_gene_Deli(self):
        df_out = pd.read_excel(params.DELI, sheet_name="Munka1")
        return df_out

if __name__ == '__main__':
    dat = GeneData()
    print(dat)
    print(dat.gal)