import pandas as pd
import numpy as np
from data_factory.data_prep import GeneData
import params
import matplotlib.pyplot as plt
from venn import venn
import math

class CalcDirectOverlap:
    def __init__(self):
        self.gal_total = GeneData().gal
        self.puvogel = GeneData().puvogel
        self.garcia = GeneData().garcia
        self.deli = GeneData().deli
        self.deli_genes = self.deli["Name"]
        self.step_size = params.step_size
        self.puvogel_genes = self.filtering_genes_puvogel()
        self.garcia_genes = self.filtering_genes_garcia()
        self.gal_genes = self.filtering_genes_gal(None)

    def __str__(self):
        return str(GeneData()) + "\n## After filtering:\nPuvogel genes: " + str(len(self.puvogel_genes)) + "\nGarcia genes: " + str(len(self.garcia_genes)) + "\nGal genes: " + str(len(self.gal_genes)) + "\nDeli genes: " + str(len(self.deli_genes))

    def calculate_rate(self, x, total):
        return float(x)/float(total)

    def filter_df(self, df, filters):
        filtered_df = df.copy()
        for column, condition in filters.items():
            operator = condition[0]
            value = condition[1]

            if operator == '>':
                filtered_df = filtered_df[filtered_df[column] > value]
            elif operator == '<':
                filtered_df = filtered_df[filtered_df[column] < value]
            elif operator == '>=':
                filtered_df = filtered_df[filtered_df[column] >= value]
            elif operator == '<=':
                filtered_df = filtered_df[filtered_df[column] <= value]
            elif operator == '!=':
                filtered_df = filtered_df[filtered_df[column] != value]
            elif operator == '==':
                filtered_df = filtered_df[filtered_df[column] == value]
        return filtered_df

    def test_p_relevance(self):
        for l in ["puvogel", "garcia", "deli"]:
            if l == "puvogel":
                BBB_list = self.puvogel_genes
            if l == "garcia":
                BBB_list = self.garcia_genes
            if l == "deli":
                BBB_list = self.deli_genes
            plot_x = []
            plot_y = []
            figure_name = l + "_pos_vs_neg_rate"
            for i in np.arange(self.step_size,params.upperbound, self.step_size):
                filtered_gal_genes = self.filtering_genes_gal(i)
                overlap = set(BBB_list).intersection(set(filtered_gal_genes))
                plot_y.append((self.calculate_rate(len(overlap), len(set(filtered_gal_genes)) - len(overlap)))/len(BBB_list))
                plot_x.append(i)
            plt.scatter(plot_x, plot_y, marker='.', color='black')
            plt.title(figure_name)
            plt.axvline(x=1e-05, color='green')
            plt.savefig("%s/%s_%s_%s.jpeg" % (params.RESULTS, figure_name, params.step_size, params.upperbound))
            plt.clf()
            plt.close()

    def calc_enrichment(self): ## NOT RUN, currently not working
        for l in ["puvogel", "garcia", "deli"]:
            if l == "puvogel":
                BBB_list = self.puvogel_genes
            if l == "garcia":
                BBB_list = self.garcia_genes
            if l == "deli":
                BBB_list = self.deli_genes
            y = 0
            plot_x = []
            plot_y = []
            figure_name = l + "_pos_vs_neg_enrichment"
            for i in range(0, len(self.gal_total["SYMBOL"].tolist())):
                if self.gal_total["SYMBOL"].tolist()[i] in set(BBB_list):
                    y += 1
                else:
                    y -= 1
                plot_y.append(y)
                plot_x.append(i)
            print(len(plot_y))# plot_y = [y/len(BBB_list) for y in plot_y]
            print(len(plot_x))
            distances = [(y + x)/len(BBB_list) for x, y in zip(plot_x, plot_y)]
            print(len(distances))
            max_distance = max(distances)
            plt.scatter(plot_x, plot_y, marker='.', color='black')
            plt.axline((0,0), slope=-1, color='green')
            plt.text(0.95, 0.95, max_distance, ha='right', va='top', transform=plt.gca().transAxes)
            plt.savefig("%s/%s_%s.jpeg" % (params.RESULTS, l, figure_name))
            plt.clf()
            plt.close()


    def filtering_genes_puvogel(self):
        puvogel_filtered = self.filter_df(self.puvogel, params.filter_puvogel)
        return puvogel_filtered["gene"].tolist()

    def filtering_genes_garcia(self):
        garcia_filtered = self.filter_df(self.garcia, params.filter_garcia)
        return garcia_filtered["gene"].tolist()
    def filtering_genes_gal(self, i):
        if i == None:
            gal_filtered = self.filter_df(self.gal_total, params.filter_gal)
        else:
            gal_filtered = self.gal_total[self.gal_total["P"] <= i]
        return gal_filtered["SYMBOL"].tolist()

    def get_overlaps(self):
        sets = {"Puvogel": set(self.puvogel_genes), "Garcia": set(self.garcia_genes), "Deli": set(self.deli_genes), "Gal": set(self.gal_genes)}
        venn(sets)
        plt.savefig("%s/venn_results.pdf" % (params.RESULTS))
        sets_and_combinations = {"Gal": set(self.gal_genes),
                        "Garcia": set(self.garcia_genes),
                        "Puvogel": set(self.puvogel_genes),
                        "Deli": set(self.deli_genes),
                        "Gal_Deli": set(self.gal_genes).intersection(set(self.deli_genes)),
                        "Deli_Garcia": set(self.garcia_genes).intersection(set(self.deli_genes)),
                        "Deli_Puvogel": set(self.puvogel_genes).intersection(set(self.deli_genes)),
                        "Puvogel_Garcia": set(self.garcia_genes).intersection(set(self.puvogel_genes)),
                        "Gal_Garcia": set(self.gal_genes).intersection(set(self.garcia_genes)),
                        "Gal_Puvogel": set(self.gal_genes).intersection(set(self.puvogel_genes)),
                        "Gal_Deli_Garcia": set(self.gal_genes).intersection(set(self.deli_genes)).intersection(set(self.garcia_genes)),
                        "Gal_Deli_Puvogel": set(self.gal_genes).intersection(set(self.deli_genes).intersection(set(self.puvogel_genes))),
                        "Gal_Garcia_Puvogel": set(self.gal_genes).intersection(set(self.garcia_genes)).intersection(set(self.puvogel_genes)),
                        "Gal_Garcia_Puvogel_Deli": set(self.gal_genes).intersection(set(self.garcia_genes)).intersection(set(self.puvogel_genes)).intersection(set(self.deli_genes))}
        df_results = pd.DataFrame.from_dict(sets_and_combinations, orient='index')
        df_results = df_results.transpose()
        df_results.to_csv("%s/results.csv" % (params.RESULTS), header=True, index=False, sep="\t")
        df_results.to_excel("%s/results.xlsx" % (params.RESULTS), sheet_name="Overlaps", index=False)
        return sets_and_combinations