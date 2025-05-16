def loadFile():
    gene_lists = []
    with open('TopGeneO.txt') as f:
        lines = f.readlines()
        for line in lines:
            if line == "":
                break
            if len(line) < 10:
                continue
            line = line.strip()
            gene_list = eval(line)
            gene_lists.append(gene_list)
    return gene_lists
def addDictCount(d,k):
    try:
        v = d[k]
    except:
        v = 0
    d[k] = v + 1

def sort_dict(dd):
    kvs = []
    for key, value in sorted(dd.items(), key=lambda p: (p[1], p[0])):
        kvs.append([key, value])
    return kvs[::-1]


def preprocess(gene_lists):
    geneCounts = dict()
    for gene_list in gene_lists:
        for gene in gene_list:
            addDictCount(geneCounts, gene)
    sortedGeneCounts = sort_dict(geneCounts)
    for geneCount in sortedGeneCounts:
        gene, count = geneCount
        print(gene, count * 1.0/10)


if __name__ == '__main__':
    gene_lists = loadFile()
    preprocess(gene_lists)


