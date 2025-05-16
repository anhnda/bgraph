from calcs.cal_direct_overlap import CalcDirectOverlap

def run():
    dat = CalcDirectOverlap()
    print(dat)
    # dat.test_p_relevance()
    # dat.calc_enrichment()
    dat.get_overlaps()

if __name__ == '__main__':
    run()
