import pandas as pd
from mlxtend.preprocessing import TransactionEncoder 
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

class AlgoritmaFpgrowth:
    def fp_growth(self, data, min_supp):
        # untuk pembentukan fptree akan menggunakan library dengan metode fpgrowth
        itemsets = fpgrowth(data, min_support = min_supp, use_colnames=True)
        
        return itemsets

    def get_association_rules(self, itemsets, min_conf):
        # untuk pembentukan fptree akan menggunakan library dengan metode association_rules
        rules = association_rules(itemsets, metric="confidence", min_threshold = min_conf)
        
        return rules