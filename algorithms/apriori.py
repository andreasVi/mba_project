import pandas as pd
from itertools import combinations
from collections import defaultdict
import datetime

class Mapping:
    def item_dictionary(self, data):
        item_dict = dict()
        item_list = list(data.columns)

        for i, item in enumerate(item_list):
            # menyimpan setiap itemset dan kodenya
            item_dict[item] = i + 1
            
        return item_dict
    
    def list_transaction(self, data):
        transactions = list()
        item_dict = self.item_dictionary(data)
        # mengubah setiap item yang ada di sebuah transaksi atau data menjadi kode
        for i, row in data.iterrows():
            transaction = set()

            for item in item_dict:
                if row[item] == 1:
                    transaction.add(item_dict[item])
            transactions.append(transaction)

        return transactions

class AlgoritmaApriori:
    def get_support(self, transactions, item_set):
        match_count = 0 # untuk menghitung kemunculan tiap item
        for transaction in transactions:
            if item_set.issubset(transaction):
                match_count += 1
                # menerapkan rumus support
                # jumlah suatu item dibagi total data
        return float(match_count/len(transactions))

    def self_join(self, frequent_item_sets_per_level, level):
        current_level_candidates = list()
        last_level_items = frequent_item_sets_per_level[level - 1]

        if len(last_level_items) == 0:
            return current_level_candidates
        
        # mencari kombinasi dengan union
        for i in range(len(last_level_items)):
            for j in range(i+1, len(last_level_items)):
                itemset_i = last_level_items[i][0]
                itemset_j = last_level_items[j][0]
                union_set = itemset_i.union(itemset_j)

                if union_set not in current_level_candidates and len(union_set) == level:
                    current_level_candidates.append(union_set)

        return current_level_candidates

    def get_single_drop_subsets(self, item_set):
        single_drop_subsets = list()
        for item in item_set:
            temp = item_set.copy()
            temp.remove(item)
            single_drop_subsets.append(temp)

        return single_drop_subsets

    def is_valid_set(self, item_set, prev_level_sets):
        single_drop_subsets = self.get_single_drop_subsets(item_set)

        for single_drop_set in single_drop_subsets:
            if single_drop_set not in prev_level_sets:
                return False
        return True

    def pruning(self, frequent_item_sets_per_level, level, candidate_set):
        post_pruning_set = list()
        if len(candidate_set) == 0:
            return post_pruning_set

        prev_level_sets = list()
        for item_set, _ in frequent_item_sets_per_level[level - 1]:
            prev_level_sets.append(item_set)

        for item_set in candidate_set:
            if self.is_valid_set(item_set, prev_level_sets):
                post_pruning_set.append(item_set)

        return post_pruning_set
    
    def apriori(self, data, transactions, min_support):
        item_list = list(data.columns)
        frequent_item_sets_per_level = defaultdict(list)
        # print("level : 1", end = " ")
        
        # untuk menghitung setiap 1-itemset yang ada dalam transactions
        for item in range(1, len(item_list) + 1):
            # menghitung nilai support setiap 1-itemset
            support = self.get_support(transactions, {item})
            if support >= min_support: 
                # support yang memenuhi akan disimpan
                frequent_item_sets_per_level[1].append(({item}, support))
        
        # untuk menghitung k-itemset setelah mendapat nilai 1-itemset
        for level in range(2, len(item_list) + 1):
            # print(level, end = " ")
            
            # setiap itemset akan dibentuk kombinasinya dengan self_join
            current_level_candidates = self.self_join(frequent_item_sets_per_level, level)
            
            # memeriksa apakah kombinasi yang sudah dihasilkan ada di level sebelumnya
            post_pruning_candidates = self.pruning(frequent_item_sets_per_level, level, current_level_candidates)
            
            if len(post_pruning_candidates) == 0:
                break

            for item_set in post_pruning_candidates:
                # menghitung nilai support untuk setiap itemset yang sudah didapat
                support = self.get_support(transactions, item_set)
                
                # melakukan pengecekan apakah memenuhi min_support atau tidak
                if support >= min_support:
                    frequent_item_sets_per_level[level].append((item_set, support))

        return frequent_item_sets_per_level

class GenerateAssociationRules:
    def get_support_dict(self, item_dict, frequent_item_sets_per_level):
        item_support_dict = dict()
        item_list = list()

        key_list = list(item_dict.keys())
        val_list = list(item_dict.values())

        for level in frequent_item_sets_per_level:
            for set_support_pair in frequent_item_sets_per_level[level]:
                for i in set_support_pair[0]:
                    item_list.append(key_list[val_list.index(i)])
                item_support_dict[frozenset(item_list)] = set_support_pair[1]
                item_list = list()
        
        return item_support_dict
    
    def find_subset(self, item, item_length):
        combs = []
        for i in range(1, item_length + 1):
            combs.append(list(combinations(item, i)))

        subsets = []
        for comb in combs:
            for elt in comb:
                subsets.append(elt)

        return subsets
    
    def association_rules(self, min_confidence, support_dict):
        rules = list()
        for item, support in support_dict.items():
            item_length = len(item)
            
            # rules akan dicari dari 2-itemset sampai k-itemset
            if item_length > 1:
                # memisahkan item dari frozenset menjadi array
                subsets = self.find_subset(item, item_length)
                
                # Menentukan nilai confidence dari setiap kombinasi suatu itemset
                # item(A) -> item(B) berbeda dengan item(B) -> item(A)
                for A in subsets:
                    B = item.difference(A)

                    if B:
                        A = frozenset(A)

                        AB = A | B
                        # menghitung nilai confidence
                        support = support_dict[AB]
                        confidence = support / support_dict[A]
                        if confidence >= min_confidence:
                            # menghitung nilai lift
                            lift = confidence / support_dict[B]
                            rules.append((A, B, support, confidence, lift))

        return rules