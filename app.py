from flask import Flask, render_template, request
import pandas as pd
from algorithms.apriori import *
from algorithms.fpgrowth import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/results')
@app.route('/results', methods=['POST'])
def result():
    # Menerima input dataset dari form
    dataset = request.files['dataset']

    # Menerima input support dan confidence dari form
    support = float(request.form['support'])/100
    confidence = float(request.form['confidence'])/100

    # Menerima input jenis algoritma yang akan digunakan
    algorithm_selected = request.form['algoritma']

    # Simpan dataset ke local storage
    dataset_path = 'datasets/' + dataset.filename
    dataset.save(dataset_path)

    if (algorithm_selected == 'apriori'):
        algo = AlgoritmaApriori()
        item = Mapping()
        rules = GenerateAssociationRules()

        data = pd.read_csv(dataset_path, index_col=0)

        item_dict = item.item_dictionary(data)
        transaction = item.list_transaction(data)

        # Mendapatkan itemset
        results_itemset = algo.apriori(data, transaction, support)

        # Generate Rules
        support_dict = rules.get_support_dict(item_dict, results_itemset)
        results_rules = rules.association_rules(confidence, support_dict)

    elif (algorithm_selected == 'fpgrowth'):
        # Menjalankan algoritma Association Rules
        algo = AlgoritmaFpgrowth()

        data = pd.read_csv(dataset_path, index_col=0)
        results_itemset = algo.fp_growth(data, support)

        rules = algo.get_association_rules(results_itemset, confidence)
        results_rules = rules.values.tolist()

    converted_data = []
    for item in results_rules:
        converted_item = []
        for value in item:
            if isinstance(value, frozenset):
                converted_item.append(list(value))
            else:
                converted_item.append(value)
        converted_data.append(converted_item)
    total_rules = len(converted_data)

    return render_template('result.html', results=converted_data, metode=algorithm_selected, sum_result=total_rules)

if __name__ == '__main__':
    app.run(debug=True)
