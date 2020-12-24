from . import fill_tsv, model, topological_sort

import os, re

def create_name():
    """
        Creates a unique filename in the paragraph folder
    """
    index = 0
    results_dir = './application/models/sentence_reorder/paragraph/'
    os.makedirs(results_dir, exist_ok=True) 
    dir_content = os.listdir(results_dir)
    if len(dir_content) > 0:
        index_list = []
        for filename in dir_content:
            match = re.findall(r"^\d(?=_)", filename)
            if len(match) > 0:
                num = match[0]
                index_list.append(int(num))            
        if len(index_list)>0:
            max_index = max(index_list)
            index = max_index + 1
    name = f"{index}_test.tsv"
    return name 


def book_file(filename):
    results_dir = './application/models/sentence_reorder/paragraph/'
    index = filename.split("_")[0]
    f = open(f"{results_dir}/{index}_book","w+")
    f.close()



def get_order(sentences:list)->list:
    filename = create_name()
    book_file(filename)
    fill_tsv.prepare_data(sentences, filename=filename)
    model.compute_probabilities(filename=filename)
    order = topological_sort.get_order(filename=filename) #list of indexes
    model.clean_cache()
    return order
    

def reorder(sentences:list, order:list)->list:
    ordered_sentences = [sentences[i] for i in order]
    return ordered_sentences


def load_model():
    model.load_model()