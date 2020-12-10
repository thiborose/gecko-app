from . import fill_tsv, model, topological_sort

def get_order(sentences:list)->list:
    fill_tsv.prepare_data(sentences)
    model.compute_probabilities()
    order = topological_sort.get_order() #list of indexes
    return order
    

def reorder(sentences:list, order:list)->list:
    ordered_sentences = [sentences[i] for i in order]
    return ordered_sentences
