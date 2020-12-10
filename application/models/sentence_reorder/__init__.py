from . import fill_tsv, model, topological_sort

def reorder(sentences:list)->list:
    fill_tsv.prepare_data(sentences)
    model.compute_probabilities()
    order = topological_sort.get_order() #list of indexes
    ordered_sentences = [sentences[i] for i in order]
    return ordered_sentences
