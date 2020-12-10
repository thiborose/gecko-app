import csv
import random

from transformers import (WEIGHTS_NAME, BertConfig,
                BertForSequenceClassification, BertTokenizer)

for _, tokenizer_class, pretrained_weights in \
        [(BertForSequenceClassification, 
        BertTokenizer, 
        'bert-base-uncased')]:
            tokenizer = tokenizer_class.from_pretrained(
                                            pretrained_weights)

def truncate_test(sent1, sent2):
        s1i = tokenizer.encode(sent1)
        s2i = tokenizer.encode(sent2)
        if len(s1i) < 50:
            sent2 = tokenizer.decode(s2i[:100-len(s1i)])
        elif len(s2i) < 50:
            sent1 = tokenizer.decode(s1i[:100-len(s2i)])
        else:
            sent1 = tokenizer.decode(s1i[:50])
            sent2 = tokenizer.decode(s2i[:50])
        inp = tokenizer.encode(sent1, sent2, add_special_tokens=True)
        assert len(inp) < 105
        return sent1, sent2

def prepare_data(sentences:list, filename="test.tsv", out_dir="paragraph/"):
    story_sentences = {0:sentences}

    x, y = 0, 0
    filename = out_dir + filename
    with open(filename, "w") as out:
        tsv_writer = csv.writer(out, delimiter='\t')
        for story_id in story_sentences.keys():
            y += 1                
            if y%100 == 0:
                print(y, x) 

            story = story_sentences[story_id]

            tmp = []
            for i in range(len(story)):
                for j in range(i+1, len(story)):

                    sent1 = story[i]
                    sent2 = story[j]

                    #check if tokenized input is greater than 100
                    inputs = tokenizer.encode(
                                        sent1.lower(), 
                                        sent2.lower(), 
                                        add_special_tokens=True)

                    length = len(inputs)
                    if length > 100:
                        #print(sent1, sent2)
                        sent1, sent2 = truncate_test(
                                                    sent1, sent2)
                        #print(sent1, sent2)

                    x += 1
                    r = random.random()
                    if r >= 0.5:
                        tmp.append([str(y)+'-'+str(len(story)), \
                                            sent1, sent2, 1, i, j])
                    else:
                        tmp.append([str(y)+'-'+str(len(story)), \
                                            sent2, sent1, 0, j, i])

            for row in tmp:
                #adding no of pairs of sentences in the end
                row[0] += '-' + str(len(tmp))
                tsv_writer.writerow(row)


if __name__ == "__main__":
    print('lets go')
    sentences=['It was a very nice day!.','This is because the weather was sunny.','Yesterday, I hiked some mountains']
    prepare_data(sentences)