# GECko+


![logo of gecko](https://github.com/psawa/gecko-app/blob/master/application/static/img/GECko_logo_small.png)

## More than a Grammatical Error Corrector
GECko+ is an English language assisting tool that corrects mistakes of various types on written texts. 
While many well-settled pieces of software of its kind correct mistakes at the grammatical level (orthography and syntax), our novel approach allows the tool to perform corrections both at **grammatical** and at **discourse** level.
<!--- add demo link when live -->
![demo](https://github.com/psawa/gecko-app/blob/master/application/static/img/final_layout.png) 

### Use cases examples

<!--- add screenshot (possibly gif of correction) -->
Original text | Corrected text
------------ | -------------
Whoever is happy wil make other persons happy to. | Whoever is happy will make other people happy too.
The weather was so nice! Yesterday I go to beach. | Yesterday I went to the beach. The weather was so nice!
The wood are lovely, dark,, and deep. And miles to go before I sleep. But I have promises to keep. | The woods are lovely, dark, and deep. But I have promises to keep. And miles to go before I sleep.
Fool me twice, shame on me. Fool me once, shame on you. | Fool me once, shame on you. Fool me twice, shame on me.
Secondy, prepare the pan using oil and butter. Then, put onions and carrots together with salt an pepper, inside the pan. Lastly, let them cooked for 15 minutes, and remove off the food fom the pan. First of all, cut some onions and carrots. | First of all, cut some onions and carrots. Secondly, prepare the pan using oil and butter. Then, put the onions and carrots together with salt and pepper, inside the pan. Lastly, let them cook for 15 minutes, and remove the food from the pan.

## Running locally and developing
### Option 1 - Using a virtual environment
1. Clone the repository.
1. Create a virtual environment with python 3.7.
1. Install the requirements files `pip install -r requirements.txt`.
1. Download the models by executing:
    * `mkdir -p /application/models/gector/data/model_files && cd application/models/gector/data/model_files && curl -O https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gector.th`.
    * `mkdir -p /application/models/sentence_reorder cd /application/models/sentence_reorder && curl -O http://tts.speech.cs.cmu.edu/sentence_order/nips_bert.tar && tar -xf nips_bert.tar && rm nips_bert.tar && mv nips_bert/ model/`.

1. Run the web app locally by executing `run.py`. The default development URL is `http://127.0.0.1:5000/`.

### Option 2 - Using Docker
1. Build the docker image: `docker build -t gecko-app .`.
2. Run the image: `docker run -d -p 3000:80 gecko-app`. The URL will be `http:localhost:3000/`.

## Acknowledgments
Our tool implements the two following models, for tackling, respectively, grammatical and discourse errors:

- Kostiantyn Omelianchuk, Vitaliy Atrasevych, Artem Chernodub and Oleksandr Skurzhanskyi **"GECToR -- Grammatical Error Correction: Tag, Not Rewrite"**. In Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications. [[arXiv]](https://arxiv.org/abs/2005.12592)
- Prabhumoye, Shrimai, Ruslan Salakhutdinov, and Alan W. Black. **"Topological Sort for Sentence Ordering."** In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. [[arXiv]](https://arxiv.org/abs/2005.00432)

## Contribute
We accept contributions, whether they intend to fix an issue or to add new functionalities. Don't hesitate to submit a pull-request!

[![https://github.com/psawa/gecko-app/issues](https://img.shields.io/github/issues/psawa/gecko-app)](https://github.com/psawa/gecko-app/issues)

## License
The software is distributed under Common Creative 4.0 license.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
