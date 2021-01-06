# GECko+


![logo of gecko](https://github.com/psawa/gecko-app/blob/master/application/static/img/GECko_logo_small.png)

## More than a Grammatical Error Corrector
GECko+ is an English language assisting tool that corrects mistakes of various types on written texts. 
While many well-settled pieces of software of its kind correct mistakes at the grammatical level (orthography and syntax), our novel approach allows the tool to perform corrections both at **grammatical** and at **discourse** level.
<!--- add demo link when live -->
![demo](https://github.com/psawa/gecko-app/blob/master/application/static/img/demo_screen.png) 

### Use cases examples

<!--- add screenshot (possibly gif of correction) -->
Original text | Corrected text
------------ | -------------
This chemical is widly used in the swimming pools market. Chlorine is well known for its sanatizing properties. | Chlorine is well known for its sanatizing properties. This chemical is widly used in the swimming pools market.
Add examples... | Add examples... 

## Running locally and developing
After cloning the repository, execute `setup.sh`. This simple script will install the requirement packages, and download the models which are too heavy for being stored on GitHub. The usage of a virtual environment having python 3.7 is preferred.

To run the web app locally, execute `run.py`. The default development URL is `http://127.0.0.1:5000/`.

## Acknowledgments
Our tool implements the two following models, for tackling, respectively, grammatical and discourse errors:

- Kostiantyn Omelianchuk, Vitaliy Atrasevych, Artem Chernodub and Oleksandr Skurzhanskyi **"GECToR -- Grammatical Error Correction: Tag, Not Rewrite"**. In Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications. [[arXiv]](https://arxiv.org/abs/2005.12592)
- Prabhumoye, Shrimai, Ruslan Salakhutdinov, and Alan W. Black. **"Topological Sort for Sentence Ordering."** In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. [[arXiv]](https://arxiv.org/abs/2005.00432)

## Contribute
[![https://github.com/psawa/gecko-app/issues](https://img.shields.io/github/issues/psawa/gecko-app)](https://github.com/psawa/gecko-app/issues)

We accept contributions, whether they intend to fix an issue or to add new functionalities. Don't hesitate to submit a pull-request.
## License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
