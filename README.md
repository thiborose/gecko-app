# GECko+
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-sa/4.0/)

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

## Installation
After cloning the repository, execute `setup.sh`. This simple script will install the requirement packages, and download the models which are too heavy for being stored on GitHub. We strongly suggest to create a dedicated virtual envirionment beforehand.

The project was tested using Python 3.7.

## Usage
To launch the web app, run `run.py`.

## Acknowledgments
Our tool implements the two following models, for tackilng, respectively, grammatical and discourse errors:

- Kostiantyn Omelianchuk, Vitaliy Atrasevych, Artem Chernodub and Oleksandr Skurzhanskyi **"GECToR -- Grammatical Error Correction: Tag, Not Rewrite"**. In Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications. [[arXiv]](https://arxiv.org/abs/2005.12592)
- Prabhumoye, Shrimai, Ruslan Salakhutdinov, and Alan W. Black. **"Topological Sort for Sentence Ordering."** In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. [[arXiv]](https://arxiv.org/abs/2005.00432)

