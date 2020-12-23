# Gecko+
![logo of gecko](https://github.com/psawa/gecko-app/blob/master/application/static/img/GECko_logo_small.png)

## More than a grammar error corector
GECko+ is a language assisting tool that will correct mistakes of various types on written texts. 
While many well settled softwares of its kind correct mistakes at the grammatical level (orthography and syntax), our novel approach allow the tool perform corrections both at **grammatical** and  **discourse** level.
<!--- add demo link when live -->
![demo](/application/static/img/demo.png?raw=true) 

### Use cases examples

<!--- add screenshot (possibly gif of correction) -->
Original text | Corrected text
------------ | -------------
This chemical is widly used in the swimming pools market. Chlorine is well known for its sanatizing properties. | Chlorine is well known for its sanatizing properties. This chemical is widly used in the swimming pools market.
Add examples... | Add examples... 

## Installation
After cloning the repository, execute `setup.sh`. This simple script will install the requirement packages, and download the models which are too heavy for being for stored on GitHub.

The project was tested using Python 3.7.

## Usage
To launch the web app, run `run.py`

## Acknowledgments
- Kostiantyn Omelianchuk, Vitaliy Atrasevych, Artem Chernodub and Oleksandr Skurzhanskyi **"GECToR -- Grammatical Error Correction: Tag, Not Rewrite"**. In Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications. [[arXiv]](https://arxiv.org/abs/2005.12592)
- Prabhumoye, Shrimai, Ruslan Salakhutdinov, and Alan W. Black. **"Topological Sort for Sentence Ordering."** In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. [[arXiv]](https://arxiv.org/abs/2005.00432)
