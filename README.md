This work seeks to determine if the relationships between entities in a sentence is significant or not.

It accomplishes this by constructing a model that predicts significance given a sentence and two named entities in it, and use Mutual Information and Pointwise Mutual Information scores to validate the predictions.

The hypothesis suggests that when MI/PMI scores are high, one can learn about interesting relationships between entities.

This work's focus is on infrastructure and is intended to construct the necessary framework for training and evaluating the described model, to produce initial results, and to facilitate further research.

Each of the 10 work parts is supported by corresponding scripts, which can be found within this Git repository. Additionally, you can access all the generated outputs in the [project's storage drive](https://drive.google.com/drive/u/4/folders/1v3YdVXgeByow9xkSgPJom6rx2KxB9QSu).

For comprehensive details, references, and guidance, please use the [report pdf](https://github.com/mhornstein/NESRR/blob/main/report.pdf).

---

The scripts were tested in the following config:
* OS: Windows
* Python version: 3.10.8
* Spacy version: 3.5.3 with en_core_web_lg model installed
