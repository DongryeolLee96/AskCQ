# AskCQ README

This is the repository documenting the paper
[Asking Clarification Questions to Handle Ambiguity in Open-Domain QA](https://arxiv.org/abs/2305.13808) (EMNLP 2023)
by Dongryeol Lee, Segwang Kim, Minwoo Lee, Joonsuk Park, Sang-Woo Lee, and Kyomin Jung.

* Read the [paper](https://arxiv.org/abs/2305.13808)
* Download the dataset: (https://drive.google.com/drive/folders/1bujroxJ02ym8SgEmC10IsVnCc8HAwTLH?usp=sharing) 

## Content
1. [Citation](#citation)
2. [Dataset Contents](#dataset-contents)
    * [CAmbigNQ format](#cambignq)
    * [Relevant-passages](#relevant-passages)
3. [Baseline codes](#baseline-codes)

## Citation

If you find the our task or CAmbigNQ dataset useful, please cite our paper:
```
@article{lee2023asking,
  title={Asking Clarification Questions to Handle Ambiguity in Open-Domain QA},
  author={Lee, Dongryeol and Kim, Segwang and Lee, Minwoo and Lee, Hwanhee and Park, Joonsuk and Lee, Sang-Woo and Jung, Kyomin},
  journal={arXiv preprint arXiv:2305.13808},
  year={2023}
}
```

Please also make sure to credit and cite the creators of AmbigNQ and Natural Questions,
the dataset which we built ours off of:
```
@inproceedings{ min2020ambigqa,
    title={ {A}mbig{QA}: Answering Ambiguous Open-domain Questions },
    author={ Min, Sewon and Michael, Julian and Hajishirzi, Hannaneh and Zettlemoyer, Luke },
    booktitle={ EMNLP },
    year={2020}
}

@article{ kwiatkowski2019natural,
  title={ Natural questions: a benchmark for question answering research},
  author={ Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia and Collins, Michael and Parikh, Ankur and Alberti, Chris and Epstein, Danielle and Polosukhin, Illia and Devlin, Jacob and Lee, Kenton and others },
  journal={ Transactions of the Association for Computational Linguistics },
  year={ 2019 }
}
```


## Dataset Contents

### CambigNQ

We provide our new dataset CAmbigNQ:

- cq_train.json (23.3M)
- cq_dev.json (3.8M)


`train.json` and `dev.json` files contain a list of dictionary that represents a single datapoint, with the following keys

- `id` (string): an identifier for the question, consistent with the original NQ dataset.
- `question` (string): a question. This is identical to the question in the original NQ except we postprocess the string to start uppercase and end with a question mark.
- `viewed_doc_titles` (a list of strings): a list of titles of Wikipedia pages viewed by crowdworkers during annotations. This is an underestimate, since Wikipedia pages viewed through hyperlinks are not included. Note that this should not be the input to a system. It is fine to use it as extra supervision, but please keep in mind that it is an underestimate.
- `used_queries` (a list of dictionaries): a list of dictionaries containing the search queries and results that were used by crowdworkers during annotations. Each dictionary contains `query` (a string) and `results` (a list of dictionaries containing `title` and `snippet`). Search results are obtained through the Google Search API restricted to Wikipedia (details in the paper). Note that this should not be the input to a system. It is fine to use it as extra supervision.
- `nq_doc_title` (string): an associated Wikipedia page title in the original NQ.
- `dqs` (list of strings): Disambiguated questions for a given question, annotated by AmbigNQ.
- `clarification_answers` (list of lists): Answer set of each corresponding Disambiguated Question. 
- `clarification_question` (string): Our Clarification Questions which clarify the given ambiguous question.

### Relevant-passages

We release top-100 retrieved passages for a given ambiguous question.

- rel_psg_input_ids_bart_train.pkl (106.8M)
- rel_psg_input_ids_bart_dev.pkl (13.3M)

Each file contains a list of encoded relevant passages.


## Baseline codes

Instruction for running baseline codes will be available soon!






