# nlg-eval
Evaluation code for various unsupervised automated metrics for NLG (Natural Language Generation).
It takes as input a hypothesis file, and one or more references files and outputs values of metrics.
Rows across these files should correspond to the same example.

## Metrics ##
- BLEU
- METEOR
- ROUGE
- CIDEr
- SkipThought cosine similarity
- Embedding Average cosine similarity
- Vector Extrema cosine similarity
- Greedy Matching score

## Requirements ##
Tested using

- java 1.8.0
- python 2.7
  - click 6.3
  - nltk 3.1
  - numpy 1.11.0
  - scikit-learn 0.17
  - gensim 0.12.4
  - Theano 0.8.1
  - scipy 0.17.0

## Setup ##

For the initial one-time setup, make sure java 1.8.0 is installed. After that just run:

    # install the python dependencies
    pip install -e .

    # download required data files
    ./setup.sh

## Usage ##
Once setup has completed, the metrics can be evaluated by just running:

### Standalone ###

    nlg-eval --hypothesis=examples/hyp.txt --references=examples/ref1.txt --references=examples/ref2.txt

where each line in the hypothesis file is a generated sentence and the corresponding
lines across the reference files are ground truth reference sentences for the
corresponding hypothesis.

### functional API: for the entire corpus ###

    from nlgeval import compute_metrics
    metrics_dict = compute_metrics(hypothesis='examples/hyp.txt',
                                   references=['examples/ref1.txt', 'examples/ref2.txt'])

### functional API: for only one sentence ###

    from nlgeval import compute_individual_metrics
    metrics_dict = compute_individual_metrics(references, hypothesis)

where `references` is a list of ground truth reference text strings and
`hypothesis` is the hypothesis text string.

### object oriented API for repeated calls in a script ###

    from nlgeval import NLGEval
    nlgeval = NLGEval()  # loads the models
    metrics_dict = nlgeval.evaluate(references, hypothesis)

where `references` is a list of ground truth reference text strings and
`hypothesis` is the hypothesis text string.

## Reference ##
If you use this code as part of any published research, please cite the following paper:

Shikhar Sharma, Layla El Asri, Hannes Schulz, and Jeremie Zumer.
**"Relevance of Unsupervised Metrics in Task-Oriented Dialogue for Evaluating Natural Language Generation"**
*arXiv preprint arXiv:1706.09799* (2017)

    @article{sharma2017nlgeval,
      author  = {Sharma, Shikhar and El Asri, Layla and Schulz, Hannes and Zumer, Jeremie},
      title   = {Relevance of Unsupervised Metrics in Task-Oriented Dialogue for Evaluating Natural Language Generation},
      journal = {CoRR},
      volume  = {abs/1706.09799},
      year    = {2017},
      url     = {http://arxiv.org/abs/1706.09799}
    }

## Example ##
Running

    nlg-eval --hypothesis=examples/hyp.txt --references=examples/ref1.txt --references=examples/ref2.txt

gives

    Bleu_1: 0.550000
    Bleu_2: 0.428174
    Bleu_3: 0.284043
    Bleu_4: 0.201143
    METEOR: 0.295797
    ROUGE_L: 0.522104
    CIDEr: 1.242192
    SkipThoughtsCosineSimilairty: 0.626149
    EmbeddingAverageCosineSimilairty: 0.884690
    VectorExtremaCosineSimilarity: 0.568696
    GreedyMatchingScore: 0.784205

## Important Note ##
CIDEr by default (with idf parameter set to "corpus" mode) computes IDF values using the reference sentences provided. Thus,
CIDEr score for a reference dataset with only 1 image (or example for NLG) will be zero. When evaluating using one (or few)
images, set idf to "coco-val-df" instead, which uses IDF from the MSCOCO Vaildation Dataset for reliable results. This has
not been adapted in this code. For this use-case, apply patches from
[vrama91/coco-caption](https://github.com/vrama91/coco-caption).

## Microsoft Open Source Code of Conduct ##
This project has adopted the [Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct
FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com)
with any additional questions or comments.

## License ##
See [LICENSE.md](LICENSE.md).
