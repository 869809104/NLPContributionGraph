import os
import glob
import numpy as np
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

MAX_TOKEN = 512
EPOCHS = 1
BATCH_SIZE = 8


def loadArticle(data_dir):
    articles = []
    catalogues = []

    for category in os.listdir(data_dir):  # ./datasets/training-data
        if category != 'README.md' and category != '.git' and category != 'submission.zip':
            article_category = os.path.join(data_dir, category)  # ./datasets/training-data/natural_language_inference
            # print(article_category)

            for foldname in sorted(os.listdir(article_category)):
                article_index = os.path.join(article_category, foldname)  # ./datasets/training-data/natural_language_inference/0
                # print(article_index)

                indices = article_index.split('/')
                catalogue = indices[-2] + '/' + indices[-1]


                # print(catalogue)
                temp = catalogue.split('\\')
                catalogue = temp[-2] + '\\' + temp[-1]
                print(catalogue)


                catalogues.append(catalogue)


                with open(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))[0], encoding='utf-8') as f:
                    article = f.read()
                    articles.append(article.lower())

                # break
            # break
    return articles, catalogues


def loadeEntities(data_dir):
    entities = []

    for category in os.listdir(data_dir):  # ./datasets/training-data
        if category != 'README.md' and category != '.git':
            article_category = os.path.join(data_dir, category)  # ./datasets/training-data/natural_language_inference
            # print(article_category)

            for foldname in sorted(os.listdir(article_category)):
                article_index = os.path.join(article_category, foldname)  # ./datasets/training-data/natural_language_inference/0
                # print(article_index)

                # indices = article_index.split('/')
                # catalogue = indices[-2] + '/' + indices[-1]
                # catalogues.append(catalogue)

                # with open(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))[0], encoding='utf-8') as f:
                #     article = f.read()
                #     articles.append(article.lower())

                with open(os.path.join(article_index, 'entities.txt'), encoding='utf-8') as f:
                    entity = {}
                    for line in f.readlines():
                        line = line.strip().split('\t')
                        if len(line) > 0:
                            article_sentence = int(line[0])
                            entity[article_sentence] = {}

                with open(os.path.join(article_index, 'entities.txt'), encoding='utf-8') as f:
                    for line in f.readlines():
                        line = line.strip().split('\t')
                        if len(line) > 0:
                            article_sentence = int(line[0])
                            span = line[-1]
                            if 'spans' in entity[article_sentence]:
                                entity[article_sentence]['spans'].append(span)
                            else:
                                entity[article_sentence]['spans'] = [span]
                    entities.append(entity)
                # break
            # break
    return entities


def article2ContributionSentenceAndContributionSpans(articles, entities):
    contribution_sentences = []
    contribution_spans = []
    for i, article in enumerate(articles):
        spans = entities[i]
        # print('entities.txt:', spans)

        sents = article.split('\n')[0:-1]
        contribution_span = []
        for row, sent in enumerate(sents):
            if (row + 1) in spans.keys():
                # print(row + 1)
                contribution_sentences.append(sent)
                contribution_span.append(spans[row + 1]['spans'])
        contribution_spans.append(contribution_span)
    return contribution_sentences, contribution_spans


article_dir = '../datasets/evaluation-phase1'
articles, catalogues = loadArticle(article_dir)
entity_dir = './evaluation-phase2'
entities = loadeEntities(entity_dir)
# print(articles)
# print(entities)
# print(catalogues)

contribution_sentences, contribution_spans = article2ContributionSentenceAndContributionSpans(articles, entities)
# print('contribution_sentences:', contribution_sentences)
print('contribution_spans:', contribution_spans)

all_spans = []
for article_index, contribution_span in enumerate(contribution_spans):
    article_spans = []
    for sentence_index, spans in enumerate(contribution_span):
        for span in spans:
            # print(span)
            article_spans.append(span)
    # print("*********************")
    all_spans.append(article_spans)
print(all_spans)

for i, article_catalogue in enumerate(catalogues):
    article_spans = all_spans[i]

    # 已解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/research-problem.txt'), 'w', encoding='utf-8') as f:
        for j in range(len(article_spans)):
            f.write('(Contribution||has research problem||' + article_spans[j] + ')\n')

    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/approach.txt'), 'w', encoding='utf-8') as f:
        f.write('(Contribution||has||Approach)' + '\n')
        for j in range(len(article_spans) - 2):
            f.write('(Approach||has||' + article_spans[j] + ')\n')
            f.write('(' + article_spans[j] + '||' + article_spans[j+1] + '||' + article_spans[j+2] + ')\n')

    # 已解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/model.txt'), 'w', encoding='utf-8') as f:
        f.write('(Contribution||has||Model)' + '\n')
        f.write('(Model||' + article_spans[0] + '||' + article_spans[1] + ')\n')
        for j in range(2, len(article_spans) - 1):
            f.write('(' + article_spans[j-1] + '||' + article_spans[j] + '||' + article_spans[j+1] + ')\n')

    # 已解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/code.txt'), 'w', encoding='utf-8') as f:
        for j in range(len(article_spans)):
            f.write('(Contribution||Code||' + article_spans[j] + ')\n')

    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/experimental-setup.txt'), 'w', encoding='utf-8') as f:
        f.write('(Contribution||has||Experimental setup)' + '\n')
        for j in range(len(article_spans) - 2):
            f.write('(Experimental setup||has||' + article_spans[j] + ')\n')
            f.write('(' + article_spans[j] + '||' + article_spans[j + 1] + '||' + article_spans[j + 2] + ')\n')

    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/hyperparameters.txt'), 'w', encoding='utf-8') as f:
        f.write('(Contribution||has||Hyperparameters)' + '\n')
        for j in range(len(article_spans) - 3):
            f.write('(Hyperparameters||' + article_spans[j] + '||' + article_spans[j + 1] + ')\n')
            f.write('(' + article_spans[j + 1] + '||' + article_spans[j+2] + '||' + article_spans[j + 3] + ')\n')

    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/baselines.txt'), 'w', encoding='utf-8') as f:
        f.write('(Contribution||has||Baselines)' + '\n')
        for j in range(len(article_spans) - 3):
            f.write('(Baselines||' + article_spans[j] + '||' + article_spans[j + 1] + ')\n')
            f.write('(' + article_spans[j + 1] + article_spans[j + 2] + '||' + article_spans[j + 3] + ')\n')

    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/results.txt'), 'w', encoding='utf-8') as f:
        f.write('(Contribution||has||Results)' + '\n')
        for j in range(len(article_spans) - 3):
            f.write('(Results||' + article_spans[j] + '||' + article_spans[j + 1] + ')\n')
            f.write('(' + article_spans[j + 1] + '||' + article_spans[j + 2] + '||' + article_spans[j + 3] + ')\n')

    # 已解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/tasks.txt'), 'w', encoding='utf-8') as f:
        f.write('(Contribution||has||Tasks)' + '\n')
        for j in range(len(article_spans) - 2):
            f.write('(Tasks||name' + article_spans[j] + ')\n')
            f.write('(' + article_spans[j] + '||' + article_spans[j+1] + '||' + article_spans[j+2] + ')\n')

    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/experiments.txt'), 'w', encoding='utf-8') as f:
        f.write('(Contribution||has||Experimental setup)' + '\n')
        for j in range(1, len(article_spans) - 3):
            f.write('(Experimental setup||' + article_spans[j] + '||' + article_spans[j + 1] + ')\n')
            f.write('(' + article_spans[j+1] + '||' + article_spans[j+2] + '||' + article_spans[j+3] + ')\n')

    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/ablation-analysis.txt'), 'w', encoding='utf-8') as f:
        f.write('(Contribution||has||Ablation analysis)' + '\n')
        for j in range(1, len(article_spans) - 3):
            f.write('(Ablation analysis||' + article_spans[j] + '||' + article_spans[j + 1] + ')\n')
            f.write('(' + article_spans[j + 1] + '||' + article_spans[j + 2] + '||' + article_spans[j + 3] + ')\n')
