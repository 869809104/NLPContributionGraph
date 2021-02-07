import os
import glob


def loadArticle(data_dir):
    articles = []
    catalogues = []

    for category in os.listdir(data_dir):  # ./datasets/training-data
        if category != 'README.md' and category != '.git' and category != 'submission.zip':
            article_category = os.path.join(data_dir, category)  # ./datasets/training-data/natural_language_inference
            # print(article_category)

            for foldname in sorted(os.listdir(article_category)):
                article_index = os.path.join(article_category,
                                             foldname)  # ./datasets/training-data/natural_language_inference/0
                # print(article_index)

                indices = article_index.split('/')
                catalogue = indices[-2] + '/' + indices[-1]

                # print(catalogue)
                temp = catalogue.split('\\')
                catalogue = temp[-2] + '\\' + temp[-1]
                # print(catalogue)

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
                article_index = os.path.join(article_category,
                                             foldname)  # ./datasets/training-data/natural_language_inference/0
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
# article_dir = '../datasets/trial-data'
articles, catalogues = loadArticle(article_dir)
entity_dir = './evaluation-phase2-pa'
# entity_dir = '../datasets/trial-data'
entities = loadeEntities(entity_dir)

# print(articles)
# print(entities)
# print(catalogues)

contribution_sentences, contribution_spans = article2ContributionSentenceAndContributionSpans(articles, entities)
# print('contribution_sentences:', contribution_sentences)
print('contribution_spans:', contribution_spans)

for article_index, article_spans in enumerate(contribution_spans):
    article_catalogue = catalogues[article_index]
    # print(article_spans)

    # 已解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/research-problem.txt'), 'w',
              encoding='utf-8') as f:
        for sentence_index, sentence_spans in enumerate(article_spans):
            # print(sentence_spans)
            for spans in sentence_spans:
                # print(article_catalogue)
                # print('(Contribution||has research problem||' + spans + ')')
                f.write('(Contribution||has research problem||' + spans + ')\n')

    # 已解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/approach.txt'), 'w',
              encoding='utf-8') as f:
        # print('(Contribution||has||Approach)')
        f.write('(Contribution||has||Approach)\n')
        for sentence_index, sentence_spans in enumerate(article_spans):
            if len(sentence_spans) >= 2:
                # print('(Approach||' + sentence_spans[0] + '||' + sentence_spans[1] + ')')
                f.write('(Approach||' + sentence_spans[0] + '||' + sentence_spans[1] + ')\n')
                start = 1
                while start <= len(sentence_spans) - 3:
                    # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                    f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                    start += 2

    # 1/2概率
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/model.txt'), 'w',
              encoding='utf-8') as f:
        # print('(Contribution||has||Model)')
        f.write('(Contribution||has||Model)\n')
        for sentence_index, sentence_spans in enumerate(article_spans):
            if len(sentence_spans) >= 2:
                # print('(Model||' + sentence_spans[0] + '||' + sentence_spans[1] + ')')
                f.write('(Model||' + sentence_spans[0] + '||' + sentence_spans[1] + ')\n')
                start = 1
                while start <= len(sentence_spans) - 3:
                    # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                    f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                    start += 2
        for sentence_index, sentence_spans in enumerate(article_spans):
            # print('(Model||has||' + sentence_spans[0] + ')')
            f.write('(Model||has||' + sentence_spans[0] + ')\n')
            start = 0
            while start <= len(sentence_spans) - 3:
                # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                start += 2

    # 已解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/code.txt'), 'w', encoding='utf-8') as f:
        for sentence_index, sentence_spans in enumerate(article_spans):
            # print(sentence_spans)
            for spans in sentence_spans:
                # print(article_catalogue)
                # print('(Contribution||Code||' + spans + ')')
                f.write('(Contribution||Code||' + spans + ')\n')

    # 未解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/experimental-setup.txt'), 'w', encoding='utf-8') as f:
        # print('(Contribution||has||Experimental setup)')
        f.write('(Contribution||has||Experimental setup)\n')
        for sentence_index, sentence_spans in enumerate(article_spans):
            if len(sentence_spans) >= 2:
                # print('(Experimental setup||' + sentence_spans[0] + '||' + sentence_spans[1] + ')')
                f.write('(Experimental setup||' + sentence_spans[0] + '||' + sentence_spans[1] + ')\n')
                start = 1
                while start <= len(sentence_spans) - 3:
                    # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                    f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                    start += 2
        for sentence_index, sentence_spans in enumerate(article_spans):
            # print('(Experimental setup||has||' + sentence_spans[0] + ')')
            f.write('(Experimental setup||has||' + sentence_spans[0] + ')\n')
            start = 0
            while start <= len(sentence_spans) - 3:
                # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                start += 2

    # 未解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/hyperparameters.txt'), 'w',
                  encoding='utf-8') as f:
        # print('(Contribution||has||Hyperparameters)')
        f.write('(Contribution||has||Hyperparameters)\n')
        for sentence_index, sentence_spans in enumerate(article_spans):
            if len(sentence_spans) >= 2:
                # print('(Hyperparameters||' + sentence_spans[0] + '||' + sentence_spans[1] + ')')
                f.write('(Hyperparameters||' + sentence_spans[0] + '||' + sentence_spans[1] + ')\n')
                start = 1
                while start <= len(sentence_spans) - 3:
                    # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                    f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                    start += 2
        for sentence_index, sentence_spans in enumerate(article_spans):
            # print('(Hyperparameters||has||' + sentence_spans[0] + ')')
            f.write('(Hyperparameters||has||' + sentence_spans[0] + ')\n')
            start = 0
            while start <= len(sentence_spans) - 3:
                # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                start += 2

    # 已解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/baselines.txt'), 'w',
                encoding='utf-8') as f:
        # print('(Contribution||has||Baselines)')
        f.write('(Contribution||has||Baselines)\n')
        for sentence_index, sentence_spans in enumerate(article_spans):
            if len(sentence_spans) >= 2:
                # print('(Baselines||' + sentence_spans[0] + '||' + sentence_spans[1] + ')')
                f.write('(Baselines||' + sentence_spans[0] + '||' + sentence_spans[1] + ')\n')
                start = 1
                while start <= len(sentence_spans) - 3:
                    # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                    f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                    start += 2
        for sentence_index, sentence_spans in enumerate(article_spans):
            # if len(sentence_spans) > 2:
            # print('(Baselines||has||' + sentence_spans[0] + ')')
            f.write('(Baselines||has||' + sentence_spans[0] + ')\n')
            start = 0
            while start <= len(sentence_spans) - 3:
                # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                start += 2

    # 半解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/results.txt'), 'w', encoding='utf-8') as f:
        # print('(Contribution||has||Results)')
        f.write('(Contribution||has||Results)\n')
        for sentence_index, sentence_spans in enumerate(article_spans):
            if len(sentence_spans) >= 2:
                # print('(Results||' + sentence_spans[0] + '||' + sentence_spans[1] + ')')
                f.write('(Results||' + sentence_spans[0] + '||' + sentence_spans[1] + ')\n')
                start = 1
                while start <= len(sentence_spans) - 3:
                    # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                    f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                    start += 2
        for sentence_index, sentence_spans in enumerate(article_spans):
            # print('(Results||has||' + sentence_spans[0] + ')')
            f.write('(Results||has||' + sentence_spans[0] + ')\n')
            start = 0
            while start <= len(sentence_spans) - 3:
                # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                start += 2

    # 半解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/tasks.txt'), 'w', encoding='utf-8') as f:
        # print('(Contribution||has||Tasks)')
        f.write('(Contribution||has||Tasks)\n')
        for sentence_index, sentence_spans in enumerate(article_spans):
            if len(sentence_spans) >= 2:
                # print('(Tasks||' + sentence_spans[0] + '||' + sentence_spans[1] + ')')
                f.write('(Tasks||' + sentence_spans[0] + '||' + sentence_spans[1] + ')\n')
                start = 1
                while start <= len(sentence_spans) - 3:
                    # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                    f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                    start += 2
        for sentence_index, sentence_spans in enumerate(article_spans):
            # print('(Tasks||has||' + sentence_spans[0] + ')')
            f.write('(Tasks||has||' + sentence_spans[0] + ')\n')
            start = 0
            while start <= len(sentence_spans) - 3:
                # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                start += 2
    # 未解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/experiments.txt'), 'w', encoding='utf-8') as f:
        # print('(Contribution||has||Experiments)')
        f.write('(Contribution||has||Experiments)\n')
        for sentence_index, sentence_spans in enumerate(article_spans):
            if len(sentence_spans) >= 2:
                # print('(Experiments||' + sentence_spans[0] + '||' + sentence_spans[1] + ')')
                f.write('(Model||' + sentence_spans[0] + '||' + sentence_spans[1] + ')\n')
                start = 1
                while start <= len(sentence_spans) - 3:
                    # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                    f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                    start += 2
        for sentence_index, sentence_spans in enumerate(article_spans):
            # print('(Experiments||has||' + sentence_spans[0] + ')')
            f.write('(Experiments||has||' + sentence_spans[0] + ')\n')
            start = 0
            while start <= len(sentence_spans) - 3:
                # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                start += 2

    # 未解决
    with open(os.path.join('./evaluation-phase2/' + article_catalogue, 'triples/ablation-analysis.txt'), 'w', encoding='utf-8') as f:
        # print('(Contribution||has||Ablation analysis)')
        f.write('(Contribution||has||Ablation analysis)\n')
        for sentence_index, sentence_spans in enumerate(article_spans):
            if len(sentence_spans) >= 2:
                # print('(Ablation analysis||' + sentence_spans[0] + '||' + sentence_spans[1] + ')')
                f.write('(Ablation analysis||' + sentence_spans[0] + '||' + sentence_spans[1] + ')\n')
                start = 1
                while start <= len(sentence_spans) - 3:
                    # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                    f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                    start += 2
        for sentence_index, sentence_spans in enumerate(article_spans):
            # print('(Ablation analysis||has||' + sentence_spans[0] + ')')
            f.write('(Ablation analysis||has||' + sentence_spans[0] + ')\n')
            start = 0
            while start <= len(sentence_spans) - 3:
                # print('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')')
                f.write('(' + sentence_spans[start] + '||' + sentence_spans[start + 1] + '||' + sentence_spans[start + 2] + ')\n')
                start += 2
