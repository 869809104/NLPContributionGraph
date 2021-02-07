# submission
import os
import glob
import numpy as np
import tensorflow as tf
from transformers import BertConfig, BertTokenizerFast, TFBertForTokenClassification

EPOCHS = 3
BATCH_SIZE = 8
MAX_TOKEN = 512


def loadArticles(data_dir):
    articles = []
    articles_raw = []
    catalogues = []

    for category in sorted(os.listdir(data_dir)):
        if category != 'README.md' and category != '.git' and category != 'submission.zip':
            article_category = os.path.join(data_dir, category)
            # print(article_category)

            for foldname in sorted(
                    os.listdir(article_category)):  # ./datasets/testing-data/natural_language_inference/0
                if foldname != '.ipynb_checkpoints':
                    article_index = os.path.join(article_category, foldname)
                    # print(article_index)

                    indices = article_index.split('/')
                    catalogue = indices[-2] + '/' + indices[-1]
                    catalogues.append(catalogue)

                    with open(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))[0], encoding='utf-8') as f:
                        article = f.read()
                        articles.append(article.lower())
                        articles_raw.append(article)

                    # break
            # break
    return articles, articles_raw, catalogues


def loadSentences(data_dir):
    contributions = []

    for category in sorted(os.listdir(data_dir)):
        if category != 'README.md' and category != '.git' and category != 'submission.zip':
            article_category = os.path.join(data_dir, category)
            # print(article_category)

            for foldname in sorted(os.listdir(article_category)):  # ./submission/natural_language_inference/0
                article_index = os.path.join(article_category, foldname)
                # print(article_index)

                # if foldname != '.ipynb_checkpoints':
                with open(os.path.join(article_index, 'sentences.txt'), encoding='utf-8') as f:
                    contribution = []
                    for line in f.readlines():
                        article_contribution = int(line.strip())
                        contribution.append(article_contribution)
                    contributions.append(contribution)
                # break
        # break
    return contributions


def article2ContributionSentence(articles, articles_raw, contributions):
    contribution_sentences = []
    contribution_sentences_raw = []
    contributions_ascend = []
    sent_count = []
    for i, article in enumerate(articles):
        contribution = contributions[i]
        # print('sentences.txt:', contribution)

        count = 0
        sents = article.split('\n')[0:-1]
        for row, sent in enumerate(sents):
            if (row + 1) in contribution:
                # print(row + 1)
                # print(sent)
                contribution_sentences.append(sent)
                count += 1
        sent_count.append(count)

        sents_raw = articles_raw[i].split('\n')[0:-1]
        contribution_sentence_raw = []
        contribution_ascend = []
        for row, sent_raw in enumerate(sents_raw):
            if (row + 1) in contribution:
                contribution_sentence_raw.append(sent_raw)
                contribution_ascend.append(row + 1)
        contribution_sentences_raw.append(contribution_sentence_raw)
        contributions_ascend.append(contribution_ascend)
    return contribution_sentences, contribution_sentences_raw, contributions_ascend, sent_count


def alignSpansBySentence(contribution_sentences):
    sent_tokens = []
    sent_token_ids = []
    sent_token_spans = []

    maxTokenLen = 0
    for i, sent in enumerate(contribution_sentences):
        # print(sent)
        tokens = tokenizer.tokenize(sent)
        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        # print('tokens:', tokens)
        sent_tokens.append(tokens)

        token_spans = []
        token_ids = np.zeros(len(tokens), dtype='int32')

        end = 0
        for index, token in enumerate(tokens):
            token_id = tokenizer.convert_tokens_to_ids(token)
            token_ids[index] = token_id

            if token in ['[CLS]', '[UNK]']:
                token_spans.append((end, end))
            elif token == '[SEP]':
                # end = end + len(sent)
                token_spans.append((end, end))
            else:
                token = token.replace('##', '')
                start = sent.find(token, end)
                end = start + len(token)

                token_spans.append((start, end))

        # print('token_ids:', token_ids)
        sent_token_ids.append(token_ids)

        # print('token_spans:', token_spans)
        sent_token_spans.append(token_spans)

        maxTokenLen = len(tokens) if maxTokenLen < len(tokens) else maxTokenLen

        # if i == 3:
        #     break

    # print('max token length: {0}'.format(maxTokenLen))
    # print(sent_tokens)
    # print(sent_token_ids)
    # print(sent_token_spans)
    return maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans


def chunkData(maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans):
    input_ids = []
    input_masks = []
    token_spans = sent_token_spans

    if maxTokenLen > MAX_TOKEN:
        maxTokenLen = MAX_TOKEN
    for i, token_ids in enumerate(sent_token_ids):
        ids = np.zeros(maxTokenLen, dtype=int)
        ids[0:len(token_ids)] = token_ids
        input_ids.append(ids)

        mask = np.copy(ids)
        mask[mask > 0] = 1
        input_masks.append(mask)

    # print(input_ids)
    # print(input_masks)

    return input_ids, input_masks, token_spans


def buildData(contribution_sentences):
    maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans = alignSpansBySentence(contribution_sentences)
    input_ids, input_masks, token_spans = chunkData(maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans)

    x = dict(
        input_ids=np.array(input_ids, dtype=np.int32),
        attention_mask=np.array(input_masks, dtype=np.int32),
        token_type_ids=np.zeros(shape=(len(input_ids), maxTokenLen))
    )
    return x, token_spans, maxTokenLen


test_article_dir = '/content/drive/MyDrive/ncg/datasets/evaluation-phase1'
test_sentence_dir = '/content/drive/MyDrive/ncg/datasets/evaluation-phase2'  # 使用phase2提供的sentences.txt
articles, articles_raw, catalogues = loadArticles(test_article_dir)
contributions = loadSentences(test_sentence_dir)
# print(articles)
# print(articles_raw)
# print(contributions)
# print(catalogues)
test_sentences, contribution_sentences_raw, contributions_ascend, sent_count = article2ContributionSentence(articles,
                                                                                                            articles_raw,
                                                                                                            contributions)
# print(test_sentences)
# print(contribution_sentences_raw)
# print(contributions_ascend)
# print(sent_count)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2

test_x, token_spans, maxTokenLen = buildData(test_sentences)
print('test data loaded:({0})'.format(len(test_sentences)))

model = TFBertForTokenClassification.from_pretrained('bert-base-uncased', config=config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.load_weights('/content/drive/MyDrive/ncg/model-SI-BERT/')
print('***********************model loaded!***************************')

np.set_printoptions(threshold=1e6)
y_pred = model.predict(test_x, batch_size=BATCH_SIZE)
# print(y_pred)
# print(y_pred[0])
# print(y_pred[0].shape)
# print(type(y_pred[0]))
result = np.argmax(y_pred[0], axis=-1)
# print(result)
# print(type(result))
# print(result[0])
# print(list(result[0]))
# print(list(result[1]))
# print(list(result[2]))

# print(len(result))


start = 0
for article_index, sent_num in enumerate(sent_count):
    end = start + sent_num

    article_tags = []
    article_spans = []
    article_catalogue = catalogues[article_index]

    # if not os.path.exists('/content/drive/MyDrive/ncg/submission/' + article_catalogue):
    #     os.makedirs('/content/drive/MyDrive/ncg/submission/' + article_catalogue)

    if not os.path.exists('/content/drive/MyDrive/ncg/phase2/evaluation-phase2/' + article_catalogue + '/triples'):
        os.makedirs('/content/drive/MyDrive/ncg/phase2/evaluation-phase2/' + article_catalogue + '/triples')

    for j in range(start, end):
        article_tags.append(list(result[j]))
        article_spans.append(token_spans[j])
    # print(article_tags) # [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
    # print(article_spans) # [[(0, 0), (0, 8), (9, 15), (16, 31), (32, 37), (38, 40), (40, 41), (42, 44), (44, 48), (48, 49), (50, 51), (52, 56), (56, 59), (60, 63), (64, 75), (76, 83), (84, 95), (95, 95)], [(0, 0), (0, 5), (6, 10), (11, 15), (16, 18), (19, 27), (28, 30), (31, 36), (37, 43), (44, 52), (53, 56), (57, 59), (59, 60), (61, 62), (63, 67), (68, 73), (74, 81), (82, 84), (85, 86), (87, 92), (93, 99), (100, 107), (108, 120), (121, 125), (126, 129), (130, 132), (133, 137), (138, 140), (141, 146), (147, 149), (150, 153), (154, 166), (167, 173), (174, 175), (176, 181), (182, 184), (184, 185), (186, 192), (193, 194), (194, 194)]]

    # 合并序列
    predict_article_spans = []
    for m in range(len(article_tags)):  # 句数
        predict_sentence_spans = []
        for n in range(maxTokenLen):  # 词数
            if article_tags[m][n] == 1:
                spans = article_spans[m][n]
                # print(spans)
                predict_sentence_spans.append(spans)
        predict_article_spans.append(predict_sentence_spans)
    # print(predict_article_spans) # [[(64, 75), (84, 95)], [(57, 59), (59, 60), (87, 92), (93, 99), (100, 107), (108, 120), (133, 137), (138, 140), (147, 149), (154, 166), (167, 173), (174, 175), (176, 181), (182, 184), (184, 185), (186, 192)]]

    predict_article_spans_list = []
    for i in range(len(predict_article_spans)):
        sentence_spans = []
        for j in range(len(predict_article_spans[i])):
            sentence_spans.append(list(predict_article_spans[i][j]))
        # print(sentence_spans)
        predict_article_spans_list.append(sentence_spans)
    # print(predict_article_spans_list) # [[[64, 75], [84, 95]], [[57, 59], [59, 60], [87, 92], [93, 99], [100, 107], [108, 120], [133, 137], [138, 140], [147, 149], [154, 166], [167, 173], [174, 175], [176, 181], [182, 184], [184, 185], [186, 192]]]

    union_predict_article_spans = []
    for i in range(len(predict_article_spans_list)):
        sentence_spans = predict_article_spans_list[i]
        union_sentence_spans = []
        for sentence_span in sentence_spans:
            union_sentence_spans.append(list(sentence_span))
        # print(union_sentence_spans)
        for j in range(len(union_sentence_spans) - 1):
            for i in range(len(union_sentence_spans) - 1):
                # print(i)
                if union_sentence_spans[i][1] == union_sentence_spans[i + 1][0] or union_sentence_spans[i + 1][0] - \
                        union_sentence_spans[i][1] == 1:
                    union_sentence_spans[i][1] = union_sentence_spans[i + 1][1]
                    del union_sentence_spans[i + 1]
                    # print(union_sentence_spans)
                    break
            # print(union_sentence_spans)
        union_predict_article_spans.append(union_sentence_spans)
    # print(union_predict_article_spans) # [[[64, 75], [84, 95]], [[57, 60], [87, 120], [133, 140], [147, 149], [154, 192]]]

    contribution_sentence_row = []
    contribution_sentence_content = []
    for sentence_index in range(sent_num):
        contribution_sentence_row.append(contributions_ascend[article_index][sentence_index])
        contribution_sentence_content.append(contribution_sentences_raw[article_index][sentence_index])
    # print(contribution_sentence_row)
    # print(contribution_sentence_content)

    # with open(os.path.join('/content/drive/MyDrive/ncg/submission/' + article_catalogue, 'entities.txt'), 'w', encoding='utf-8') as f:
    with open(os.path.join('/content/drive/MyDrive/ncg/phase2/evaluation-phase2/' + article_catalogue, 'entities.txt'),
              'w', encoding='utf-8') as f:

        for m in range(len(union_predict_article_spans)):  # 句数
            for n in range(len(union_predict_article_spans[m])):
                spans = union_predict_article_spans[m][n]
                # print(spans)
                # contribution_sentence_row = contributions[article_index][m]
                # print(contribution_sentence_row)
                contribution_sentence_content_spans = contribution_sentence_content[m][spans[0]:spans[1]]
                # print(contribution_sentence_content)
                # print(str(contribution_sentence_row) + '\t' + str(spans[0]) + '\t' + str(spans[1]) + '\t' + contribution_sentence_content + '\n')
                f.write(str(contribution_sentence_row[m]) + '\t' + str(spans[0]) + '\t' + str(
                    spans[1]) + '\t' + contribution_sentence_content_spans + '\n')

    start = end

