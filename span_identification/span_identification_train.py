# train
import os
import glob
import numpy as np
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertForTokenClassification
from sklearn.model_selection import train_test_split

MAX_TOKEN = 512
EPOCHS = 3
BATCH_SIZE = 8


def loadArticles(train_data_dir, trial_data_dir):
    articles = []
    entities = []

    for category in os.listdir(train_data_dir):  # ./datasets/training-data
        if category != 'README.md' and category != '.git':
            article_category = os.path.join(train_data_dir, category)  # ./datasets/training-data/natural_language_inference
            # print(article_category)

            for foldname in sorted(os.listdir(article_category)):
                article_index = os.path.join(article_category,
                                             foldname)  # ./datasets/training-data/natural_language_inference/0
                # print(article_index)
                # print(glob.glob(os.path.join(article_index, '*-Stanza-out.txt')))
                # if len(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))) != 0:
                with open(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))[0], encoding='utf-8') as f:
                    article = f.read()
                    articles.append(article.lower())

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
                            span = (int(line[1]), int(line[2]))
                            if 'spans' in entity[article_sentence]:
                                entity[article_sentence]['spans'].append(span)
                            else:
                                entity[article_sentence]['spans'] = [span]
                    entities.append(entity)
            #     break
            # break
    # print(len(articles))  # 237
    # print(len(entities))  # 237

    for category in os.listdir(trial_data_dir):  # ./datasets/training-data
        if category != 'README.md' and category != '.git':
            article_category = os.path.join(trial_data_dir, category)  # ./datasets/training-data/natural_language_inference
            # print(article_category)

            for foldname in sorted(os.listdir(article_category)):
                article_index = os.path.join(article_category,
                                             foldname)  # ./datasets/training-data/natural_language_inference/0
                # print(article_index)
                # print(glob.glob(os.path.join(article_index, '*-Stanza-out.txt')))
                # if len(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))) != 0:
                with open(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))[0], encoding='utf-8') as f:
                    article = f.read()
                    articles.append(article.lower())

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
                            span = (int(line[1]), int(line[2]))
                            if 'spans' in entity[article_sentence]:
                                entity[article_sentence]['spans'].append(span)
                            else:
                                entity[article_sentence]['spans'] = [span]
                    entities.append(entity)
            #     break
            # break
    return articles, entities


def article2ContributionSentenceAndContributionSpans(articles, entities):
    contribution_sentences = []
    contribution_spans = []
    for i, article in enumerate(articles):
        spans = entities[i]
        # print('entities.txt:', spans)

        sents = article.split('\n')[0:-1]
        for row, sent in enumerate(sents):
            if (row + 1) in spans.keys():
                # print(row + 1)
                contribution_sentences.append(sent)
                contribution_spans.append(spans[row + 1]['spans'])
    # print(len(contribution_sentences))
    # print(len(contribution_spans))
    return contribution_sentences, contribution_spans


def alignSpansBySentence(contribution_sentences, contribution_spans):
    sent_tokens = []
    sent_token_ids = []
    sent_token_spans = []
    sent_token_tags = []

    maxTokenLen = 0
    for i, sent in enumerate(contribution_sentences):
        # print(sent)
        sent_spans = contribution_spans[i]
        tokens = tokenizer.tokenize(sent)
        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        # print('tokens:', tokens)
        sent_tokens.append(tokens)

        token_spans = []
        token_ids = np.zeros(len(tokens), dtype='int32')
        token_tags = np.zeros(len(tokens), dtype='int')

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
                # print('sent_spans:', sent_spans)
                # print('sent_spans长度:', len(sent_spans))
                if len(sent_spans) != 0:
                    for sent_span in sent_spans:
                        # print(sent_span)
                        if start >= sent_span[0] and end <= sent_span[1]:
                            token_tags[index] = 1
                            # print(sent[start:end], end=' ')
                token_spans.append((start, end))
        # print()

        # print('token_ids:', token_ids)
        sent_token_ids.append(token_ids)

        # print('token_spans:', token_spans)
        sent_token_spans.append(token_spans)

        # print('token_tags:', token_tags)
        sent_token_tags.append(token_tags)

        maxTokenLen = len(tokens) if maxTokenLen < len(tokens) else maxTokenLen

        # if i == 3:
        #     break

    # print('max token length: {0}'.format(maxTokenLen))
    # print(sent_tokens)
    # print(sent_token_ids)
    # print(sent_token_spans)
    # print(sent_token_tags)
    return maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans, sent_token_tags


def chunkData(maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans, sent_token_tags):
    input_ids = []
    input_masks = []
    input_tags = []
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

        token_tags = sent_token_tags[i]
        tags = np.zeros(maxTokenLen, dtype=int)
        tags[0:len(token_tags)] = token_tags
        input_tags.append(tags)

    # print(input_ids)
    # print(input_masks)
    # print(input_tags)
    # print(token_spans)
    return input_ids, input_masks, input_tags, token_spans


def buildData(contribution_sentences, contribution_spans):
    maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans, sent_token_tags \
        = alignSpansBySentence(contribution_sentences, contribution_spans)
    input_ids, input_masks, input_tags, token_spans = \
        chunkData(maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans, sent_token_tags)

    x = dict(
        input_ids=np.array(input_ids, dtype=np.int32),
        attention_mask=np.array(input_masks, dtype=np.int32),
        token_type_ids=np.zeros(shape=(len(input_ids), maxTokenLen))
    )
    y = np.array(input_tags, dtype=np.int32)
    return x, y


train_data_dir = '/content/drive/MyDrive/ncg/datasets/training-data'
trial_data_dir = '/content/drive/MyDrive/ncg/datasets/trial-data'
articles, entities = loadArticles(train_data_dir, trial_data_dir)
# print(articles)
# print(sentences)
# print(entities)

train_sentences, train_spans = article2ContributionSentenceAndContributionSpans(articles, entities)
# print('contribution_sentences:', train_sentences)
# print('contribution_spans:', train_spans)


train_sentences, val_sentences, train_spans, val_spans = train_test_split(train_sentences, train_spans, test_size=.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2

# maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans, sent_token_tags = \
#     alignSpansBySentence(train_sentences, train_spans)
# input_ids, input_masks, input_tags, token_spans = \
#     chunkData(maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans, sent_token_tags)

train_x, train_y = buildData(train_sentences, train_spans)
train_dataset = tf.data.Dataset.from_tensor_slices((
    train_x,
    train_y
))
print('train data loaded:({0})'.format(len(train_y)))

val_x, val_y = buildData(val_sentences, val_spans)
val_dataset = tf.data.Dataset.from_tensor_slices((
    val_x,
    val_y
))
print('validation data loaded:({0})'.format(len(val_y)))

model = TFBertForTokenClassification.from_pretrained('bert-base-uncased', config=config)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset.shuffle(len(train_y)).batch(BATCH_SIZE),
          validation_data=val_dataset.batch(BATCH_SIZE),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE)

model.save_weights('/content/drive/MyDrive/ncg/model-SI-BERT/')
print('*****************model saved!******************')