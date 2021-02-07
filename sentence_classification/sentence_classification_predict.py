# submission
import os
import glob
import tensorflow as tf
import numpy as np
from transformers import BertConfig, BertTokenizerFast, TFBertForSequenceClassification

EPOCHS = 3
BATCH_SIZE = 8


def loadArticles(data_dir):
    articles = []
    catalogues = []

    for category in os.listdir(data_dir):
        if category != 'README.md' and category != '.git' and category != 'submission.zip':
            article_category = os.path.join(data_dir, category)
            # print(article_category)

            for foldname in sorted(os.listdir(article_category)):
                article_index = os.path.join(article_category, foldname)  # ./datasets/testing-data/natural_language_inference/0
                indices = article_index.split('/')
                catalogue = indices[-2] + '/' + indices[-1]
                catalogues.append(catalogue)

                with open(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))[0], encoding='utf-8') as f:  # ./datasets/training-data/natural_language_inference/0/1606.01549v3-Stanza-out.txt
                    article = f.read()
                    articles.append(article.lower())

            #     break
            # break
    return articles, catalogues


def article2SentenceAndLables(articles):
    article_sentences = []
    article_labels = []
    sent_count = []
    for i, article in enumerate(articles):
        article_sentence = []
        article_label = []
        sents = article.split('\n')[0:-1]
        # sents = article.split('\n')
        print(sents)
        count = 0
        for row, sent in enumerate(sents):
            article_sentence.append(sent)
            article_label.append(-1)
            count += 1
        article_sentences.append(article_sentence)
        article_labels.append(article_label)
        sent_count.append(count)
    return article_sentences, article_labels, sent_count


test_data_dir = '/content/drive/MyDrive/ncg/datasets/evaluation-phase1'
# test_data_dir = '../datasets/evaluation-phase1'

articles, catalogues = loadArticles(test_data_dir)
# print(articles)
# print(catalogues)

article_sentences, article_labels, sent_count = article2SentenceAndLables(articles)
# print(article_sentences)
# print(article_labels)
# print(sent_count)


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2

test_sentences = []
test_labels = []
for i, article_sentence in enumerate(article_sentences):
    for j, sentence in enumerate(article_sentence):
        test_sentences.append(sentence)
        test_labels.append(article_labels[i][j])
# print(test_sentences)
# print(test_labels)

test_encodings = tokenizer(test_sentences, truncation=True, padding=True)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
))

print('test data loaded: ({0})'.format(len(test_labels)))

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.load_weights('/content/drive/MyDrive/ncg/model-SC-BERT/')

y_pred = model.predict(test_dataset.batch(BATCH_SIZE))
# print(y_pred)
# print(y_pred[0])
# print(y_pred[0].shape)
np.set_printoptions(threshold=1e6)
result = np.argmax(y_pred[0], axis=-1)
# print(result)
# with open('/content/drive/MyDrive/ncg/result.txt', 'w', encoding='utf-8') as f:
#   f.write(str(result))
# f.close()

if not os.path.exists('/content/drive/MyDrive/ncg/submission'):
    os.makedirs('/content/drive/MyDrive/ncg/submission')

start = 0
for i, count in enumerate(sent_count):
    end = start + count
    article_label = result[start:end]
    article_catalogue = catalogues[i]

    if not os.path.exists('/content/drive/MyDrive/ncg/submission/' + article_catalogue):
        os.makedirs('/content/drive/MyDrive/ncg/submission/' + article_catalogue)

    with open(os.path.join('/content/drive/MyDrive/ncg/submission/' + article_catalogue, 'sentences.txt'), 'w', encoding='utf-8') as f:
        for j in range(len(article_label)):
            if article_label[j] == 1:
                f.write(str(j + 1) + '\n')
    start = end

