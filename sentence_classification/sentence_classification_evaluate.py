# evaluate
import os
import glob
import numpy as np
import tensorflow as tf
from transformers import BertConfig, BertTokenizerFast, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

EPOCHS = 3
BATCH_SIZE = 8


def loadArticles(data_dir):
    articles = []
    contributions = []

    for category in os.listdir(data_dir):
        if category != 'README.md' and category != '.git' and category != 'submission.zip':
          article_category = os.path.join(data_dir, category)
          # print(article_category)

          for foldname in sorted(os.listdir(article_category)):
              article_index = os.path.join(article_category, foldname)

              # print(glob.glob(os.path.join(article_index, '*-Stanza-out.txt')))
              # if len(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))) != 0:
              with open(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))[0], encoding='utf-8') as f:
                  article = f.read()
                  articles.append(article.lower())

              with open(os.path.join(article_index, 'sentences.txt'), encoding='utf-8') as f:
                  contribution = []
                  for line in f.readlines():
                      article_contribution = int(line.strip())
                      contribution.append(article_contribution)
                  contributions.append(contribution)
              # break
          # break
    return articles, contributions


def article2SentenceAndLables(articles, contributions):
    sentences = []
    labels = []
    for i, article in enumerate(articles):
        contribution = contributions[i]

        sents = article.split('\n')[0:-1]
        # sents = article.split('\n')
        for row, sent in enumerate(sents):
            sentences.append(sent)
            if (row + 1) in contribution:
                labels.append(1)
            else:
                labels.append(0)
    # print(sentences)
    # print(labels)
    return sentences, labels


test_data_dir = '/content/drive/My Drive/ncg/datasets/trial-data'
# test_data_dir = '../datasets/trial-data'

articles, contributions = loadArticles(test_data_dir)

test_sentences, test_labels = article2SentenceAndLables(articles, contributions)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2

test_encodings = tokenizer(test_sentences, truncation=True, padding=True)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
))
print('test data loaded:({0})'.format(len(test_labels)))

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-8),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.fit(train_dataset.shuffle(len(train_labels)).batch(BATCH_SIZE),
#           validation_data=val_dataset.batch(BATCH_SIZE),
#           epochs=EPOCHS,
#           batch_size=BATCH_SIZE)

model.load_weights('/content/drive/MyDrive/ncg/model-SC-BERT/')
print('*****************model loaded!******************')

# y_pred = model.predict(test_dataset.batch(BATCH_SIZE))
# result = np.argmax(y_pred[0], axis=1)
# print(result)

test_loss, test_accuracy = model.evaluate(test_dataset.batch(BATCH_SIZE), batch_size=BATCH_SIZE)
print('test loss:', test_loss)
print('test accuracy:', test_accuracy)

