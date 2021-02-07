# train
import os
import glob
import tensorflow as tf
from transformers import BertConfig, BertTokenizerFast, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

EPOCHS = 3
BATCH_SIZE = 8


def loadArticles(data_dir):
    articles = []
    contributions = []

    for category in os.listdir(data_dir):  # ./datasets/training-data
        if category != 'README.md' and category != '.git':
          article_category = os.path.join(data_dir, category)  # ./datasets/training-data/natural_language_inference
          # print(article_category)

          for foldname in sorted(os.listdir(article_category)):
              article_index = os.path.join(article_category, foldname)  # ./datasets/training-data/natural_language_inference/0

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
          #     break
          # break
    return articles, contributions


def article2SentenceAndLables(articles, contributions):
    sentences = []
    labels = []
    for i, article in enumerate(articles):
        contribution = contributions[i]

        sents = article.split('\n')[0:-1]
        # sents = article.split('\n')
        print(sents)
        for row, sent in enumerate(sents):
            sentences.append(sent)
            if (row + 1) in contribution:
                labels.append(1)
            else:
                labels.append(0)
    # print(sentences)
    # print(labels)
    return sentences, labels


train_data_dir = '/content/drive/My Drive/ncg/datasets/training-data'
# train_data_dir = '../datasets/training-data'

train_articles, train_contributions = loadArticles(train_data_dir)

train_sentences, train_labels = article2SentenceAndLables(train_articles, train_contributions)

train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_sentences, train_labels, test_size=.2)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2

train_encodings = tokenizer(train_sentences, truncation=True, padding=True)
val_encodings = tokenizer(val_sentences, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))
print('train data loaded:({0})'.format(len(train_labels)))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))
print('validation data loaded:({0})'.format(len(val_labels)))

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset.shuffle(len(train_labels)).batch(BATCH_SIZE),
          validation_data=val_dataset.batch(BATCH_SIZE),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE)

model.save_weights('/content/drive/MyDrive/ncg/model-SC-BERT/')
print('*****************model saved!******************')

