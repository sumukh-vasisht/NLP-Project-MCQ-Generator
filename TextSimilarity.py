from xml.parsers.expat import model
import nltk
import gensim
# nltk.download('stopwords')
# nltk.download('popular')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from summarizer import Summarizer
from nltk.corpus import wordnet as wn
from math import sqrt, pow, exp
import spacy
import en_core_web_sm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import tensorflow as tf
import tensorflow_hub as hub

def jaccard_similarity(x, y):
  """ returns the jaccard similarity between two lists """
  intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
  union_cardinality = len(set.union(*[set(x), set(y)]))
  return intersection_cardinality/float(union_cardinality)

def squared_sum(x):
  """ return 3 rounded square rooted value """ 
  return round(sqrt(sum([a*a for a in x])),3)
 
def euclidean_distance(x, y):
  """ return euclidean distance between two lists """ 
  return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def distance_to_similarity(distance):
  return 1/exp(distance)

def cos_similarity(x,y):
  """ return cosine similarity between two lists """
 
  numerator = sum(a*b for a,b in zip(x,y))
  denominator = squared_sum(x)*squared_sum(y)
  return round(numerator/float(denominator),3)

def create_heatmap(similarity, title, cmap = "YlGnBu"):
  df = pd.DataFrame(similarity)
  df.columns = labels
  df.index = labels
  fig, ax = plt.subplots(figsize=(5, 5))
  sns.heatmap(df, cmap=cmap)
  title = 'Heatmap for text similarity using '+ title
  plt.title(title)
  plt.show()

def cv(textLines):
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(textLines)
  arr = X.toarray()
  similarity = cosine_similarity(arr)
  create_heatmap(similarity, 'Count Vec')
  return similarity

def tfidf(textLines):
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(textLines)
  arr = X.toarray()
  similarity = cosine_similarity(arr)
  create_heatmap(similarity, 'TF-IDF')
  return similarity

def roberta(sentences):
  model = SentenceTransformer('stsb-roberta-large')

  embeddings = model.encode(sentences, convert_to_tensor=True)

  similarity = []
  for i in range(len(sentences)):
    row = []
    for j in range(len(sentences)):
      row.append(util.pytorch_cos_sim(embeddings[i], embeddings[j]).item())
    similarity.append(row)     
  create_heatmap(similarity, 'ROBERTA')
  return similarity

def use(text):
  module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
  model = hub.load(module_url)
  embeddings = model(text)
  similarity = cosine_similarity(embeddings)
  create_heatmap(similarity, 'USE')
  return similarity

with open('text/modelAnswer.txt') as textFile:
    modelText = textFile.read()

with open('text/studentAnswer.txt') as textFile:
    answerText = textFile.read()

texts = [modelText, answerText]
nlp = en_core_web_sm.load()
embeddings = [nlp(text).vector for text in texts]

# Jaccard Similarity
jac_similarity = jaccard_similarity(texts[0], texts[1])

# Euclidean Distance
euc_distance = euclidean_distance(embeddings[0], embeddings[1])
euc_distance = distance_to_similarity(euc_distance) 

# Cosine similarity
c_similarity = cos_similarity(embeddings[0], embeddings[1])

labels = ['Model Answer', 'Student Answer']
textLines = [modelText, answerText]

# Count Vectorizer
cv_similarity = cv(textLines)

# TD-IDF Vectorizer
tfidf_similarity = tfidf(textLines)

# RoBERTa based models implemented in the sentence-transformer
roberta_similarity = roberta(textLines)

# Universal Sentence Encoder
use_similarity = use(textLines)

print('Jaccard Similarity: ', jac_similarity)
print('Euclidean Distance: ', euc_distance)
print('Cosine Similarity: ', c_similarity)
print('CV Similarity: ', cv_similarity[0][1])
print('TFIDF Similarity: ', tfidf_similarity[0][1])
print('ROBERTA Similarity: ', roberta_similarity[0][1])
print('USE Similarity: ', use_similarity[0][1])
