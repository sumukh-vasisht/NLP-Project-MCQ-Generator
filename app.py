from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import pandas as pd
import os

import nltk
# nltk.download('stopwords')
# nltk.download('popular')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from summarizer import Summarizer
import pprint
import itertools
import re
import pke
import string
from flashtext import KeywordProcessor
import requests
import json
import random
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import simple_lesk
from pywsd.lesk import cosine_lesk
from nltk.corpus import wordnet as wn

from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.jinja_env.filters['zip'] = zip

class MCQGenerator:

    def BERTSummarizer(text):
        model = Summarizer()
        result = model(text, min_length=60, max_length = 500 , ratio = 0.4)

        summary = ''.join(result)
        return summary

    def customTextSummarizer(text):
        stop = set(stopwords.words("english"))
        words = word_tokenize(text)

        freqTable = dict()
        for word in words:
            word = word.lower()
            if word in stop:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1
            
        sentences = sent_tokenize(text)
        sentenceValue = dict()

        for sentence in sentences:
            for word, freq in freqTable.items():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq

        sumValues = 0
        for sentence in sentenceValue:
            sumValues += sentenceValue[sentence]

        average = int(sumValues / len(sentenceValue))

        summary = ''

        for sentence in sentences:
            if(sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
                summary += " " + sentence

        return summary

    def getNounsMultipartite(text):
        out=[]
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=text)
        pos = {'PROPN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)
        extractor.candidate_weighting(alpha=1.1,
                                    threshold=0.75,
                                    method='average')
        keyphrases = extractor.get_n_best(n=20)
        for key in keyphrases:
            out.append(key[0])
        return out

    def tokenizeSentences(text):
        sentences = [sent_tokenize(text)]
        sentences = [y for x in sentences for y in x]
        
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences

    def getSentencesForKeyword(keywords, sentences):
        keyword_processor = KeywordProcessor()
        keyword_sentences = {}
        for word in keywords:
            keyword_sentences[word] = []
            keyword_processor.add_keyword(word)
        for sentence in sentences:
            keywords_found = keyword_processor.extract_keywords(sentence)
            for key in keywords_found:
                keyword_sentences[key].append(sentence)
        for key in keyword_sentences.keys():
            values = keyword_sentences[key]
            values = sorted(values, key=len, reverse=True)
            keyword_sentences[key] = values
        return keyword_sentences

    def getDistractorsWordnet(syn, word):
        distractors=[]
        word= word.lower()
        orig_word = word
        if len(word.split())>0:
            word = word.replace(" ","_")
        hypernym = syn.hypernyms()
        if len(hypernym) == 0: 
            return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name()
            #print ("name ",name, " word",orig_word)
            if name == orig_word:
                continue
            name = name.replace("_"," ")
            name = " ".join(w.capitalize() for w in name.split())
            if name is not None and name not in distractors:
                distractors.append(name)
        return distractors

    def getWordsense(sent, word):
        word= word.lower()
        
        if len(word.split())>0:
            word = word.replace(" ","_")
        
        
        synsets = wn.synsets(word,'n')
        if synsets:
            wup = max_similarity(sent, word, 'wup', pos='n')
            adapted_lesk_output =  adapted_lesk(sent, word, pos='n')
            lowest_index = min (synsets.index(wup),synsets.index(adapted_lesk_output))
            return synsets[lowest_index]
        else:
            return None

    # Distractors from http://conceptnet.io/
    def getDistractorsConceptnet(word):
        word = word.lower()
        original_word= word
        if (len(word.split())>0):
            word = word.replace(" ","_")
        distractor_list = [] 
        url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word)
        obj = requests.get(url).json()

        for edge in obj['edges']:
            link = edge['end']['term'] 

            url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
            obj2 = requests.get(url2).json()
            for edge in obj2['edges']:
                word2 = edge['start']['label']
                if word2 not in distractor_list and original_word.lower() not in word2.lower():
                    distractor_list.append(word2)
                    
        return distractor_list

def roberta(sentences):
  model = SentenceTransformer('stsb-roberta-large')

  embeddings = model.encode(sentences, convert_to_tensor=True)

  similarity = []
  for i in range(len(sentences)):
    row = []
    for j in range(len(sentences)):
      row.append(util.pytorch_cos_sim(embeddings[i], embeddings[j]).item())
    similarity.append(row)     
  return similarity

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/mcqGenerator', methods = ['GET', 'POST'])
def mcqGenerator():
    if request.method=="POST":
        text = request.form['text']

        BERTSummary = MCQGenerator.BERTSummarizer(text)
        keywords = MCQGenerator.getNounsMultipartite(text)

        filteredKeys=[]
        for keyword in keywords:
            if keyword.lower() in BERTSummary.lower():
                filteredKeys.append(keyword)

        sentences = MCQGenerator.tokenizeSentences(BERTSummary)
        keywordSentenceMapping = MCQGenerator.getSentencesForKeyword(filteredKeys, sentences)

        keyDistractorList = {}

        for keyword in keywordSentenceMapping:
            wordsense = MCQGenerator.getWordsense(keywordSentenceMapping[keyword][0],keyword)
            if wordsense:
                distractors = MCQGenerator.getDistractorsWordnet(wordsense,keyword)
                if len(distractors) ==0:
                    distractors = MCQGenerator.getDistractorsConceptnet(keyword)
                if len(distractors) != 0:
                    keyDistractorList[keyword] = distractors
            else:        
                distractors = MCQGenerator.getDistractorsConceptnet(keyword)
                if len(distractors) != 0:
                    keyDistractorList[keyword] = distractors

        numberOfOptions = 4

        index = 1
        indexes = []
        questions = []
        choicesFinal = []
        correctOptions = []
        for each in keyDistractorList:
            sentence = keywordSentenceMapping[each][0]
            pattern = re.compile(each, re.IGNORECASE)
            question = pattern.sub( " _______ ", sentence)
            indexes.append(index)
            questions.append(question)
            # print ("%s)"%(index),question)
            choices = [each.capitalize()] + keyDistractorList[each]
            top4choices = choices[:numberOfOptions]
            random.shuffle(top4choices)
            optionchoices = ['a','b','c','d']
            choicesFinal.append(top4choices)
            # for idx,choice in enumerate(top4choices):
            #     print ("\t",optionchoices[idx],")"," ",choice)
            # print ("\nMore options: ", choices[4:20],"\n\n")
            index = index + 1
            correctOption = each
            correctOptions.append(correctOption)
        
        return render_template('mcqs.html', inputText = text, indexes = indexes, questions = questions, options = choicesFinal, correctOptions = correctOptions, optionChoices = optionchoices)
    return render_template('mcq.html')

@app.route('/textSimilarity', methods = ['GET', 'POST'])
def textSimilarity():
    if request.method=="POST":
        modelText = request.form['modelText']
        answerText = request.form['answerText']
        textLines = [modelText, answerText]
        roberta_similarity = roberta(textLines)
        score = roberta_similarity[0][1]*100
        return render_template('similarity.html', modelText = modelText, answerText = answerText, score = score)
    return render_template('similarity.html', modelText = '', answerText = '')
		
if __name__ == '__main__':
   app.run(debug = True)