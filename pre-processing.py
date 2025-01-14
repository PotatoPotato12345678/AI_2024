import nltk
from textblob import Word
import re
import itertools
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def step9_p(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = text
    filtered_sentence = [w for w in word_tokens if not w.lower() in set(stopwords.words('english'))]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    return filtered_sentence

def convert_to_present(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)  # Tokenize the text into words
    pos_tags = pos_tag(words)    # Get part-of-speech tags
    
    present_text = []
    for word, tag in pos_tags:
        if tag.startswith('VB'):  # Check if the word is a verb
            present_text.append(lemmatizer.lemmatize(word, 'v'))  # Convert to base form
        else:
            present_text.append(word)  # Keep the word as is

    return ' '.join(present_text)

def step6_8(text):
    words = text.split()
    returnText = list()
    for i in range(len(words)):
        #step6
        if words[i].isdigit():
            words[i] = "˜"
        #step7
        special_words = '!\"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
        step7 = re.sub(f"[{re.escape(special_words)}]", "", words[i]) 
        #step8
        step8 = step7.replace("-", " ").split()
        returnText+= step8
    return returnText


f_ast = open('./original/astronomy.txt', 'r').read()
f_psy1 = open('./original/psychology.txt', 'r').read()
f_psy2 = open('./original/psychology_emotions.txt', 'r').read()
f_scio = open('./original/sociology.txt', 'r').read()
f_scio_add = open('./original/sociology_added.txt', 'r').read()

data = [f_ast,f_psy1,f_psy2,f_scio]
outputFile = ["astronomy.txt", "psychology_emotions.txt", "psychology.txt", "sociology.txt"]

data = [f_scio_add]
outputFile = ["sociology_added.txt"]

for d in data:
    single_abstract = d.split("\n\n")
    for single_data in single_abstract:
        step1 = re.sub(r'https?://\S+|www\.\S+', '', single_data)
        step2 = convert_to_present(Word(single_data).pluralize())
        step3 = step2.replace('º', '').replace('˜','')
        step4 = step3.lower()
        step5 = re.sub(r'\$.*?\$', 'º', step4)
        step8 = step6_8(step5)
        step9 = step9_p(step8)

        path = "./processed/"+outputFile[data.index(d)]
        with open(path, "a", encoding="utf-8") as f:
                f.write(" ".join(step9) + "\n\n")


