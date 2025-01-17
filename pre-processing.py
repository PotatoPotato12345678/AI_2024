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
    #Removing stopwards by nltk
    stop_words = set(stopwords.words('english')) # list the stopwards
    word_tokens = text
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words: # if the word is not a stop words.
            filtered_sentence.append(w)

    return filtered_sentence

def convert_to_present(text):
    lemmatizer = WordNetLemmatizer() # set up Lemmatizer
    words = word_tokenize(text)  # Tokenize the text into words
    pos_tags = pos_tag(words)    # get part-of-speech of words
    
    present_text = []
    for word, tag in pos_tags:
        if tag.startswith('VB'):  # Check if the word is a verb
            present_text.append(lemmatizer.lemmatize(word, 'v'))  # Convert to presense form (base form)
        else:
            present_text.append(word)

    return ' '.join(present_text)

def step6_8(text):
    words = text.split() # data is splitted into words
    returnText = list()
    for i in range(len(words)):
        #step6: Replacing words with number expression with ˜
        if words[i].isdigit(): # check if it's number
            words[i] = "˜"
        #step7: Removing words with special characters: ! " # $ % & ' ( ) * + ,. / : ; < = > ? @ [ \ ] ^ _ ` { | } ~
        special_words = '!\"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
        step7 = re.sub(f"[{re.escape(special_words)}]", "", words[i]) 
        #step8: Replacing "-" with a single space.
        step8 = step7.replace("-", " ").split()
        returnText+= step8
    return returnText


# read all the data
f_ast = open('./original/astronomy.txt', 'r').read()
f_psy1 = open('./original/psychology.txt', 'r').read()
f_psy2 = open('./original/psychology_emotions.txt', 'r').read()
f_scio = open('./original/sociology.txt', 'r').read()
f_scio_add = open('./original/sociology_added.txt', 'r').read()

data = [f_ast,f_psy1,f_psy2,f_scio, f_scio_add]

#specify the output file
outputFile = ["astronomy.txt", "psychology_emotions.txt", "psychology.txt", "sociology.txt", "sociology_added.txt"]


for d in data:
    single_abstract = d.split("\n\n") # take each abstract. now the original one is separated by a blank line.
    for single_data in single_abstract:
        step1 = re.sub(r'https?://\S+|www\.\S+', '', single_data) # remove URLs
        step2 = convert_to_present(Word(single_data).pluralize()) # Converting all Verbs to presense and singular words to the plural forms by nltk and textblob
        step3 = step2.replace('º', '').replace('˜','') # Removing special characters: º and ˜
        step4 = step3.lower() # Lowercasing
        step5 = re.sub(r'\$.*?\$', 'º', step4) # Replacing words with Latex expression with a special word: º
        step8 = step6_8(step5)
        step9 = step9_p(step8)

        path = "./processed/"+outputFile[data.index(d)]
        with open(path, "a", encoding="utf-8") as f:
                f.write(" ".join(step9) + "\n\n")