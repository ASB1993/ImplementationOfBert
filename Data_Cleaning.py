
import inflect
import nltk
import csv

import re
import unicodedata
from bs4 import BeautifulSoup
import cleantext
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def cleanData(textFile):
    # before tokenizing, clean data properly

    with open(textFile, 'r', encoding='utf-8') as file:
        text = file.read()
        file.close()


    # strip html, remove between square brackets and denoise text
    text = denoise_text(text)
    #print(text)

    """would be good if we could solve the problem of installing all packages, because it would be a really nice code as it searches for all don't, couldn't... and 
    replaces it with do not, could not..."""
    #resTex = replace_contractions(text)

    # dealing with HTML
    #''.join(xml.etree.ElementTree.fromstring(resTex).iteratetext())

    # delete unwanted characters
    ''.join(e for e in text if e.isalnum())
    # print(text)

    """remove passage manually where text looks weird
    new_text = re.sub('[^a-zA-Z0-9\n\.]', ' ', text)
    middle = new_text.find("96 Dembsey")
    middleEnd = new_text.find("About the Author")
    start = new_text.find("International Journal of Science  Technology and Society")
    end = new_text.find("Venkatesh  V.    Davis  F. D.  2000 . A")
    textTwo = new_text
    if start != -1 and middle != -1:
        resTex = new_text[start:middle]

    if middleEnd != -1 and end != -1:
        resTex = resTex + new_text[middleEnd:end]"""

    """remove blank lines"""
    #re.sub(r'\n\s*\n', '', text)

    """use of cleantext library"""
    cleantext.clean(text)

    """fix_unicode=True,  # fix various unicode errors
    to_ascii=True,  # transliterate to closest ASCII representation
    lower=True,  # lowercase text
    no_line_breaks=False,  # fully strip line breaks as opposed to only normalizing them
    no_urls=True,  # replace all URLs with a special token
    no_emails=True,  # replace all email addresses with a special token
    no_phone_numbers=True,  # replace all phone numbers with a special token
    no_numbers=False,  # replace all numbers with a special token
    no_digits=False,  # replace all digits with a special token
    no_currency_symbols=False,  # replace all currency symbols with a special token
    no_punct=False,  # fully remove punctuation
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"  # set to 'de' for German special handling"""


    """ convert all letters to lower case """
    #resTex = resTex.lower()

    """ removing punctuations, accent marks and other diacritics"""
    # table = str.maketrans("","")
    #resTex = resTex.translate(str.maketrans('', '', string.punctuation))
    f = open("test2.txt", "w", encoding='utf-8')
    f.write(text)
    f.close()


    #read to csv
    #csv_Maker(textFile, resTex)

    """tokenize words"""
    #words = nltk.word_tokenize(resTex)

    """normalization: remove non-ascii words, all words to lowercase, remove punctuation,
        replace numbers, remove stopwords"""
    #words = normalize(words)

    #stems, lemmas = stem_and_lemmatize(words)


    # Named-entity recognition
    #ne_chunk(pos_tag(resTex))

    """ removing white space"""
    #resTex = resTex.strip()
    # print(resTex)

    """ remove stop words"""
    #stop_words = set(stopwords.words('english'))
    #tokens = word_tokenize(resTex)
    #resTex = [i for i in tokens if not i in stop_words]
    # print(resTex)

    # spelling correction
    #import train
    #train['tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))

    # common word removal
    # freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]
    # print("freq: ", freq)

    """ stemming """
    #stemmer = PorterStemmer()
    # for word in resTex:
        #stemmer.stem(word)

    # POS tagging
    # result = TextBlob(resTex)
    # print(result.tags)

    """lemmatization"""
    #lemmatizer = WordNetLemmatizer()
    #for word in resTex:
     #   lemmatizer.lemmatize(word)



    # write to new file
    #p = open("words.txt", "w+", encoding='utf-8')
    #p.write(words)
    #p.close()
    #print(words)
    #f.write(words)
    #f.close()

    # collocation extraction using ICE
    # extractor = CollocationExtractor.with_collocation_pipeline("T1", bing_key = "Temp", pos_check = False)
    # print(extractor.get_collocations_of_length(result, length = 3))



    # remove sparse terms and particular words

    # tc.main_cleaner('b.txt')  # another tool for cleaning

    # split into sentences
    # sentences = sent_tokenize(text)
    # print(sentences)


# file to csv
def csv_Maker(filename, text):
    #with open('testcsv1.csv', mode='w') as file:
     #   writer = csv.writer(file, delimiter=',')

        # way to write to csv file
      #  writer.writerow(['Title', 'Content'])
       # writer.writerow([filename, text])

    with open('outputdata.csv', 'w') as outfile:
        mywriter = csv.writer(outfile)
        # manually add header

        mywriter.writerow(['Filename', 'Content'])
        mywriter.writerow([filename, text])



# deep learning to tie contents together to make sure it relates to contents/questions

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

"""def replace_contractions(text):
    Replace contractions in string of text
    return contractions.fix(text)"""

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctutation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all integer occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = nltk.LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verby in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctutation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas

