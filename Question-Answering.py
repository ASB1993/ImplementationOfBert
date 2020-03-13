import logging
logging.basicConfig(level=logging.INFO)
import sys
import os
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import Data_Cleaning
import cleantext
import PyPDF2
import textract

string = ''
question = ''
text = ''
def read_text_data_to_string(file):
    if not file.endswith(".txt") or not file.endswith(".pdf"):
        if os.path.exists(file):
            if file.endswith(".txt"):
                Data_Cleaning.cleanData(file)
                try:
                    f = open(file, "r", encoding='utf-8')
                    global string
                    string = f.read()
                    f.close()
                except OSError:
                    print("could not open/read file ", file)
                    sys.exit()
                string = string.replace('\n', ' ').replace('\r', '')

            # since it is only a small pdf-file, we can use PyPDF2
            elif file.endswith(".pdf"):
                try:
                    #open pdf-file
                    f = open(file, "rb")
                    # create pdf-reader with library PyPDF2
                    pdfReader = PyPDF2.PdfFileReader(f)

                    #number of pages - important for PyPDF2
                    totalPageNumber = pdfReader.numPages

                    currentPage = 0
                    text = ''

                    #loop through all pages to get text
                    while(currentPage < totalPageNumber):
                        pdfPage = pdfReader.getPage(currentPage)

                        # get text
                        text = text + pdfPage.extractText()

                        currentPage += 1
                    if(text == ''):
                        # if still not readable, use textract
                        text = textract.process(file, method='tesseract', encoding= 'utf-8')
                    f.close()
                except OSError:
                    print("could not open/read file ", file)
                    sys.exit()

                # open or create new textfile, since Data_Cleaning.py only reads files.
                try:
                    f = open('test2.txt', 'w+')
                    f.write(text)
                    f.close()
                except OSError:
                    print("Sorry, something went wrong.")
                    sys.exit()

                #clean text and read cleaned text to global string-variable
                Data_Cleaning.cleanData("test2.txt")
                try:
                    f = open('test2.txt', "r")
                    string = f.read()
                    # see, whether reading from file was successful
                    print(string)
                    f.close()
                except OSError:
                    print("OS Exception.")
                    sys.exit()
                string = string.replace('\n', ' ').replace('\r', '')
        else:
            print("File does not exist. Please enter a file that is of type \".txt\" or \".pdf\"")
            sys.exit()
    else:
        print("File must be of type \".txt\" or \".pdf\", please try again")
        sys.exit()

def question_answering(questionString):

    global question

    question = questionString

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    question, text = question, string

    input_ids = tokenizer.encode(question, text)
    #print(input_ids)
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    #print(token_type_ids)
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    #print(start_scores)
    #print(end_scores)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
    print(answer)
    #assert answer == "going to a restaurant"

if __name__ == "__main__":

    import time
    textfile = input("Please enter a text- or pdf-file to be read by the system \n")
    print(f'you entered {textfile}')
    read_text_data_to_string(textfile)

    question = input("What is your question?")
    print(f'the answer to your question is ' + "\n")
    t0 = time.time()
    question_answering(question)
    t1 = time.time()
    total = t1 - t0
    print(total)