
"""
 * Course: SSC300E Erasmus Final Year Project
 * Assignment: Creating a Question-Answering System that reads ".txt"-files or ".pdf"-files asks for a question provided
   by the user and answers the question based on the file that was given by the user
 * Author: Anna-Sophie Bartle
 * Description: Loads ands saves instances of class Address.java to files
 * <p>
 * Honor Code:  I pledge that this program represents my own work. The code for the model itself (def question_answering(questionString):)
   partly comes from the Transformer's website who provided the fine-tuned version on question-answering of BERT.
 * I received help from: Daniel Onah, PhD and Huggingface/Transformers (https://github.com/huggingface/transformers/pull/1502/files)
 * in designing and debugging my program.
"""


#import logging
#logging.basicConfig(level=logging.INFO)
import sys
import os
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import Data_Cleaning
import cleantext
import PyPDF2
import textract
import matplotlib.pyplot as plt
import seaborn as sns

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

    # initialize tokenizer -- for BERT, only the BERT-tokenizer is working
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # initialize the fine-tuned model
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # text receives global variable string, question is the questionString that is the question asked by the user
    question, text = question, string

    # question and text are encoded to numbers by the tokenizer
    input_ids = tokenizer.encode(question, text)
    print("input ids", input_ids)
    print('The input has a total of {:} tokens.'.format(len(input_ids)))
    # all tokens  that belong to the question receive a 0, all tokens that belong to the paragraph receives a 1
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))] # segment embeddings
    print("token type ids", token_type_ids)
    # start_scores and end_scores represent the questions and the context paragraph that are now run through the model
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))


    # all ids are converted back into token strings, but how are the words
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # For each token and its id...
    for token, id in zip(all_tokens, input_ids):

        # If this is the [SEP] token, add some space around it to make it stand out.
        if id == tokenizer.sep_token_id:
            print('')

        # Print the token string and its ID in two columns.
        print('{:<12} {:>6,}'.format(token, id))

        if id == tokenizer.sep_token_id:
            print('')

    # the start_scores and end_scores which would give the highest maximum are taken as an answer and since,
    # in the output-layer, all tokens are shifted to the right by one, +1 is needed, since we don't want this token,
    # but the one beside it. tokens are then put together and displayed as an answer
    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

    #assert answer == "going to a restaurant"


    # just for demonstration purposes
    # we read the calculations of the highest probabilities of the start and end scores in separate variables to save it
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    print("answer start", answer_start)
    print("answer end", answer_end)
    # Start with the first token.
    answer = all_tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if all_tokens[i][0:2] == '##':
            answer += all_tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + all_tokens[i]

    print(answer)


    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    # sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (16, 8)
    # Pull the scores out of PyTorch Tensors and convert them to 1D numpy arrays.
    s_scores = start_scores.detach().numpy().flatten()
    e_scores = end_scores.detach().numpy().flatten()

    # We'll use the tokens as the x-axis labels. In order to do that, they all need
    # to be unique, so we'll add the token index to the end of each one.
    token_labels = []
    for (i, token) in enumerate(all_tokens):
        token_labels.append('{:} - {:>2}'.format(token, i))
    # Create a barplot showing the start word score for all of the tokens.
    ax = sns.barplot(x=token_labels, y=s_scores, ci=None)

    # Turn the xlabels vertical.
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

    # Turn on the vertical grid to help align words to scores.
    ax.grid(True)

    plt.title('Start Word Scores')

    plt.show()

    # Create a barplot showing the end word score for all of the tokens.
    ax = sns.barplot(x=token_labels, y=e_scores, ci=None)

    # Turn the xlabels vertical.
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

    # Turn on the vertical grid to help align words to scores.
    ax.grid(True)

    plt.title('End Word Scores')

    plt.show()

if __name__ == "__main__":

    import time # to time the processing time of the model to answer the question

    # user is asked to enter text-file which is cleaned and read to a global variable
    textfile = input("Please enter a text- or pdf-file to be read by the system \n")
    print(f'you entered {textfile}') # confirms, if it is wrong, the user can re-start the program
    # calls function to read file
    read_text_data_to_string(textfile)

    # asks user for a content-specific question
    question = input("What is your question based on the content of the file?")
    print(f'the answer to your question is ' + "\n")
    #stops time
    t0 = time.time()
    # question is passed to question_answering function where it is processed and answered by the model
    question_answering(question)
    t1 = time.time()
    # total time it has taken for the program to answer the question
    total = t1 - t0
    print(total)

