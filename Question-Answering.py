#import logging
#logging.basicConfig(level=logging.INFO)
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import Data_Cleaning
from tika import parser
import json
import PyPDF2

string = ''
question = ''
text = ''
def read_text_data_to_string(file):

    if file.endswith(".txt"):
        Data_Cleaning.cleanData(file)
        f = open(file, "r")
        global string
        string = f.read()
        f.close()

    # since it is only a small pdf-file, we can use PyPDF2
    elif file.endswith(".pdf"):
        dict = parser.from_file(file)
        #print(dict['content'])
        #print(dict)
        dict = {'dict': dict}
        with open('pdf.txt', 'w+') as textfile:
            textfile.write(json.dumps(dict) + "\n\n")
            textfile.close()
        Data_Cleaning.cleanData("pdf.txt")
        f = open("pdf.txt", "r")
        string = f.read()
        f.close()


    #string = string.replace('\n', '')
    #print(string)

def question_answering(questionString):

    global question

    question = questionString

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    question, text = question, string


    input_ids = tokenizer.encode(question, text)
    print(input_ids)
    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
    print(answer)
    #assert answer == "going to a restaurant"

if __name__ == "__main__":
    textfile = input("Please enter a text- or pdf-file to be read by the system \n")
    print(f'you entered {textfile}')
    read_text_data_to_string(textfile)

    question = input("What is your question?")
    print(f'the answer to your question is ' + "\n")
    question_answering(question)