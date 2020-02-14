import os

from tika import parser # version 1.23
import json






def textExtraction_saveToFile(directory):
    # print out all paths to files that have pdf extension


    # files = open("files.txt").readlines()

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file = os.path.join(directory, filename)

            print(file)

            # parse data from file
            text = parser.from_file(file)
            print(text['content'])
            print(type(text))

            # print(text)

            text = {'text': text}
            with open('test2.txt', 'a+') as file:
                file.write(json.dumps(text) + "\n\n")
                file.close()

            # save to textfile
            #f = open("csvFile.csv", "a+", encoding='utf-8')
            #f.writerow({'file name ': file, 'text ': text})
            #f.close()
            # files.append(file)

            #myData = [[1, 2], [file, text]]
            #myFile = open('csvexample3.csv', 'w', encoding='utf-8')
            #with myFile:
             #   writer = csv.writer(myFile)
              #  writer.writerows(myData)

            #f = open("test2.txt", "a+", encoding='utf-8')
            #f.write(text)
            #f.close()


        else:
            continue


    # with open("files.txt", "a+") as f:
    # f.write("\n".join(files))



if __name__ == "__main__":
    import Data_Cleaning
    textExtraction_saveToFile('./texts')
    Data_Cleaning.cleanData('test2.txt')

