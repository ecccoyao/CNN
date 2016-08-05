import re
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets.
    """
    print "before :{:d}".format(len(string))
    string = ''.join([i if ord(i)>=32 and ord(i) < 127 else ' ' for i in string])
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"([\.=#]){2,}"," ",string)
    string = re.sub(r"(([\.=#])\s?){2,}\s"," ",string)
    # remove all the words that more than 20 characters (like website, some numbers combination, etc)
    string = re.sub(r"(OBJ_NUM[ ]+){2,}","OBJ_NUM ",string)
    # only keep one obj_num for continuously words
    string = re.sub(r"\s(\d){5,}\s", " ",string)
    # remove all the numbers that more than 4 digits
    string = re.sub(r"\s((\d)+([^\s\w])+(\d)+)+\s"," ",string)
    # remove all the combination of numbers and special character
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s[\w]{18,}\s"," ",string)
    # remove all the space
    # remove special characters except: ():_@/\.\'
    print "after :{:d}".format(len(string))
    return string


def write_to_file():
    lists = []
    with open("sample.txt","r") as infile, open("sample2","w") as outfile:
        for line in infile:
            try:
                line = clean_str(line)
                outfile.write(line + "\n")
                total = len(line.split(" "))
                print "total words is : {:d}".format(total)
                lists.append(str(total))
            except:
                print "no result"
    infile.close()
    outfile.close()
    return lists


def plot_text(lists):
    h = sorted(lists)  # sorted
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  # this is a fitting indeed
    plt.plot(h, fit, '-.')
    plt.hist(h, normed=True)  # use this to draw histogram of your data
    plt.show()

def  main():
    write_to_file()
    """
    with open("stats.txt","w") as file:
        file.write("\n".join(lists))
    plot_text(lists)
    """

if __name__ == '__main__':
    main()
