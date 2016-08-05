import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets.
    """
    string = ''.join([i if ord(i)>=32 and ord(i) < 127 else ' ' for i in string])
    string = re.sub(r"'s", " 's", string)
    string = re.sub(r"'ve", " 've", string)
    string = re.sub(r"n't", " n't", string)
    string = re.sub(r"'re", " 're", string)
    string = re.sub(r"'d", " 'd", string)
    string = re.sub(r"'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"([\.=#@\+\?/]){2,}"," ",string)
    string = re.sub(r"(([\.=#@\+\?/])\s*){2,}\s"," ",string)
    # remove all the words that more than 20 characters (like website, some numbers combination, etc)
    string = re.sub(r"(OBJ_NUM[ ]+){2,}","OBJ_NUM ",string)
    string = re.sub(r"\s[\w]{30,}\s"," ",string)
    # only keep one obj_num for continuously words
    string = re.sub(r"\s(\d){5,}\s", " ",string)
    # remove all the numbers that more than 4 digits
    string = re.sub(r"\s((\d)+([^\s\w])+(\d)+)+\s"," ",string)
    # remove all the combination of numbers and special character
    string = re.sub(r"\s{2,}", " ", string)
    # remove all the space
    # remove special characters except: ():_@/\.\'
    return string

