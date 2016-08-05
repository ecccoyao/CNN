from itertools import izip
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import csv
import operator

def getUnion(list1,list2,top,list1_top1_weight,list1_all_weight,list2_top1_weight,list2_all_weight):
    """
    get the union of two list, multiply by their probablity of being correct
    :param list1: RF list result
    :param list2: CNN list result
    :param list1_top1_weight: RF weight for top1
    :param list1_all_weight: RF weight for all other than top1
    :param list2_top1_weight: CNN weight for top1
    :param list2_all_weight: CNN weight for all other than top1
    :return: list of sorted top10 engineer
    """
    global unions
    unions.append(getUnionLength([item[1] for item in list1],[item[1] for item in list2]))
    eng_probs = defaultdict(lambda: 0)
    for i in range(len(list1)):
        prob1,engID1 = list1[i]
        prob2,engID2 = list2[i]
        if i ==0 :
            eng_probs[engID1] += prob1 * list1_top1_weight
            eng_probs[engID2] += prob2 * list2_top1_weight
        else:
            eng_probs[engID1] += prob1 * list1_all_weight
            eng_probs[engID2] += prob2 * list2_all_weight

    top10_list = sorted(eng_probs, key=eng_probs.get, reverse = True)[:top]
    return top10_list



def getUnionLength(list1,list2):
    """
    get the union length for one list
    :param list1: RF
    :param list2: CNN
    :return: the int length
    """
    return len(list(set(list1) | set(list2)))


def plot(RF_file, CNN_file, top,list1_top1_weight,list1_all_weight,list2_top1_weight,list2_all_weight):
    """
    plot the two list performance
    :param file1: csv file
    :param file2: csv file
    :param top: how many intersection or union
    :return: None
    """
    global eng_rank_list
    eng_rank_list = []
    accuracy1, accuracy5, accuracy10,accuracy20,total = 0,0,0,0,0
    with open(RF_file, 'r') as csvfile1,open(CNN_file,'r') as csvfile2:
        # get number of columns
        for line1,line2 in izip(csvfile1,csvfile2):
            total += 1
            rf_result,cnn_result = line1.split(','),line2.split(',')
            engID = int(cnn_result[1])
            prediction_list = getUnion(getTop(top,rf_result[2:]),getTop(top,cnn_result[2:]),\
                                       top,list1_top1_weight,list1_all_weight,list2_top1_weight,list2_all_weight)
            eng_rank_list.append(rf_result[:2]+prediction_list)
            if engID == prediction_list[0]:
                accuracy1 += 1
            if engID in prediction_list[:5]:
                accuracy5 += 1
            if engID in prediction_list[:10]:
                accuracy10 += 1
            if engID in prediction_list[:20]:
                accuracy20 += 1
    return [list1_top1_weight, list1_all_weight,float(accuracy1)/total, float(accuracy5)/total,float(accuracy10)/total, float(accuracy20)/total]

def df(lists):
    """
    plot the density plot
    :param lists: list of rank for right engineers
    :return: None
    """
    h = sorted(lists)  # sorted
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  # this is a fitting indeed
    plt.plot(h, fit, '-.')
    plt.hist(h, bins = 40,normed=True)  # use this to draw histogram of your data
    plt.show()


def cdf(lists,top):
    """
    plot the cumulative density plot
    :param lists: list of rank for right engineers
    :return: None
    """
    h = sorted(lists)
    n_bins = 20
    plt.hist(h, n_bins, normed=1, histtype='step', cumulative=True)
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.xlim(0, top )
    plt.title('cumulative step')
    plt.show()

def getTop(top,prob_list):
    """
    given one list of probability, return the list of tuples as (probability, engineer ID)
    :param prob_list: 1*871 list
    :return: 1 * 10 list of (probability, engID)
    """
    prob_list = map(float,prob_list)
    ranks = (-np.array(prob_list)).argsort()
    top_list = [(prob_list[index], index) for index in ranks[:top]]
    return top_list


def writeToCSV(top,file1,file2,file3,file4):
    """
    write ensemble result to csv
    :param file1: RF_PD
    :param file2: CNN_PD
    :param file3: RF_ALL
    :param file4: CNN_ALL
    :return: None
    """
    with open(str(top)+"_result.csv", "w") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["result for pd data"])
        spamwriter.writerow(["rf top1 weight", "rf all weight", "top1", "top5", "top10","top20"])
        spamwriter.writerow(plot(file1, file2, top,1, 1, 0, 0))
        spamwriter.writerow(plot(file1, file2, top,0, 0, 1, 1))
        for i in range(14, 20):
            for ii in range(5, i + 1):
                spamwriter.writerow(plot(file1, file2, top,i, ii, 1, 1))
        spamwriter.writerow([])
        spamwriter.writerow(["result for all case data"])
        spamwriter.writerow(["rf top1 weight", "rf all weight", "top1", "top5", "top10","top20"])
        spamwriter.writerow(plot(file3, file4,  top,1, 1, 0, 0))
        spamwriter.writerow(plot(file3, file4, top,0, 0, 1, 1))
        for i in range(14, 20):
            for ii in range(5, i + 1):
                spamwriter.writerow(plot(file3, file4,top, i, ii, 1, 1))

unions = []
eng_rank_list = []

def writeProbToCSV(top,file_name):
    with open(str(top)+"eng_result_"+file_name +"_udpate.csv","w") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        global eng_rank_list
        for eng_list in eng_rank_list:
            spamwriter.writerow(eng_list)

def filterCSV():
    with open("RF_case_prob_pd_filter.csv","r") as infile, open("RF_case_prob_pd_filter2.csv","w") as outfile:
        for case_line in infile.readlines():
            parts = case_line.split(',')
            probs = np.array(map(float, parts[2:]), dtype=np.float_)

            # print type(probs), probs.dtype
            # print probs.shape
            # print type(probs[probs == 0]), probs[probs == 0].dtype
            # print probs[probs == 0].shape
            if len(probs[probs == 0]) < len(probs): # probs is a 0 vector --> no problem description
                outfile.write(case_line)



def main():
    """
    main function to call
    :return: None
    """
    file1 = "RF_case_prob_pd_filter2.csv"
    file2 = "CNN_case_prob_pd_6.csv"
    file3 = "RF_case_prob_all_filter.csv"
    file4 = "CNN_case_prob_all_5.csv"
    top = 20
    print plot(file1,file2,top,14,7,0,0)
    writeProbToCSV(top,"pd")
    print plot(file3,file4,top,14,5,1,1)
    writeProbToCSV(top,"all")
    global unions
    # writeToCSV(top,file1,file2,file3,file4)
    print float(sum(unions))/len(unions)


if __name__ == '__main__':
    main()
