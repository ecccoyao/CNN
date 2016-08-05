# Ajay Jain
# July 30, 2016

import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

def cdf(lists,top,title):
    """
    plot the cumulative density plot
    :param lists: list of rank for right engineers
    :return: None
    """
    for model in lists:
        h = sorted(model)
        n_bins = 871
        plt.hist(h, n_bins, normed=1, histtype='step', cumulative=True)
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.xlim(0, top)
    plt.title('cumulative result for ' + title)
    plt.show()


def calculate_accuracy(data_type,csv_filepath):
    num_cases = 0
    num_match = 0
    num_match5 = 0
    num_match10 = 0
    num_match20 = 0
    prediction_ranks = []

    with open(csv_filepath, 'r') as f:
        for case_line in f.readlines():
            parts = case_line.split(',')
            caseId = parts[0]
            engUUID = int(parts[1])
            if data_type == "ensemble":
                ranks = map(int, parts[2:])
            else:
                ranks = (-np.array(map(float,parts[2:]))).argsort()
            try:
                predict_rank = np.where(np.array(ranks) == engUUID)[0][0]
            except:
                predict_rank = 871


            num_cases = num_cases + 1
            if predict_rank == 0:
                num_match += 1
            if predict_rank < 5:
                num_match5 += 1
            if predict_rank < 10:
                num_match10 += 1
            if predict_rank < 20:
                num_match20 += 1

            prediction_ranks.append(predict_rank)

    print csv_filepath, 'Exact Matches :', num_match
    print csv_filepath, 'Cases   :', num_cases
    print csv_filepath, 'top1 Accuracy:', float(num_match) / num_cases
    print csv_filepath, 'top5 Accuracy:', float(num_match5) / num_cases
    print csv_filepath, 'top10 Accuracy:', float(num_match10) / num_cases
    print csv_filepath, 'top20 Accuracy:', float(num_match20) / num_cases
    return prediction_ranks


# calculate_accuracy('20eng_result_all.csv')
rank_CNN= calculate_accuracy("",'CNN_case_prob_all_6.csv')
rank_RF =calculate_accuracy("","RF_case_prob_all_filter.csv")
rank_ensemble = calculate_accuracy("ensemble",'20eng_result_all_udpate.csv')
cdf([rank_CNN,rank_RF,rank_ensemble],20,"CNN & RF & ensemble")
# with open("accuracy_pd_recent.txt","w") as pd_file, open("accuracy_all_recent.txt","w") as all_file:
#     pd_file.write(",".join(pd_predictions))
#     all_file.write(",".join(all_predictions))
