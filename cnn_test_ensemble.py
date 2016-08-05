# Ajay Jain
# July 20, 2016

__import__('sys').path.append('/var/log/mldata/Intelligent-Routing')
import matplotlib.pyplot as plt
import numpy as np

import csv
from data_helpers_cnn import load_obj
# from RFClassifier import RFClassifier
from CNN_classifier import CNNClassifier
# from FeatureClassifier import FeatureClassifier
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import post_filter

def strip(s):
    return s.strip()

def divideCase(input_case):
    parts = map(strip, input_case.split("#~$~#"))
    caseId = parts[0]
    label = parts[1]
    series = parts[2]
    platform = parts[3]

    notes = []

    for note_string in map(strip, parts[4].split("#$^$#")):
        note_array = note_string.split(' ')
        note_type = note_array[0]
        note = ' '.join(note_array[1:]).strip()

        notes.append((note_type, note))

    return caseId, label, series, platform, notes

def lower(notes):
    notes = notes.split(" ")
    notes = [note.lower() if note not in ("OBJ_IP_ADDRESS","OBJ_NUM") else note for note in notes]
    return " ".join(notes)


def process_case_string(case_string, pd_only=True, prepend_feat=False):
    caseId, label, series, platform, notes = divideCase(case_string)

    X_case_data = []
    for note_type, note in notes:
        if note_type == 'PD':
            note_type_weight = 1.
        elif not pd_only:
            if note_type == 'RN':
                note_type_weight = 1.5/2
            else: note_type_weight = 1. / 2
        else:
            note_type_weight = 0.

        if note_type_weight > 0:
            if prepend_feat:
                note = ' '.join([series, platform, note])

            note = post_filter.clean_str(note)
            note = lower(note)
            X_case_data.append((note_type_weight, [series, platform], note))
    return (caseId, label, X_case_data)

# (caseId, label, [(note_type, [series, platform], filtered_note_with_feat),
#                  (note_type, [series, platform], filtered_note_with_feat),
#                  (note_type, [series, platform], filtered_note_with_feat)])
# ensemble.predict_proba_case([(note_type, [series, platform], filtered_note_with_feat),
#                              (note_type, [series, platform], filtered_note_with_feat),
#                              (note_type, [series, platform], filtered_note_with_feat)]) --> [871 probs]

# @profile
def load_test_data(filepath='caseId_note_aggregation_test_data_with_features', pd_only=True, prepend_feat=False):
    cases = []

    with open(filepath, 'r') as f:
        # need to buffer lines because some cases are split across multiple lines
        for case_string in f.xreadlines():
            processed_case = process_case_string(case_string, pd_only=pd_only, prepend_feat=prepend_feat)
            # if len(processed_case[2]) > 0:
            cases.append(processed_case)
    return cases[-1250:]

def write_to_txt(dataset):
    global total_notes
    global notes_len
    # with open('pd_test_set','wb') as file:
    for case in dataset:
        caseID, label, case_notes = case
        if case_notes:
            whole_note = ""
            for note in case_notes:
                total_notes += 1
                notes_len.append(len(note[2].split(" ")))
                whole_note += note[2]
                whole_note += " "
                # file.write(caseID+" "+ label+" "+whole_note+"\n")



def calculate_medium(file):
    with open(file,"r") as file:
        length = []
        for data in file.readlines():
            length.append(len(data.split(" "))-1)
        length = sorted(length)
        print "medium is: ", length[len(length)/2]

def write_to_CSV(data,dataset):
    with open('CNN_case_prob_'+ data + '_6.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for case in dataset:
            caseID, label, case_notes = case
            if case_notes:
                print "label is",label
                probs = classify_case_cnn(case_notes,label)
                row = [caseID,label]
                spamwriter.writerow(row + list(probs))

def cdf(lists,title):
    """
    plot the cumulative density plot
    :param lists: list of rank for right engineers
    :return: None
    """

    h = sorted(lists)
    print h[len(h)/2]
    n_bins = 10000
    plt.hist(h, n_bins, normed=1, histtype='step', cumulative=True)
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.xlim(0,300)
    plt.title('cumulative result for ' + title)
    plt.show()



classifier = None

N_CLASS = 871

total_notes = 0
notes_len = []

def classify_case_cnn(case,label):
    if len(case) > 0:
        return classifier.predict_proba_case(case,label)

def main():
    calculate_medium("stats_for_data.txt")
    global classifier
    global total_notes
    global notes_len
    test_data_pd = load_test_data(pd_only=True, prepend_feat=False)
    write_to_txt(test_data_pd)
    print total_notes
    print  "the average length is:",float(sum(notes_len))/total_notes
    cdf(notes_len,"note length")

    test_data_all = load_test_data(pd_only=False, prepend_feat=False)
    print time.clock(), "start cnn testing"
    model = "dataset1_L_update5"
    pickle_file = "obj"
    cnn = CNNClassifier(pickle_file, model)
    cnn.load_model()
    print time.clock(), "the cnn loads model END"
    classifier = cnn

    # with ThreadPoolExecutor(1) as e:
    #     probs_cnn_pd = e.map(classify_case_cnn, [case[2] for case in test_data_pd])
    #     probs_cnn_pd_list = list(probs_cnn_pd)
    #d
    # with open("CNN_case_prob_pd.csv", 'w') as f:
    #     for i, prob_vec in enumerate(probs_cnn_pd_list):
    #         caseId, label, X = test_data_pd[i]
    #         csv_row = ','.join([caseId, label] + map(str, list(prob_vec))) + '\n'
    #         f.write(csv_row)
    # write_to_CSV("all",test_data_all)
    # print "the data training has ended"
    print time.clock(), "start all set"
    write_to_CSV("pd",test_data_pd)

if __name__ == '__main__':
    main()