def calculate_medium(file):
    with open(file,"r") as file:
        length = []
        for data in file.readlines():
            length.append(len(data.split(" "))-1)
        length = sorted(length)
        print "medium is: ", length[len(length)/2]


calculate_medium("filter_data1")