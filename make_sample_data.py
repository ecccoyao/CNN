
input_file = open('filter_data', 'r')

output_file1 = open('sample2.txt','w')


with input_file as f:
    lines = f.readlines()
    count = 0;
    for line in lines:
        count += 1
        if count % 2 == 0:
            output_file1.write(line)

input_file.close()
output_file1.close()

