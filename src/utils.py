from collections import Counter

with open('../Dataset_A1/brown_plain.txt') as file:
    data = file.readlines()

len_data = []
for i in data:
    len_data.append(round(len(i),-3))

fin = Counter(len_data)
print(fin)

cumm_sum = 0
for i in range(1000,22000,1000):
    cumm_sum += fin[i]
    print(i, cumm_sum)