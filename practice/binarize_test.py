from sklearn.preprocessing import binarize


test_Data = [[1,4,30,2,4000,90,20]]
test_data = [[]]
inner_arr = []
for i in range(101):
	if i%5==0:
		test_data[0].append(inner_arr)
		inner_arr = []
		inner_arr.append(i)
	else:
		inner_arr.append(i)	

binarized_data = binarize(test_Data, threshold=50)

print(binarized_data)

