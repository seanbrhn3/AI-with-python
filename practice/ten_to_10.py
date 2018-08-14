from sklearn import preprocessing

numbers = ['one','two','three','four','five','six','seven','eight','nine','ten']

encoder = preprocessing.LabelEncoder()

encoder.fit(numbers)
#enumerate that shit!
for i, item in enumerate(encoder.classes_):
	print(i, item)


