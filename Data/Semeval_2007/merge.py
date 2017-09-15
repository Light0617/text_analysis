import json
data, data1 = {}, {}
with open('trainData.json', 'r') as f:
	data = json.load(f)
with open('testData.json', 'r') as f:
	data1 = json.load(f)
for key in data1:
	data[key] = data1[key]
with open('allData.json', 'w') as f:
	json.dump(data, f, indent = 2)
