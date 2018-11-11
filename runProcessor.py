import re
import csv
import matplotlib

matplotlib.use('TkAgg')
source = "runs/small-grid.out"


with open(source) as f:
	content = f.readlines()

data = []
last_episode = 1

for line in content:
	match = re.search(r'Episode #([0-9]*) .* 100 Wins Avg: ([0,1].[0-9]*) .* 100 Reward Avg: ([-]?[0-9]*.[0-9]*)', line)
	if match:
		episode = match.group(1)
		if last_episode == episode:
			data.pop()
			data.append(match.groups())
		else:
			data.append(match.groups())
		last_episode = episode

x = []
y = []

with open(source + '.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile, dialect='excel')
	for item in data:
		if int(item[0]) > 100000:
			break
		writer.writerow([item[0], item[1], item[2]])








