import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

with open("runs/smallClassic.out") as f:
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

for item in data:
	if int(item[0]) > 10000:
		break
	if int(item[0]) % 100 == 0:
		x.append(item[0])
		y.append(item[1])


plt.plot(x, y)
plt.xlabel('Episodes')
plt.ylabel('100 Games Win Rate')
plt.show()




