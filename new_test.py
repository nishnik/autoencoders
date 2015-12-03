global_list = []
for i in range(1,10):
	global_list.append(i)

import matplotlib.pyplot as plt
plt.plot(range(len(global_list)),global_list)
plt.show()