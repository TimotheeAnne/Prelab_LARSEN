import time

with open('test.txt', 'a') as f:
	f.write(str(time.time())+"\n")
f.close()
