import time

with open('./Documents/Prelab_LARSEN/handful-of-trials/scripts/test.txt', 'a') as f:
	f.write(str(time.time())+"\n")
f.close()
