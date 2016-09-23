import sys

a_double = []
a        = []
data = []
col = int(sys.argv[1])
row = 0

with open(sys.argv[2], "r") as f:
	for line in f.readlines():
		a_double.append(line.split())


for i in range(len(a_double)):
	a.append(a_double[i][col])

a.sort(key=float)

for i in range(len(a)-1):
	print a[i], 
	sys.stdout.softspace=False
	print ",",
	sys.stdout.softspace=False

print a[len(a)-1],
sys.stdout.softspace=False





