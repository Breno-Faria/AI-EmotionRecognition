import json

fo = open("./results/engaged_images.txt")
arr = json.load(fo)

i = 0
for item in arr:
    i+=1

print(i)