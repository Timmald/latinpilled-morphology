import random

lines = [line.strip() for line in open("./Latin_stuff/ORIGINAL_lat.trn", "r", encoding='utf8') if line != '\n']
random.shuffle(lines)
dev = [line+"\n" for line in lines[:int(len(lines)*.1)]]
train = [line+"\n" for line in lines[int(len(lines)*.1):]]
with open("./Latin_stuff/lat.dev","w") as writer:
    writer.writelines(dev)
with open("./Latin_stuff/lat.trn","w") as writer:
    writer.writelines(train)

