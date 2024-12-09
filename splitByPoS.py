from sklearn.model_selection import train_test_split
import pandas as pd

# Put Latin data into a dataframe with columns Lemma inflected, inflection, making sure to skip lines with +'s in the inflection
lines = []
with open('Latin_stuff/lat') as f:
    lines = [line.rstrip('\n') for line in f]
    
lat = pd.read_table("Latin_stuff/lat", sep='\t', names=['Lemon', 'Infected', 'Infection'], skiprows=lambda x: '+' in lines[x])

# Label every row with the part of speech extracted from the inflection: N, PROPN, V, V.PTCP, ADJ
lat['PartoSpeech'] = lat['Infection'].str.extract(r'(N|PROPN|V|V.PTCP|ADJ);')

# get a list of the unique lemmas and a list of the number of parts of speech
uniqueLemmas = lat.drop_duplicates(subset = ['Lemon'])

partSample = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'V.PTCP'].sample(n=112)
adjSample = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'ADJ'].sample(n=105)
nounSample = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'N'].sample(n=185)
verbSample = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'V'].sample(n=41)
propSample = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'PROPN'].sample(n=343)

partTrain, partTest = train_test_split(partSample, test_size=2000/12000, random_state=56)
partTest, partDev = train_test_split(partTest, test_size=0.5, random_state=56)

adjTrain, adjTest = train_test_split(adjSample, test_size=2000/12000, random_state=56)
adjTest, adjDev = train_test_split(adjTest, test_size=0.5, random_state=56)

nounTrain, nounTest = train_test_split(nounSample, test_size=2000/12000, random_state=56)
nounTest, nounDev = train_test_split(nounTest, test_size=0.5, random_state=56)

verbTrain, verbTest = train_test_split(verbSample, test_size=2000/12000, random_state=56)
verbTest, verbDev = train_test_split(verbTest, test_size=0.5, random_state=56)

propTrain, propTest = train_test_split(propSample, test_size=2000/12000, random_state=56)
propTest, propDev = train_test_split(propTest, test_size=0.5, random_state=56)

uniqueTrain = pd.concat([partTrain, adjTrain, nounTrain, verbTrain, propTrain])
uniqueTest = pd.concat([partTest, adjTest, nounTest, verbTest, propTest])
uniqueDev = pd.concat([partDev, adjDev, nounDev, verbDev, propDev])

def getlist(lemons):
    splitslist = []
    for lemon in lemons['Lemon']:
        for row in lat[lat['Lemon'] == lemon].to_numpy().tolist():
            splitslist.append(row)
    return pd.DataFrame(splitslist, columns= ["Lemon", "Infected", "Infection", "PartoSpeech"])

train = getlist(uniqueTrain)
test = getlist(uniqueTest)
dev = getlist(uniqueDev)

print("Train set size", train.shape)
print("Test set size", test.shape)
print("Dev set size", dev.shape)

print("Train set split into parts of speech")
print(train[train['PartoSpeech'] == 'V.PTCP'].shape)
print(train[train['PartoSpeech'] == 'ADJ'].shape)
print(train[train['PartoSpeech'] == 'N'].shape)
print(train[train['PartoSpeech'] == 'V'].shape)
print(train[train['PartoSpeech'] == 'PROPN'].shape)

print("Test set split into parts of speech")
print(test[test['PartoSpeech'] == 'V.PTCP'].shape)
print(test[test['PartoSpeech'] == 'ADJ'].shape)
print(test[test['PartoSpeech'] == 'N'].shape)
print(test[test['PartoSpeech'] == 'V'].shape)
print(test[test['PartoSpeech'] == 'PROPN'].shape)

print("Dev set split into parts of speech")
print(dev[dev['PartoSpeech'] == 'V.PTCP'].shape)
print(dev[dev['PartoSpeech'] == 'ADJ'].shape)
print(dev[dev['PartoSpeech'] == 'N'].shape)
print(dev[dev['PartoSpeech'] == 'V'].shape)
print(dev[dev['PartoSpeech'] == 'PROPN'].shape)

def toFile(frame, fileName, fileType):
    frame.to_csv(path_or_buf= './Latin_stuff/' + fileName + fileType,sep= "\t", encoding= "utf8", index= False, header=False, columns= ["Lemon", "Infected", "Infection"])

toFile(train, 'lat', '.trn')
toFile(test, 'lat', '.tst')
toFile(dev, 'lat', '.dev')