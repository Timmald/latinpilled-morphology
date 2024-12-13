from sklearn.model_selection import train_test_split
import pandas as pd

# Put Latin data into a dataframe with columns Lemma inflected, inflection, making sure to skip lines with +'s or spaces in them
lines = []
with open('Latin_stuff/lat_in.trn') as f:
    lines = [line.rstrip('\n') for line in f]
    
lat = pd.read_table("Latin_stuff/lat_in.trn", sep='\t', names=['Lemon', 'Infected', 'Infection'], skiprows=lambda x: '+' in lines[x])
print(lat.shape)
# Label every row with the part of speech extracted from the inflection: N, PROPN, V, V.PTCP, ADJ
lat['PartoSpeech'] = lat['Infection'].str.extract(r'(N|PROPN|V|V.PTCP|ADJ);')

# # get a list of the unique lemmas and a list of the number of parts of speech
infinitives = lat[lat['Infection'] == 'V;NFIN;ACT;PRS']
uniqueLemmas = lat.drop_duplicates(subset = ['Lemon', 'PartoSpeech'])

# Split the unique lemmas into dataframes by part of speech while also cutting it down using numbers I calculated elsewhere
partSample = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'V.PTCP'].sample(n=112)
adjSample = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'ADJ'].sample(n=105)
nounSample = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'N'].sample(n=185)
verbSample = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'V'].sample(n=41)
propSample = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'PROPN'].sample(n=343)

# Split the dataframes randomly into train, test, and dev sets in a 10:1:1 ratio
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

# Concatenate the dataframes together to get all the unique lemmas in the test, train, and dev sets
uniqueTrain = pd.concat([partTrain, adjTrain, nounTrain, verbTrain, propTrain])
uniqueTest = pd.concat([partTest, adjTest, nounTest, verbTest, propTest])
uniqueDev = pd.concat([partDev, adjDev, nounDev, verbDev, propDev])

# Method to get all of the other lemmas which match with the lemmas in the unique list and return it as a dataframe
def getlist(lemons):
    splitslist = []
    for lemon in lemons['Lemon']:
        for row in lat[lat['Lemon'] == lemon].to_numpy().tolist():
            splitslist.append(row)
    return pd.DataFrame(splitslist, columns= ["Lemon", "Infected", "Infection", "PartoSpeech"])

# Call the method to get the full train, test, and dev sets
train = getlist(uniqueTrain)
test = getlist(uniqueTest)
dev = getlist(uniqueDev)

# Print the sizes of the train, test, and dev sets to check that they look right
print("Train set size", train.shape)
print("Test set size", test.shape)
print("Dev set size", dev.shape)

# Print the number of rows per part of speech in the training set to check that they look right
print("Train set split into parts of speech")
print(train[train['PartoSpeech'] == 'V.PTCP'].shape)
print(train[train['PartoSpeech'] == 'ADJ'].shape)
print(train[train['PartoSpeech'] == 'N'].shape)
print(train[train['PartoSpeech'] == 'V'].shape)
print(train[train['PartoSpeech'] == 'PROPN'].shape)

# Print the number of rows per part of speech in the test set to check that they look right
print("Test set split into parts of speech")
print(test[test['PartoSpeech'] == 'V.PTCP'].shape)
print(test[test['PartoSpeech'] == 'ADJ'].shape)
print(test[test['PartoSpeech'] == 'N'].shape)
print(test[test['PartoSpeech'] == 'V'].shape)
print(test[test['PartoSpeech'] == 'PROPN'].shape)

# Print the number of rows per part of speech in the dev set to check that they look right
print("Dev set split into parts of speech")
print(dev[dev['PartoSpeech'] == 'V.PTCP'].shape)
print(dev[dev['PartoSpeech'] == 'ADJ'].shape)
print(dev[dev['PartoSpeech'] == 'N'].shape)
print(dev[dev['PartoSpeech'] == 'V'].shape)
print(dev[dev['PartoSpeech'] == 'PROPN'].shape)

# Write a method to convert the dataframes to files based on Emily's code
def toFile(frame, fileName, fileType):
    frame.to_csv(path_or_buf= './Latin_stuff/' + fileName + fileType,sep= "\t", encoding= "utf8", index= False, header=False, columns= ["Lemon", "Infected", "Infection"])

# Convert the test, train, and dev sets to files
toFile(train, 'lat', '.trn')
toFile(test, 'lat', '.tst')
toFile(dev, 'lat', '.dev')