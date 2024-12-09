#from sklearn.model_selection import train_test_split
import pandas as pd

# Put it into a dataframe
lines = []
with open('Latin_stuff/lat', encoding="utf8") as f:
    lines = [line for line in f]
lat = pd.read_table("Latin_stuff/lat.txt", sep='\t', names=['Lemon', 'Infected', 'Infection'], skiprows=lambda x: '+' in lines[x])
# I replaced all spaces with hyphens because those entries were messing with read_table
# After reading it in, I replaced the hyphens with spaces
lat = lat.replace('-', ' ', regex=True)

# Separating the data into given and inferred by the program
X = lat[['Lemon', 'Infection']]
y = lat[['Infected']]

# Split data into training and test sets, its a 9:1 split, and 89 just ensures that it's split the same way every time
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=89)

lat['PartoSpeech'] = lat['Infection'].str.extract(r'(N|PROPN|V|V.PTCP|ADJ);')
uniqueLemmas = lat.drop_duplicates(subset = ['Lemon'])
uLemonlen = len(uniqueLemmas)/90
vnum = int(uLemonlen*0.06690776468800437)
vpnum = int(uLemonlen*0.4402412287420541)
adjnum = int(uLemonlen*0.23140998224240768)
nnum = int(uLemonlen * 0.19798428655288244)
pnum = int(uLemonlen * 0.06345673777465137)

vLemon = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'V'].sample(vnum)
vpLemon = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'V.PTCP'].sample(vpnum)
adjLemon = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'ADJ'].sample(adjnum)
nLemon = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'N'].sample(nnum)
pLemon = uniqueLemmas[uniqueLemmas['PartoSpeech'] == 'PROPN'].sample(pnum)


splitLemons = pd.concat([vLemon,vpLemon,adjLemon,nLemon,pLemon])
def getList(lemons):
    splitslist = []
    for lemon in lemons['Lemon']:
        for row in lat[lat['Lemon'] == lemon].to_numpy().tolist():
            splitslist.append(row)
    return pd.DataFrame(splitslist, columns= ["Lemon", "Infected", "Infection", "PartoSpeech"])
splits = getList(splitLemons)
print(splits)
splits.to_csv(path_or_buf= 'Latin_stuff\lat.trn',sep= "\t", encoding= "utf8", index= False, header=False, columns= ["Lemon", "Infected", "Infection"])


# this is super ineffient but whatever
def removeDups(listoLemons:pd.DataFrame, uniqueL:pd.DataFrame) -> pd.DataFrame:
    return uniqueL[~uniqueL['Lemon'].isin(listoLemons)]
Lnum = int(uLemonlen / 50)
# code for lat.tst
tstUL = removeDups(splitLemons, uniqueLemmas)
tstvLemon = tstUL[tstUL['PartoSpeech'] == 'V'].sample(Lnum)
tstvpLemon = tstUL[tstUL['PartoSpeech'] == 'V.PTCP'].sample(Lnum)
tstadjLemon = tstUL[tstUL['PartoSpeech'] == 'ADJ'].sample(Lnum)
tstnLemon = tstUL[tstUL['PartoSpeech'] == 'N'].sample(Lnum)
tstpLemon = tstUL[tstUL['PartoSpeech'] == 'PROPN'].sample(Lnum)

tstLemons = pd.concat([tstvLemon,tstvpLemon,tstadjLemon,tstnLemon,tstpLemon])
tst = getList(tstLemons)
print(tst)
tst.to_csv(path_or_buf= 'Latin_stuff\lat.tst',sep= "\t", encoding= "utf8", index= False, header=False, columns= ["Lemon", "Infected", "Infection"])

#Code for lat.dev
devUL = removeDups(splitLemons, tstUL)

devvLemon = devUL[devUL['PartoSpeech'] == 'V'].sample(Lnum)
devvpLemon = devUL[devUL['PartoSpeech'] == 'V.PTCP'].sample(Lnum)
devadjLemon = devUL[devUL['PartoSpeech'] == 'ADJ'].sample(Lnum)
devnLemon = devUL[devUL['PartoSpeech'] == 'N'].sample(Lnum)
devnpLemon = devUL[devUL['PartoSpeech'] == 'PROPN'].sample(Lnum)

devLemons = pd.concat([devvLemon,devvpLemon,devadjLemon,devnLemon,devnpLemon])
dev = getList(devLemons)
print(dev)
dev.to_csv(path_or_buf= 'Latin_stuff\lat.dev',sep= "\t", encoding= "utf8", index= False, header=False, columns= ["Lemon", "Infected", "Infection"])