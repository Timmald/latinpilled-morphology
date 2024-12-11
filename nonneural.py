#!/usr/bin/env python3
"""
Non-neural baseline system for the SIGMORPHON 2020 Shared Task 0.
Author: Mans Hulden
Modified by: Tiago Pimentel
Modified by: Jordan Kodner
Modified by: Omer Goldman
Last Update: 22/03/2021
"""

import sys, os, getopt, re
from functools import wraps
from glob import glob
import pickle
import tqdm

allWordsReader = open("Latin_stuff/lat.dev","r",encoding="utf8").readlines()

def hamming(s,t):
    return sum(1 for x,y in zip(s,t) if x != y)


def halign(s,t):
    """Align two strings by Hamming distance."""
    slen = len(s)
    tlen = len(t)
    minscore = len(s) + len(t) + 1
    for upad in range(0, len(t)+1):
        upper = '_' * upad + s + (len(t) - upad) * '_'
        lower = len(s) * '_' + t
        score = hamming(upper, lower)
        if score < minscore:
            bu = upper
            bl = lower
            minscore = score

    for lpad in range(0, len(s)+1):
        upper = len(t) * '_' + s
        lower = (len(s) - lpad) * '_' + t + '_' * lpad
        score = hamming(upper, lower)
        if score < minscore:
            bu = upper
            bl = lower
            minscore = score

    zipped = list(zip(bu,bl))
    newin  = ''.join(i for i,o in zipped if i != '_' or o != '_')
    newout = ''.join(o for i,o in zipped if i != '_' or o != '_')
    return newin, newout


def levenshtein(s, t, inscost = 1.0, delcost = 1.0, substcost = 1.0):
    """Recursive implementation of Levenshtein, with alignments returned."""
    @memolrec
    def lrec(spast, tpast, srem, trem, cost):
        if len(srem) == 0:
            return spast + len(trem) * '_', tpast + trem, '', '', cost + len(trem)
        if len(trem) == 0:
            return spast + srem, tpast + len(srem) * '_', '', '', cost + len(srem)

        addcost = 0
        if srem[0] != trem[0]:
            addcost = substcost

        return min((lrec(spast + srem[0], tpast + trem[0], srem[1:], trem[1:], cost + addcost),
                   lrec(spast + '_', tpast + trem[0], srem, trem[1:], cost + inscost),
                   lrec(spast + srem[0], tpast + '_', srem[1:], trem, cost + delcost)),
                   key = lambda x: x[4])

    answer = lrec('', '', s, t, 0)
    return answer[0],answer[1],answer[4]


def memolrec(func):
    """Memoizer for Levenshtein."""
    cache = {}
    @wraps(func)
    def wrap(sp, tp, sr, tr, cost):
        if (sr,tr) not in cache:
            res = func(sp, tp, sr, tr, cost)
            cache[(sr,tr)] = (res[0][len(sp):], res[1][len(tp):], res[4] - cost)
        return sp + cache[(sr,tr)][0], tp + cache[(sr,tr)][1], '', '', cost + cache[(sr,tr)][2]
    return wrap


def alignprs(lemma, form):
    """Break lemma/form into three parts:
    IN:  1 | 2 | 3
    OUT: 4 | 5 | 6
    1/4 are assumed to be prefixes, 2/5 the stem, and 3/6 a suffix.
    1/4 and 3/6 may be empty.
    """

    al = levenshtein(lemma, form, substcost = 1.1) # Force preference of 0:x or x:0 by 1.1 cost
    alemma, aform = al[0], al[1]
    # leading spaces
    lspace = max(len(alemma) - len(alemma.lstrip('_')), len(aform) - len(aform.lstrip('_')))
    # trailing spaces
    tspace = max(len(alemma[::-1]) - len(alemma[::-1].lstrip('_')), len(aform[::-1]) - len(aform[::-1].lstrip('_')))
    return alemma[0:lspace], alemma[lspace:len(alemma)-tspace], alemma[len(alemma)-tspace:], aform[0:lspace], aform[lspace:len(alemma)-tspace], aform[len(alemma)-tspace:]


def prefix_suffix_rules_get(lemma, form):
    """Extract a number of suffix-change and prefix-change rules
    based on a given example lemma+inflected form."""
    lp,lr,ls,fp,fr,fs = alignprs(lemma, form) # Get six parts, three for in three for out

    # Suffix rules
    ins  = lr + ls + ">"
    outs = fr + fs + ">"
    srules = set()
    for i in range(min(len(ins), len(outs))):
        srules.add((ins[i:], outs[i:]))
    srules = {(x[0].replace('_',''), x[1].replace('_','')) for x in srules}

    # Prefix rules
    prules = set()
    if len(lp) >= 0 or len(fp) >= 0:
        inp = "<" + lp
        outp = "<" + fp
        for i in range(0,len(fr)):
            prules.add((inp + fr[:i],outp + fr[:i]))
            prules = {(x[0].replace('_',''), x[1].replace('_','')) for x in prules}

    return prules, srules


def apply_best_rule(lemma, msd, allprules, allsrules):
    """Applies the longest-matching suffix-changing rule given an input
    form and the MSD. Length ties in suffix rules are broken by frequency.
    For prefix-changing rules, only the most frequent rule is chosen."""

    bestrulelen = 0
    base = "<" + lemma + ">"
    if msd not in allprules and msd not in allsrules:
        return lemma # Haven't seen this inflection, so bail out
    
    #THE BELOW IS MY ADDITION FOR THE 1ST-3RD CONJ AMBIGUITY
    elif msd.startswith("V;") and lemma.endswith("ō"):
        noSuff = lemma[:-1]
        for word in allWordsReader:
            lemma2,msd2,form = word.replace("\n","").split("\t")
            if lemma2 == lemma and "NFIN" in msd2:#LOOK FOR INFINITIVE
                break
        if form == 'Icarūsa':
            pass 
        elif form.endswith("āre"):#1st conj
            conjDict = {"V;IND;ACT;PRS;1;SG":("ō","ō"),
                        "V;IND;ACT;PRS;2;SG":("ō","ās"),
                        "V;IND;ACT;PRS;3;SG":("ō","at"),
                        "V;IND;ACT;PRS;1;PL":("ō","āmus"),
                        "V;IND;ACT;PRS;2;PL":("ō","ātis"),
                        "V;IND;ACT;PRS;3;PL":("ō","ant"),
                        "V;IND;ACT;PST;IPFV;1;SG":("ō","ābam"),
                        "V;IND;ACT;PST;IPFV;2;SG":("ō","ābās"),
                        "V;IND;ACT;PST;IPFV;3;SG":("ō","ābat"),
                        "V;IND;ACT;PST;IPFV;1;PL":("ō","āmus"),
                        "V;IND;ACT;PST;IPFV;2;PL":("ō","ātis"),
                        "V;IND;ACT;PST;IPFV;3;PL":("ō","bant"),
                        "V;IND;ACT;FUT;1;SG":("ō","ābō"),
                        "V;IND;ACT;FUT;2;SG":("ō","ābis"),
                        "V;IND;ACT;FUT;3;SG":("ō","ābit"),
                        "V;IND;ACT;FUT;1;PL":("ō","ābimus"),
                        "V;IND;ACT;FUT;2;PL":("ō","ābitis"),
                        "V;IND;ACT;FUT;3;PL":("ō","ābunt"),
                        "V;SBJV;ACT;PRS;1;SG":("ō","em"),
                        "V;SBJV;ACT;PRS;2;SG":("ō","ēs"),
                        "V;SBJV;ACT;PRS;3;SG":("ō","et"),
                        "V;SBJV;ACT;PRS;1;PL":("ō","ēmus"),
                        "V;SBJV;ACT;PRS;2;PL":("ō","ētis"),
                        "V;SBJV;ACT;PRS;3;PL":("ō","ent"),
                        "V;SBJV;ACT;PST;IPFV;1;SG":("ō","ārem"),
                        "V;SBJV;ACT;PST;IPFV;2;SG":("ō","ārēs"),
                        "V;SBJV;ACT;PST;IPFV;3;SG":("ō","āret"),
                        "V;SBJV;ACT;PST;IPFV;1;PL":("ō","ārēmus"),
                        "V;SBJV;ACT;PST;IPFV;2;PL":("ō","ārētis"),
                        "V;SBJV;ACT;PST;IPFV;3;PL":("ō","ārent"),
                        "V;IMP;ACT;PRS;2;SG":("ō","ā"),
                        "V;IMP;ACT;PRS;2;PL":("ō","āte"),
                        "V;IMP;ACT;FUT;2;SG":("ō","ātō"),
                        "V;IMP;ACT;FUT;3;SG":("ō","ātō"),
                        "V;IMP;ACT;FUT;2;PL":("ō","ātōte"),
                        "V;IMP;ACT;FUT;3;PL":("ō","antō"),
                        "V;NFIN;ACT;PRS":("ō","āre"),
                        "V;V.PTCP;ACT;PRS":("ō","āns"),
                        "V;V.MSDR;GEN":("ō","andī"),
                        "V;V.MSDR;DAT":("ō","andō"),
                        "V;V.MSDR;ACC":("ō","andum"),
                        "V;V.MSDR;ABL":("ō","andō"),
                    }
            return noSuff+conjDict[msd][1]
        elif form.endswith("ēre"):#3rd conj
            pass
                

    elif msd in allsrules:
        applicablerules = [(x[0],x[1],y) for x,y in allsrules[msd].items() if x[0] in base]
        if applicablerules:
            bestrule = max(applicablerules, key = lambda x: (len(x[0]), x[2], len(x[1])))
            base = base.replace(bestrule[0], bestrule[1])

    elif msd in allprules:
        applicablerules = [(x[0],x[1],y) for x,y in allprules[msd].items() if x[0] in base]
        if applicablerules:
            bestrule = max(applicablerules, key = lambda x: (x[2]))
            base = base.replace(bestrule[0], bestrule[1])

    base = base.replace('<', '')
    base = base.replace('>', '')
    return base


def numleadingsyms(s, symbol):
    return len(s) - len(s.lstrip(symbol))


def numtrailingsyms(s, symbol):
    return len(s) - len(s.rstrip(symbol))

###############################################################################


def main(argv):
    options, remainder = getopt.gnu_getopt(argv[1:], 'ohp:', ['output','help','path='])
    TEST, OUTPUT, HELP, path = False,False, False, './Latin_stuff/'
    for opt, arg in options:
        if opt in ('-o', '--output'):
            OUTPUT = True
        if opt in ('-t', '--test'):
            TEST = True
        if opt in ('-h', '--help'):
            HELP = True
        if opt in ('-p', '--path'):
            path = arg

    if HELP:
            print("\n*** Baseline for the SIGMORPHON 2020 shared task ***\n")
            print("By default, the program runs all languages only evaluating accuracy.")
            print("To create output files, use -o")
            print("The training and dev-data are assumed to live in ./part1/development_languages/")
            print("Options:")
            print(" -o         create output files with guesses (and don't just evaluate)")
            print(" -t         evaluate on test instead of dev")
            print(" -p [path]  data files path. Default is ./Latin_stuff")
            quit()

    totalavg, numlang = 0.0, 0
    allprules, allsrules = {}, {}
    lang="lat"
    if not os.path.isfile(path + lang +  ".trn"):
        exit(0)
    lines = [line.strip() for line in open(path + lang + ".trn", "r", encoding='utf8') if line != '\n']
    
    if not os.path.exists("prefsuffbias"):
        print("CHECKING PREFIX BIAS")
    # First, test if language is predominantly suffixing or prefixing
    # If prefixing, work with reversed strings
        prefbias, suffbias = 0,0
        for l in tqdm.tqdm(lines):
            lemma, _, form = l.split(u'\t')
            aligned = halign(lemma, form)
            if ' ' not in aligned[0] and ' ' not in aligned[1] and '-' not in aligned[0] and '-' not in aligned[1]:
                prefbias += numleadingsyms(aligned[0],'_') + numleadingsyms(aligned[1],'_')
                suffbias += numtrailingsyms(aligned[0],'_') + numtrailingsyms(aligned[1],'_')

        with open("prefsuffbias","w") as writer:
            writer.writelines([str(prefbias)+"\n",str(suffbias)+"\n"])
    else:
        with open("prefsuffbias","r") as reader:
            prefbias = int(reader.readline())
            suffbias = int(reader.readline())
    if not os.path.exists("prules.json") or not os.path.exists("srules.json"):
        print("FINDING RULES")
        for l in tqdm.tqdm(lines): # Read in lines and extract transformation rules from pairs
            lemma, msd, form = l.split(u'\t')
            if prefbias > suffbias:
                lemma = lemma[::-1]
                form = form[::-1]
            prules, srules = prefix_suffix_rules_get(lemma, form)

            if msd not in allprules and len(prules) > 0:
                allprules[msd] = {}
            if msd not in allsrules and len(srules) > 0:
                allsrules[msd] = {}

            for r in prules:
                if (r[0],r[1]) in allprules[msd]:
                    allprules[msd][(r[0],r[1])] = allprules[msd][(r[0],r[1])] + 1
                else:
                    allprules[msd][(r[0],r[1])] = 1

            for r in srules:
                if (r[0],r[1]) in allsrules[msd]:
                    allsrules[msd][(r[0],r[1])] = allsrules[msd][(r[0],r[1])] + 1
                else:
                    allsrules[msd][(r[0],r[1])] = 1
        with open("prules.json","wb") as pwriter:
            pickle.dump(allprules,pwriter)
        with open("srules.json","wb") as swriter:
            pickle.dump(allsrules,swriter)
    else:
        with open("prules.json","rb") as preader:
            allprules = pickle.load(preader)
        with open("srules.json","rb") as sreader:
            allsrules = pickle.load(sreader)

    # Run eval on dev
    print("RUNNING EVAL ON DEV")
    devlines = [line.strip() for line in open(path + lang + ".dev", "r", encoding='utf8') if line != '\n']
    if TEST:
        devlines = [line.strip() for line in open(path + lang + ".tst", "r", encoding='utf8') if line != '\n']
    numcorrect = 0
    numguesses = 0
    if OUTPUT:
        outfile = open(path + lang + ".out", "w", encoding='utf8')
    for l in tqdm.tqdm(devlines):
        lemma, msd, correct = l.split(u'\t')
#                    lemma, msd, = l.split(u'\t')
        if prefbias > suffbias:
            lemma = lemma[::-1]
        outform = apply_best_rule(lemma, msd, allprules, allsrules)
        if prefbias > suffbias:
            outform = outform[::-1]
            lemma = lemma[::-1]
        if outform == correct:
            numcorrect += 1
        numguesses += 1
        if OUTPUT:
            outfile.write(lemma + "\t" + msd + "\t" + outform + "\n")

    if OUTPUT:
        outfile.close()

    numlang += 1
    totalavg += numcorrect/float(numguesses)

    print(lang + ": " + str(str(numcorrect/float(numguesses)))[0:7])

    print("Average accuracy", totalavg/float(numlang))


if __name__ == "__main__":
    main(sys.argv)
