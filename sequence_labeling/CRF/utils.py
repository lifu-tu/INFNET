from scipy.io import loadmat
import numpy as np
import math
from random import shuffle
from random import choice
from random import randint
from theano import tensor as T

def lookupWordsIDX(words,w):
    if w in words:
        return words[w]
    else:
        #print 'find UUUNKKK words',w
        return words['UUUNKKK']

def lookupTaggerIDX(tagger,w):
    #w = w.lower()
    if w in tagger:
        return tagger[w]
    else:
        #print 'find UUUNKKK words',
	print w
        return tagger['*']

def lookup_with_unk(We,words,w):
    if w in words:
        return We[words[w],:],False
    else:
        #print 'find Unknown Words in WordSim Task',w
        return We[words['UUUNKKK'],:],True

def lookupwordID(words,array):
    #w = w.strip()
    result = []
    for i in range(len(array)):
        if(array[i] in words):
            result.append(words[array[i]])
        else:
            #print "Find Unknown Words ",w
            result.append(words['UUUNKKK'])
    return result

def lookupTaggerID(tagger, array):
    #w = w.strip()
    result = []
    for i in range(len(array)):
        if(array[i] in tagger):
            result.append(tagger[array[i]])
        else:
            #print "Find Unknown tagger", array[i]
            result.append(tagger['*'])
    return result


def getData(f, words, tagger):
    data = open(f,'r')
    lines = data.readlines()
    X = []
    Y = []
    for i in lines:
        if(len(i) > 0):
	    index = i.find('|||')
	    if index == -1:
		print('file error\n')
		return None
	    x = i[:index-1]
	    y = i[index+4:-1]
	    x = x.split(' ')
	    y = y.split(' ')
            #print x
	    #print y
	    x = lookupwordID(words, x)
            y = lookupTaggerID(tagger, y)
	    #print y
            X.append(x)
	    Y.append(y)
   
    return X, Y


def getSupertagData(f, words, tagger, train = True, maxlen=50):
    data = open(f,'r')
    lines = data.readlines()
    X = []
    Y = []
    for i in lines:
        if(len(i) > 0):
            index = i.find('|||')
            if index == -1:
                print('file error\n')
                return None
            x = i[:index-1]
            y = i[index+3:-1]
            x = x.split(' ')
            y = y.split(' ')
	    
	    if (len(x) > maxlen) and train:
		continue
            x = lookupwordID(words, x)
            y = lookupTaggerID(tagger, y)
            #print y
            X.append(x)
            Y.append(y)

    return X, Y



def getptbData(f, words, tagger):
    data = open(f,'r')
    lines = data.readlines()
    X = []
    Y = []
    x = []
    y = []
    unknown = []
    unindex = []
    i =0     
    for line in lines:
	i = i+1
	line = line[:-1]
	if len(line)>0:
		line = line.split('\t')
		a = line[0].lower()
		if a =='-semicolon-':
			a = ';'
		index = lookupWordsIDX(words, a)	
		x.append(index)
		if a not in words:
			#print a, i
			unknown.append(a)
			unindex.append(i)
		yindex = lookupTaggerIDX(tagger, line[1])
		y.append(yindex)
        else:
		X.append(x)
		Y.append(y)
		x = []
		y = []
    print  'the number of unknow words', len(unknown)
    print 'first example', X[0]
    return X, Y


def getUnlabeledData(f, words):
    data = open(f,'r')
    lines = data.readlines()
    X = []
    Y = []
    for i in lines:
        if(len(i) > 0):
            x = i[:-1]
            x = x.split(' ')
            x = lookupwordID(words, x)
            #print y
            X.append(x)
    return X


def Get_Ner_bioes(filename, words, tagger):
        f = open(filename,'r')
        data = f.readlines()
        f.close()
        X = []
        Y = []
        RAWX = []
        POS = []
        x=  []
        y = []
        rawx = []
        pos = []
        for line in data:
            line = line[:-1]
            if (len(line)>0):
                line = line.split(' ')
                #print line
                pos.append(line[1])
                rawx.append(line[0])
                x.append(lookupWordsIDX(words, line[0].lower()))
                y.append(lookupTaggerIDX(tagger , line[3]))
            else:
                X.append(x)
                Y.append(y)
                RAWX.append(rawx)
                POS.append(pos)
                x= []
                y= []
                rawx = []
                pos = []
        return X, Y, RAWX, POS


def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    f.close()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]= n
        We.append(v)
    return (words, We)


def getGloveWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    f.close()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]= n+1
        We.append(v)
    return (words, We)


def getTagger(Taggerfile):
    tag = {}
    f = open(Taggerfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i = i.strip()
        tag[i] = n
    return tag

def getTaggerlist(Taggerfile):
    tag = []
    f = open(Taggerfile,'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        tag.append(line[:-1])
    return tag
