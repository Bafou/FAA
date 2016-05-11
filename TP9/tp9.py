"""
PETIT Antoine & WISSOCQ Sarah
"""

import sys
import re
import unicodedata

def list_sentences(file):
    txt = []

    with open(file, "r") as myfile:
        for sentences in myfile :
            tmp = re.sub("^[0-9]+   ", "", sentences)
            tmp = re.sub("\n","",tmp)
            tmp = unicode(tmp,'utf-8')
            tmp = unicodedata.normalize('NFD', tmp).encode('ascii', 'ignore')
            tmp = tmp.lower()
            tmp = re.sub("[],()?:.!\"/#*]","",tmp)
            txt.append(tmp)
    return txt

def occurence_word(list_sentences):

    occurences= {} #un dictionnaire vide

    for phrase in list_sentences :
        mots = phrase.split(" ")
        while '' in mots : mots.remove( '' )

        for i in range (0, len(mots) -1):
            if (not (mots[i] in occurences)):
                occurences[mots[i]] = {}

            if (mots[i+1] in occurences [mots[i]]):
                occurences[mots[i]][mots[i+1]] += 1
            else : 
                occurences[mots[i]][mots[i+1]] = 1

    return occurences

def get_next_world_order1(word,word_dictionnaire):

    proba = 0
    for key in word_dictionnaire[word]:
        if word_dictionnaire[word][key] > proba:
            res = key
            proba = word_dictionnaire[word][key]
    return res

def get_useful_word(phrase, order):

    words = phrase.split(" ")
    res = []
    for i in (0,order):
        res.append(words[len(words) - 1 - i])
    return res

def get_next_world_order2(list_useful_word,word_dictionnaire):
    last = list_useful_word[0]
    before_last = list_useful_word[1]
    proba_last = 1.
    if before_last in word_dictionnaire:
        if last in word_dictionnaire[before_last]:
            proba_last = float(word_dictionnaire[before_last][last]) / float(sum([v for k,v in word_dictionnaire[before_last].iteritems()]))
    proba_last = proba_last * 1000.
    if last in word_dictionnaire:
        predict = ""
        predict_proba = 0.
        itemsLast = word_dictionnaire[last].iteritems()
        total = float(sum([v for k,v in itemsLast]))
        for k, v in word_dictionnaire[last].iteritems():
            proba = (float(v)/total)*1000.
            proba_cond = ((proba*proba_last)/proba)
            if proba_cond > predict_proba:
                predict_proba = proba_cond
                predict = k

                return predict
            else:
                return ""

if __name__ == "__main__":

    try :
        phrase = sys.argv[1]
        order = int(sys.argv[2])
        nb_word = int(sys.argv[3])
        if not(order == 1 or order == 2 or order == 3):
            print "L'ordre de la chaine de markov doit etre compris entre 1 et 3"
            sys.exit()
    except IndexError :
        print "Pour utiliser le programme il faut entrer la commande suivante : python2 tp9.py <phrase a complete> <ordre> <nb_mots>\nOrdre droit etre compris entre 1 et 3 et designe le nombre de mots considere pour trouver le mot suivant"
        sys.exit()
    except ValueError : 
        print "Attention l'orde et le nombre de mots a predire doivent etre des entiers"
        sys.exit()

    list = list_sentences("sentences.txt")


    occurence = occurence_word(list)


    if (order == 1):
        print "Phrase predit avec une chaine markovienne  d'ordre 1 :"

        for i in range(nb_word):
            phrase += " " + get_next_world_order1(get_useful_word(phrase,order)[0],occurence)
        print phrase
    elif (order == 2):
        print "Phrase predit avec une chaine markovienne  d'ordre 2 :"
        for i in range (nb_word):
            phrase += " " + get_next_world_order2(get_useful_word(phrase,order),occurence)
        print phrase
#prendre en entree phrase + nb mot apres phrase