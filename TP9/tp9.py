"""
PETIT Antoine & WISSOCQ Sarah
"""

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

def occurence_word(list_phrases):

	occurences= {} #une bibliotheque vide

	for phrase in list_phrases :
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

def get_next_world(mot,word_dictionnaire):

	proba = 0
	for key in word_dictionnaire[mot]:
		if word_dictionnaire[mot][key] > proba:
			res = key
			proba = word_dictionnaire[mot][key]
	return res


list = list_sentences("sentences.txt")

occurence = occurence_word(list)

print get_next_world("est",occurence)

#prendre en entre phrase + nb mot apres phrase