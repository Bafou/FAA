"""
PETIT Antoine & WISSOCQ Sarah
"""

import re
import unicodedata

def list_sentences(file):
	txt = []

	with open(file, "r") as myfile:
		for sentences in myfile :
			tmp = re.sub("^[0-9]+\t", "", sentences)
			tmp = re.sub("\n","",tmp)
			tmp = unicode(tmp,'utf-8')
			tmp = unicodedata.normalize('NFD', tmp).encode('ascii', 'ignore')
			tmp = tmp.lower()
			tmp = re.sub("[],()?:.!\"/#*]","",tmp)
			txt.append(tmp)
	return txt
 
def occurence(list):
	pass

list = list_sentences("sentences.txt")

for element in list : 
	print element