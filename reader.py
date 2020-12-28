# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import pandas as pd
import pdb

from os import listdir
from os.path import join
from collections import Counter
from csv import QUOTE_ALL

def add_node(graph, word):
	if not graph.has_node(word):
		graph.add_node(word)

def add_edge(graph, last_word, word):
	if last_word is not None:
		if graph.has_edge(last_word, word):
			graph[last_word][word]['weight'] += 1
		else:
			graph.add_edge(last_word, word, weight=1)

def read_colonia_file(filename, should_remove_stopwords=True, should_separate_sentences=True):
	if should_remove_stopwords:
		sw_groups = ["PRP", "VIRG", "SENT", "PR", "CONJ", "DET", "QUOTE", "PRP+DET", \
			"CODE", "ID", "D-F", "P", "D", ",", "CONJSUB", "C", "NEG", "PONFP", "P+D", "D-UM-F", \
			"D-UM", "PRO", "CL", ".", "P+D-F-P", "P+D-F", "PRO$-F", "PRO$"]
		sw_lemmas = ["<unknown>"]
	else:
		sw_groups = ["QUOTE", "SENT", "VIRG", "CODE"]
		sw_lemmas = []

	file = open(filename, 'r', encoding="utf-8") 

	G = nx.DiGraph()
	word_counter = Counter()
	last_word = None

	for line in file.readlines() : 
		# Text lines have the format 'word \t syntactic group \t lemma'
		# Other lines indicate sentences, paragraphs, etc. These are ignored
		parsed = line.strip().split("\t")

		if should_separate_sentences:
			if parsed == ["</s>"] or parsed == ["<s>"] or (len(parsed) > 1 and parsed[1] == "SENT"):
				last_word = None

		if len(parsed) == 3 \
			and parsed[1] not in sw_groups \
			and parsed[2] not in sw_lemmas:

			lemma = parsed[2].lower()
			if parsed[2] == "<unknown>":
				lemma = parsed[0].lower()
			word = lemma
			word_counter[word] += 1

			add_node(G, word)

			if last_word is not None:
				add_edge(G, last_word, word)
			
			last_word = word

	# for u, v, d in G.edges(data=True):
	# 	d['weight'] = 1 / d['weight']
	
	return G, word_counter

def read_tychobrahe_file(filename):
	file = open(filename, 'r', encoding="utf-8") 

	G = nx.DiGraph()
	word_counter = Counter()
	last_word = None

	for line in file.readlines() : 
		# Text lines have the format 'word \t syntactic group \t lemma'
		# Other lines indicate sentences, paragraphs, etc. These are ignored
		parsed_line = line.strip().split(" ")

		for elem in parsed_line:
			splitted = elem.split("/")
			if len(splitted) > 1:
				word, pos = splitted[0], splitted[1]

				if pos not in STOP_WORD_GROUPS:
					lemma = word.lower()
					word = lemma
					word_counter[word] += 1

					add_node(G, word)
					add_edge(G, last_word, word)
					
					last_word = word

	return G, word_counter

if __name__ == '__main__':
	G, word_counter = read_colonia_file("data/test2.txt")
	nx.write_gml(G, "data/test2.gml")