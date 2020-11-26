import pdb
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

STOP_WORD_GROUPS = ["PRP", "VIRG", "SENT", "PR", "CONJ", "DET", "QUOTE", "PRP+DET"]
STOP_WORD_LEMMAS = ["<unknown>"]
FILENAMES = [
	"data/abreu1856.txt",
	"data/aires1752.txt",
	"data/alencar1857.txt",
	"data/alencar1865.txt",
	"data/alencar1875.txt",
	"data/almeida1633.txt",
]

def add_node(graph, word):
	if not graph.has_node(word):
		graph.add_node(word)

def add_edge(graph, last_word, word):
	if last_word is not None:
		if graph.has_edge(last_word, word):
			graph[last_word][word]['weight'] += 1
		else:
			graph.add_edge(last_word, word, weight=1)

def read_colonia_file(filename):
	file = open(filename, 'r', encoding="utf-8") 

	G = nx.DiGraph()
	word_counter = Counter()
	last_word = None

	for line in file.readlines() : 
		# Text lines have the format 'word \t syntactic group \t lemma'
		# Other lines indicate sentences, paragraphs, etc. These are ignored
		parsed = line.strip().split("\t")

		if len(parsed) == 3 \
			and parsed[1] not in STOP_WORD_GROUPS \
			and parsed[2] not in STOP_WORD_LEMMAS:

			word = parsed[2].lower()
			word_counter[word] += 1

			add_node(G, word)
			add_edge(G, last_word, word)
			
			last_word = word

	return G, word_counter
	
if __name__ == '__main__':
	for filename in FILENAMES:
		G, word_counter = read_colonia_file(filename)

		print("--------")
		print("filename", filename)
		print("n:", G.number_of_nodes())
		print("m:", G.number_of_edges())
		print("20 most common", word_counter.most_common(20))
		
		# nx.draw(G, with_labels=True)
		# plt.show()
