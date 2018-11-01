from os import listdir
import string

# load doc into memory
def load_doc(filename):
	# fix utf-8 error
	file = open(filename, "rb")
	text = file.read()
	text = text.decode("utf-8", errors = "ignore")  # ignore bytes that can't be decoded
	file.close()
	return text

# split a document into news story and highlights
def split_story(doc):
	# find first highlight
	index = doc.find('@highlight')
	# split into story and highlights
	story, highlights = doc[:index], doc[index:].split('@highlight')
	# strip extra white space around each highlight
	highlights = [h.strip() for h in highlights if len(h) > 0]
	return story, highlights

# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare a translation table to remove punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# strip source cnn office if it exists
		
		index = line.find("-LRB- EW.com -RRB-")
		if index > -1:
			line = line[index+len("-LRB- EW.com -RRB-"):]

		index = line.find("-LRB- Wired.com -RRB-")
		if index > -1:
			line = line[index+len("-LRB- Wired.com -RRB-"):]

		index = line.find("-LRB- CNN -RRB- --")
		if index > -1:
			line = line[index+len("-LRB- CNN -RRB- --"):]

		index = line.find("-LRB- CNN -RRB-")
		if index > -1:
			line = line[index+len("-LRB- CNN -RRB-"):]
		
		index = line.find('(CNN) -- ')
		if index > -1:
			line = line[index+len('(CNN)'):]


		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [w.translate(table) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	# remove empty strings
	cleaned = [c for c in cleaned if len(c) > 0]
	return cleaned

# load stories
def separate_data(filename):
	doc = load_doc(filename)
	story, highlights = split_story(doc)
	# store
	story = clean_lines(story.split("\n"))
	highlights = clean_lines(highlights)
