import random
import nltk
nltk.download('averaged_perceptron_tagger')
# Load the text file
with open('C:/Users/refij/OneDrive/Dokumenti/GitHub/NLP/documents/text_1.txt', 'r', encoding='utf-8') as file:
    text = file.read() 
#remove non-alphabetic and non-whitespace characters
clean_text = text.lower()
clean_text = ''.join(filter(lambda c: c.isalpha() or c.isspace(), clean_text))
#print(clean_text)

# Tokenize the text
tokens = nltk.word_tokenize(clean_text)

# Set the value of n for n-grams
n = 5

# Create n-grams
ngrams = list(nltk.ngrams(tokens, n))

# Create a dictionary of n-gram frequencies
freq_dict = {}
for gram in ngrams:
    key = ' '.join(gram[:-1])
    if key not in freq_dict:
        freq_dict[key] = []
    freq_dict[key].append(gram[-1])

# Generate 50 sentences
filename = f"generatedtext.txt" 
with open(filename, 'w', encoding='utf-8') as output_file:
#with open('C:/Users/refij/OneDrive/Dokumenti/GitHub/NLP/documents/generated_text.txt', 'w', encoding='utf-8') as output_file:
    for i in range(50):
        generated_text = ''
        start = ' '.join(random.choice(ngrams)[:-1])
        for i in range(13):
            if start not in freq_dict:
                break
            next_word = random.choice(freq_dict[start])
            generated_text += ' ' + next_word
            start = ' '.join(start.split()[1:]) + ' ' + next_word 
        print(generated_text.capitalize() + '.')
        tagged_words = nltk.pos_tag(nltk.word_tokenize(generated_text))
        capitalized_sentence = ""
        for i, (word, tag) in enumerate(tagged_words):
            if i == 0 or tag.startswith('N'):
                capitalized_sentence += word.capitalize() + ''
        output_file.write(generated_text.capitalize() + '.' + '\n')
   
