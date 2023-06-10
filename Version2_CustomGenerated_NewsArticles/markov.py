import os
import random
import string

class MarkovChain:
    def __init__(self, order):
        self.order = order
        self.transition_table = {}
        self.start_words = []

    def train(self, corpus):
        for text in corpus:
            sentences = text.split('. ')
            for sentence in sentences:
                words = sentence.strip().split()
                if len(words) >= self.order:
                    start_sequence = tuple(words[:self.order])
                    self.start_words.append(start_sequence)

                    for i in range(len(words) - self.order):
                        current_sequence = tuple(words[i:i+self.order])
                        next_word = words[i+self.order]

                        if current_sequence not in self.transition_table:
                            self.transition_table[current_sequence] = {}

                        if next_word not in self.transition_table[current_sequence]:
                            self.transition_table[current_sequence][next_word] = 0

                        self.transition_table[current_sequence][next_word] += 1

    def generate_sentence(self):
        start_sequence = random.choice(self.start_words)
        current_sequence = list(start_sequence)
        sentence = ' '.join(current_sequence) + ' '

        while True:
            if tuple(current_sequence) not in self.transition_table:
                break

            next_word = self.choose_next_word(current_sequence)
            sentence += next_word + ' '

            if next_word in string.punctuation:
                break

            if len(current_sequence) == self.order:
                current_sequence.pop(0)
            current_sequence.append(next_word)

        return sentence.strip()

    def choose_next_word(self, sequence):
        word_counts = self.transition_table[tuple(sequence)]
        total_count = sum(word_counts.values())
        probabilities = {word: count / total_count for word, count in word_counts.items()}

        # Apply smoothing method (e.g., add-k smoothing) to handle unseen sequences
        smoothing_factor = 0.1
        vocabulary_size = len(probabilities)
        unseen_probability = smoothing_factor / (total_count + smoothing_factor * vocabulary_size)
        probabilities = {word: prob + unseen_probability for word, prob in probabilities.items()}

        # Choose next word based on probabilities
        rand = random.random()
        cumulative_probability = 0
        for word, prob in probabilities.items():
            cumulative_probability += prob
            if rand <= cumulative_probability:
                return word

        # If for some reason a word is not chosen, return a random word
        return random.choice(list(probabilities.keys()))


def load_text_files(folder_path):
    texts = []
    for i in range(1, 61):
        file_path = os.path.join(folder_path, f'a{i}.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)
    return texts


def generate_news_articles(num_articles, corpus):
    markov_chain = MarkovChain(order=2)
    markov_chain.train(corpus)

    for i in range(num_articles):
        article = []
        for _ in range(random.randint(50, 100)):
            sentence = markov_chain.generate_sentence()
            article.append(sentence)

        article_text = '. '.join(article)
        with open(f'generated_article_{i+1}.txt', 'w', encoding='utf-8') as f:
            f.write(article_text)


# Load text files
documents_folder = 'HumanGenerated_NewsArticles'
corpus = load_text_files(documents_folder)

# Generate news articles
num_articles = 10
generate_news_articles(num_articles, corpus)
