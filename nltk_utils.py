import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize a PorterStemmer object for stemming operations
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Tokenizes a sentence into individual words.
    
    Args:
        sentence (str): The sentence to be tokenized.
        
    Returns:
        list: A list of tokens (words and punctuation) from the sentence.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stems a word to its root form.
    
    Args:
        word (str): The word to be stemmed.
        
    Returns:
        str: The stemmed version of the word.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Creates a bag-of-words vector for a tokenized sentence, indicating the presence or absence of words.
    
    Args:
        tokenized_sentence (list of str): A list of words from the tokenized sentence.
        words (list of str): A list of all words considered in the model, used to build the vector.
        
    Returns:
        numpy.ndarray: A vector of shape (len(words),), where each element is 1 if the word is present in the sentence, 0 otherwise.
    """
    # Stem each word in the tokenized sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Create the bag-of-words vector by checking the presence of each word in the sentence
    bag = np.array([1 if word in sentence_words else 0 for word in words], dtype=np.float32)
    return bag
