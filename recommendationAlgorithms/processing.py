import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

stopword = stopwords.words('english')


def get_wordnet_pos(tag):

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
    return 0

    
def preprocessing(text):
    # lower case
    text = text.lower()

    # remove punctuation
    text_rp = "".join([char for char in text if char not in string.punctuation])

    # word tokenization 
    tokens = word_tokenize(text_rp)

    # remove stopwords  
    
    tokens_without_stopwords = [word for word in tokens if word not in stopword]

    # lemm
    tagged_tokens = nltk.pos_tag(tokens_without_stopwords)
    #print(tagged_tokens)
    tokens_processed = []
    
    lemmatizer = WordNetLemmatizer()
    for word, tag in tagged_tokens:
        word_net_tag = get_wordnet_pos(tag)
        if word_net_tag != '':
            tokens_processed.append(lemmatizer.lemmatize(word, word_net_tag))
        else:
            tokens_processed.append(word)
    text_processed = ' '.join(tokens_processed)

    return text_processed

