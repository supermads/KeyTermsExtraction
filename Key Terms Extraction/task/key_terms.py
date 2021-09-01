from nltk import word_tokenize, WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords
from lxml import etree
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import flip, max, where

def process_article(article):
    # Tokenize article text, then lemmatize words and remove stop-words and punctuation
    tokens = word_tokenize(article[1].text.lower())
    lemmatizer = WordNetLemmatizer()
    words = []

    for token in tokens:
        word = lemmatizer.lemmatize(token)
        if word not in stopwords.words('english') and word not in punctuation:
            # Only include words that are nouns
            if pos_tag([word])[0][1] == "NN":
                words.append(word)

    words.sort(reverse=True)
    return " ".join(words)


def find_common_words(vectorizer, header, article):
    vector = vectorizer.transform([article])
    vector_array = flip(vector.toarray())[-1]

    feature_names = vectorizer.get_feature_names()
    feature_names.reverse()

    common_words = []
    while len(common_words) < 5:
        max_index = where(vector_array == max(vector_array))[0][0]
        vector_array[max_index] = 0
        common_words.append(feature_names[max_index])

    print(header)
    print(" ".join(common_words))


def main():
    root = etree.parse("news.xml").getroot()
    processed_corpus = []
    article_headers = []

    for corpus in root:
        for article in corpus:
            processed_corpus.append(process_article(article))
            article_headers.append(article[0].text + ":")

    # Find and print five most common words in article text using TfidfVectorizer on all articles
    vectorizer = TfidfVectorizer()
    vectorizer.fit(processed_corpus)

    for header, article in zip(article_headers, processed_corpus):
        find_common_words(vectorizer, header, article)


main()
