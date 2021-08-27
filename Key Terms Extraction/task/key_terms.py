from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from lxml import etree
from collections import Counter
from string import punctuation


def main():
    root = etree.parse("news.xml").getroot()

    for corpus in root:
        for article in corpus:
            # Print article header
            print(article[0].text + ":")

            # Tokenize article text, then lemmatize words and remove stop-words and punctuation
            tokens = word_tokenize(article[1].text.lower())

            lemmatizer = WordNetLemmatizer()
            words = []
            for token in tokens:
                word = lemmatizer.lemmatize(token)
                if word not in stopwords.words('english') and word not in punctuation:
                    words.append(word)

            # Find and print five most common words in article text
            words.sort(reverse=True)
            common_words = Counter(words).most_common(5)

            print(" ".join([word[0] for word in common_words]) + "\n")


main()
