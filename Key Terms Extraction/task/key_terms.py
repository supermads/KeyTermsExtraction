import nltk
from lxml import etree
from collections import Counter


def main():
    root = etree.parse("news.xml").getroot()

    for corpus in root:
        for article in corpus:
            # Print article header
            print(article[0].text + ":")

            # Create freq. dict. for tokens in the article text and find the 5 most common tokens
            tokens = nltk.word_tokenize(article[1].text.lower())
            tokens.sort(reverse=True)
            common_words = Counter(tokens).most_common(5)

            print(" ".join([word[0] for word in common_words]) + "\n")


main()
