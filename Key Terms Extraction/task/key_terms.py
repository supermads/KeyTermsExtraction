import nltk
from lxml import etree
from collections import Counter


def main():
    root = etree.parse("news.xml").getroot()

    for corpus in root:
        for article in corpus:
            # Print article header
            print(article[0].text + ":")

            # Create freq. dict. for tokens in the article text and find most common words
            tokens = nltk.word_tokenize(article[1].text.lower())
            tokens.sort(reverse=True)
            freq_dict = Counter(tokens)
            common_words = freq_dict.most_common()[0:5]

            for key, value in freq_dict.most_common()[5:]:
                if value == common_words[4]:
                    common_words.append((key, value))

            print(" ".join([word[0] for word in common_words]) + "\n")


main()
