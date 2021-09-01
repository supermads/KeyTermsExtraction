from nltk import word_tokenize, WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords
from lxml import etree
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer


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
                    # Only include nouns
                    if pos_tag([word])[0][1] == "NN":
                        words.append(word)

            # Find and print five most common words in article text using TF-IDF metric for each word in all articles
            words.sort(reverse=True)
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([" ".join(words)])
            terms = vectorizer.get_feature_names()

            ratings = tfidf_matrix.toarray()[0]

            common_word_indices = (-ratings).argsort()[:5]

            common_words = []
            for index in common_word_indices:
                common_words.append(terms[index])

            print(" ".join([word for word in common_words]))


main()
