import matplotlib.pyplot as plt
import spacy
from spacy.lang.it import stop_words
from wordcloud import WordCloud

from input_text import input_text


def preprocess(text):
    text = text.replace('\n', ' ')
    nlp = spacy.load('it_core_news_lg')
    doc = nlp(text)
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

    for entity in doc.ents:
        print(entity.text, entity.label_)

    for token in doc:
        print(token, token.lemma, token.lemma_)
    return doc


def get_lemmas(doc):
    return [token.lemma_.lower() for token in doc]


def generate_cloud(tokens):
    wordcloud = WordCloud(stopwords=stop_words.STOP_WORDS).generate(' '.join(tokens))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('vocabulary_cloud_processed.png')


def main(text):
    doc = preprocess(text)
    generate_cloud(get_lemmas(doc))


if __name__ == '__main__':
    s = input_text
    main(s)
