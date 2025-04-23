from collections import Counter, defaultdict
from typing import Dict

import matplotlib.pyplot as plt
from langdetect import detect, detect_langs, DetectorFactory
from spacy.lang.it import stop_words
from wordcloud import WordCloud

from input_text import input_text


def get_langs(text):
    tokens = split_text(text)
    lines = split_text(text, '\n')[1:-1]
    # print(f'words: {len(tokens)} ({tokens[0]})')
    # print(f'lines: {len(lines)} ({lines[0]})')

    one_lang = detect(text)  # best
    print(one_lang)
    print()
    langs_by_word_cnts, langs_by_word_perc = predict_langs(tokens)  # bad with this algorithm
    plt.pie([t[1] for t in langs_by_word_perc], labels=[t[0] for t in langs_by_word_perc])
    plt.savefig('langs_by_word_percentages.png')
    print(langs_by_word_cnts)
    print(langs_by_word_perc)
    print()
    _, langs_by_line_perc = predict_langs(lines)  # unpredictable
    print(langs_by_line_perc)
    return one_lang, langs_by_word_perc, langs_by_line_perc


def split_text(text, by_char=' '):
    return text.split(by_char)


def predict_langs(text_chunks):
    predicted = defaultdict(int)
    for chunk in text_chunks:
        langs_list = detect_langs(chunk)
        langs = {lang.lang: lang.prob for lang in langs_list}
        for locale in langs:
            predicted[locale] += 1
    predicted_cnts = sorted(predicted.items(), key=lambda t: t[1], reverse=True)
    predicted_percentages = sorted(convert_lang_percentage(predicted).items(), key=lambda t: t[1], reverse=True)
    return predicted_cnts, predicted_percentages


def convert_lang_percentage(langs: Dict[str, int]) -> Dict[str, int]:
    total = sum(v for v in langs.values())
    return {lang: round(cnt / total, 2) for lang, cnt in langs.items()}


def get_vocabulary(tokens):
    vocabulary = Counter(token.lower() for token in tokens)
    vocabulary_no_stop = remove_stops(vocabulary)
    print(f'Total n. of words:        {len(tokens)}')
    print(f'Unique n. of words:       {len(vocabulary)}')
    print(f'UniqueNoStop n. of words: {len(vocabulary_no_stop)}')
    print(vocabulary)


def generate_cloud(tokens):
    wordcloud = WordCloud(stopwords=stop_words.STOP_WORDS).generate(' '.join(tokens))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('vocabulary_cloud.png')


def remove_stops(vocabulary):
    stop_it = stop_words.STOP_WORDS
    vocabulary_no_stop = {k: v for k, v in vocabulary.items() if k not in stop_it}
    return sorted(vocabulary_no_stop.items(), key=lambda t: t[1], reverse=True)


def main(text):
    get_langs(text)
    tokens = text.split()
    get_vocabulary(tokens)
    generate_cloud(tokens)


if __name__ == '__main__':
    DetectorFactory.seed = 0
    s = input_text
    main(s)
