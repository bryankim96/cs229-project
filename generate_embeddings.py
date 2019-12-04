from datetime import datetime
import gensim
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, split_alphanum, stem_text, \
    strip_multiple_whitespaces, strip_non_alphanum
import pandas as pd

EPOCHS = 10

def apply_preprocessing(s):
    filters = [strip_non_alphanum, strip_multiple_whitespaces, split_alphanum, stem_text, remove_stopwords]
    return preprocess_string(s, filters)


def generate_embeddings(paths, text_col_names, embedding_type='fasttext'):
    if embedding_type == 'fasttext':
        embedding_model = gensim.models.fasttext.FastText()
    elif embedding_type == 'word2vec':
        embedding_model = gensim.models.Word2Vec()
    else:
        raise ValueError("Invalid embedding model type.")

    corpus = []
    for path, text_col_name in zip(paths, text_col_names):
        # Load all training report data
        df = pd.read_csv(path, sep='|')
        texts = df[text_col_name].to_list()

        # Pre-process text
        preprocessed_text = [apply_preprocessing(s) for s in texts]

        # Add to corpus
        for p in preprocessed_text:
            corpus.append(p)

    # Train (FastText) embeddings
    embedding_model.build_vocab(sentences=corpus)
    embedding_model.train(sentences=corpus, total_examples=len(corpus), epochs=EPOCHS)  # train

    return embedding_model


if __name__ == "__main__":

    file_paths = ['../haruka_pathology_reports_111618.csv', '../haruka_radiology_reports_111618.csv']
    file_text_col_names = ['REPORT', 'NOTE']

    model = generate_embeddings(file_paths, file_text_col_names)

    dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")

    # Save resulting model to file for later use
    model.save('embedding_model_{}.mdl'.format(dt_string))
    model.wv.save_word2vec_format('embedding_vecs_{}.w2vec'.format(dt_string))
