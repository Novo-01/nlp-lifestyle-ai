import spacy

nlp = spacy.load("ja_ginza")

def extract_keywords(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
