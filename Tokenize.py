import spacy
import re

class tokenize(object):
    
    def __init__(self, lang):
        if lang == "en":
            self.nlp = spacy.load("en_core_web_sm")
        elif lang == "fr":
            self.nlp = spacy.load("fr_core_news_sm")
        elif lang == "zh":
            self.nlp = spacy.load("zh_core_web_sm")
        else:
            self.nlp = spacy.load(lang)
            
    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
