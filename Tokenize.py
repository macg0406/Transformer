import sentencepiece as spm
import re

class SPMTokenize(object):
    
    def __init__(self, model_file=None, lang=None):
        if model_file:
            self.spm = spm.SentencePieceProcessor(model_file=model_file)
        elif lang == "en":
            self.spm = spm.SentencePieceProcessor(model_file="data/spm_en.model")
            # self.nlp = spacy.load("en_core_web_sm")
        # elif lang == "fr":
        #     self.nlp = spacy.load("fr_core_news_sm")
        elif lang == "zh":
            self.spm = spm.SentencePieceProcessor(model_file="data/spm_zh.model")
            # self.nlp = spacy.load("zh_core_web_sm")
        else:
            # self.nlp = spacy.load(lang)
            self.spm = spm.SentencePieceProcessor(model_file="data/spm_en.model")
        self.bos = self.spm.bos_id()
        self.eos = self.spm.eos_id()
        self.unk = self.spm.unk_id()
        self.pad_id = self.spm.pad_id()
        self.vocab_size = self.spm.vocab_size()
        self.origin_vocab_size = self.vocab_size
        if self.bos < 0:
            self.bos = self.vocab_size
            self.vocab_size += 1

        if self.eos<0:
            self.eos = self.vocab_size
            self.vocab_size += 1
        
        if self.unk < 0:
            self.unk = self.vocab_size
            self.vocab_size += 1
        
        if self.pad_id < 0:
            self.pad_id = self.vocab_size
            self.vocab_size += 1
            
    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return self.spm.encode(sentence, out_type=str)
        # return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]

    def convert_sent_to_ids(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [self.bos] + self.spm.encode(sentence) + [self.eos]
