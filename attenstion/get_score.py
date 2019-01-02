from nltk import bleu_score
class CalcBlue:
    def __init__(self, val_iter, gold_sentence_file):
        self.val_iter = val_iter
        self.filename = gold_sentence_file

    def GetBlueScore(self, model):
        self.model = model

        model.eval()

        for source in self.val_iter:
            model(source=source, phase=1)
        return 0

    def TranslateSentence(self, sentence):
        return 0
