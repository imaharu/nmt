from nltk.translate.bleu_score import sentence_bleu
class CalcBlue:
    def __init__(self, target_dict ,val_iter, gold_sentence_file):
        self.target_dict = target_dict
        self.val_iter = val_iter
        self.translate_dict = self.GetTranslateDict(target_dict)
        self.gold_sentence = self.GetGoldSentence(gold_sentence_file)

    def GetGoldSentence(self, gold_sentence_file):
        gold_sentence = []
        with open(gold_sentence_file) as lines:
            for line in lines:
                gold_sentence.append(line.split())
        return gold_sentence

    def GetBlueScore(self, model):
        self.model = model
        model.eval()
        sum_score = 0
        for index, source in enumerate(self.val_iter):
            sentence_ids = model(source=source, phase=1)
            sentence = self.TranslateSentence(sentence_ids)
            print(sentence)
            self.gold_sentence[index]
            sum_score += sentence_bleu(sentence, self.gold_sentence[index] ,weights=(1,))
            break
        return sum_score / 500 

    def GetTranslateDict(self, target_dict):
        translate_dict = {}
        for key, value in target_dict.items():
            translate_dict[value] = key
        return translate_dict

    def TranslateSentence(self, sentence_ids):
        sentence = []
        for sentence_id in sentence_ids:
            sentence.append(self.translate_dict[int(sentence_id)])
        return sentence
