
import math

start_token = '<S>'
end_token = '<E>'


class LanguageModel(object):
    def __init__(
        self, unigram_counts, total_unigrams,
        bigram_counts, trigram_counts, unk_prob,
        ignore_sentinels=False, mode='spell_check'
    ):
        self.unigram_counts = unigram_counts
        self.bigram_counts = bigram_counts
        self.trigram_counts = trigram_counts
        self.unk_prob = unk_prob
        self.total_unigrams = total_unigrams
        self.ignore_sentinels = ignore_sentinels
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode

    def unigram_score(self, token):
        if token == end_token:
            # Ignore End Token
            return 0

        cnt = self.unigram_counts.get(token, 0)
        if cnt == 0:
            if self.mode == 'spell_check':
                return math.log10(self.unk_prob)
            else:
                return  math.log(10) - math.log(self.total_unigrams * 10 ** len(token))

        return math.log10(cnt - self.unk_prob) - math.log10(self.total_unigrams)

    def bigram_score(self, ftoken, stoken):
        bicnt = self.bigram_counts.get((ftoken, stoken), 0)

        if bicnt == 0:
            biprob = self.unigram_score(stoken)
        else:
            denom = self.unigram_counts[ftoken]
            assert denom > 0, 'Unigram count cannot be zero'
            biprob = math.log10(bicnt) - math.log10(denom)

        return biprob

    def trigram_score(self, ftoken, stoken, ttoken):
        tricnt = self.trigram_counts.get((ftoken, stoken, ttoken), 0)

        if tricnt == 0:
            triprob = self.bigram_score(stoken, ttoken)
        else:
            denom = self.bigram_counts[(ftoken, stoken)]
            assert denom > 0, 'Bigram count cannot be zero'
            triprob = math.log10(tricnt) - math.log10(denom)

        return triprob

    def score(self, sentence):
        if self.ignore_sentinels:
            tokens = sentence.split()
        else:
            tokens = [start_token] + sentence.split() + [end_token]

        if len(tokens) == 1:
            return -1.0 * self.unigram_score(tokens[0])

        score = 0.0
        for ii in range(1, len(tokens), 1):
            if ii == 1:
                score += self.bigram_score(tokens[ii - 1], tokens[ii])
                continue

            if self.trigram_counts is None:
                score += self.bigram_score(tokens[ii - 1], tokens[ii])
            else:
                score += self.trigram_score(tokens[ii - 2], tokens[ii - 1], tokens[ii])

        return -1.0 * score

    def __call__(self, sentence):
        if self.mode == 'spell_check':
            return self.score(sentence)
        else:
            return self.score(sentence)
