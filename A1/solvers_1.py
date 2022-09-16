# stochastic hill climbing

from os import stat
import string
class SentenceCorrector(object):
    def __init__(self, cost_fn, conf_matrix):
        self.conf_matrix = conf_matrix
        self.cost_fn = cost_fn
        # You should keep updating following variable with best string so far.
        self.best_state = None  
        self.check = {}
        for letter in string.ascii_lowercase:
            self.check[letter] = -1
        self.BIG_WORD = 6

    def get_corr_chars(self,ch):
        lis = []
        for letter in string.ascii_lowercase:
            for chr in self.conf_matrix[letter]:
                if chr == ch: 
                    lis.append(letter)
                    break
        return lis

    def one_word(self,state):
        #at most 1 letter in a word
        stateList = list(state)
        init_cost = self.cost_fn(state)
        for j in range(len(stateList)):
            if self.check[stateList[j]] !=-1:
                possible_correct_chars = self.check[stateList[j]]
            else:
                possible_correct_chars = self.get_corr_chars(stateList[j])
                self.check[stateList[j]] = possible_correct_chars
            init_ch = stateList[j]
            for i in range(len(possible_correct_chars)):
                stateList[j] = possible_correct_chars[i]
                cost = self.cost_fn(''.join(stateList))
                if cost <= init_cost:
                    return ''.join(stateList)
            stateList[j]=init_ch
        return state
    
    def spell_check_large_words(self,state):
        stateList = state.split(' ')
        for i in range(len(stateList)):
            if len(stateList[i]) >= self.BIG_WORD:
                stateList[i] = self.one_word(stateList[i])
        
        print(self.cost_fn(''.join(stateList)))

    def search(self, start_state):
        """
        :param start_state: str Input string with spelling errors
        """
        # You should keep updating self.best_state with best string so far.
        # self.best_state = start_state
        self.best_state = start_state
        cost = self.cost_fn(start_state)
        print(cost)
        self.spell_check_large_words(start_state)