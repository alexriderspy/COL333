import string,random

class SentenceCorrector(object):
    def __init__(self, cost_fn, conf_matrix):
        self.conf_matrix = conf_matrix
        self.cost_fn = cost_fn
        # You should keep updating following variable with best string so far.
        self.best_state = None  
        self.check = {}
        for letter in string.ascii_lowercase:
            self.check[letter] = -1
        self.BIG_WORD = 8

    def get_corr_chars(self,ch):
        lis = []
        for letter in string.ascii_lowercase:
            for chr in self.conf_matrix[letter]:
                if chr == ch: 
                    lis.append(letter)
                    break
        return lis

    def partition(self,state):
        #at most 1 letter change
        stateList = list(state)
        min_cost = self.cost_fn(state)
        store = state
        for j in range(len(stateList)):
            if stateList[j] == ' ':
                continue
            if self.check[stateList[j]] !=-1:
                possible_correct_chars = self.check[stateList[j]]
            else:
                possible_correct_chars = self.get_corr_chars(stateList[j])
                self.check[stateList[j]] = possible_correct_chars
            init_ch = stateList[j]
            for i in range(len(possible_correct_chars)):
                stateList[j] = possible_correct_chars[i]
                cost = self.cost_fn(''.join(stateList))
                if cost <= min_cost:
                    min_cost = cost
                    store = ''.join(stateList)
            stateList[j]=init_ch
        return store

    def spell_check_large_words(self,state):
        stateList = state.split(' ')
        
        min_cost = self.cost_fn(state)

        # for i in range(len(stateList)):
        #     if len(stateList[i]) >= self.BIG_WORD:
        #         stateList[i] = self.partition(stateList[i])

        state = ' '.join(stateList)
        cost = self.cost_fn(state)

        if cost <= min_cost:
            self.best_state = state
            min_cost = cost
        
        lis = [i for i in range(1,len(stateList))]
        random.shuffle(lis)
        for l in lis:
            tmp_lis = [i for i in range(len(stateList)-l)]
            random.shuffle(tmp_lis)        
            for i in tmp_lis:
                slc = stateList[i:i+l+1]
                state = ' '.join(slc)
                state = self.partition(state)
                lis = state.split(' ')
                for li in range(len(lis)):
                    stateList[i+li] = lis[li]
            state = ' '.join(stateList)
            cost = self.cost_fn(state)

            if cost <= min_cost:
                self.best_state = state
                min_cost = cost
        
        print(min_cost)
        
    
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