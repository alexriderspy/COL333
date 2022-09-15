import string
class SentenceCorrector(object):
    def __init__(self, cost_fn, conf_matrix):
        self.conf_matrix = conf_matrix
        self.cost_fn = cost_fn

        # You should keep updating following variable with best string so far.
        self.best_state = None  

    def get_corr_chars(self,ch):
        lis = []
        for letter in string.ascii_lowercase:
            for chr in self.conf_matrix[letter]:
                if chr == ch: 
                    lis.append(letter)
                    break
        return lis

    def dfs(self,st,index,min_cost):
        if index == len(st):
            return
        possible_correct_chars = self.get_corr_chars(st[index])
        if(st[index]==' '):
            self.dfs(st,index+1,min_cost)
        for ch in possible_correct_chars:
            prev_ch = st[index]
            st[index] = ch
            cost = self.cost_fn(''.join(st))
            if cost < min_cost:
                self.best_state = ''.join(st)
                min_cost = cost
                self.dfs(st,index+1,min_cost)
            st[index]=prev_ch

    def search(self, start_state):
        """
        :param start_state: str Input string with spelling errors
        """
        # You should keep updating self.best_state with best string so far.
        # self.best_state = start_state
        self.best_state = start_state
        cost = self.cost_fn(start_state)
        lis = list(start_state)
        self.dfs(lis,0,10000)