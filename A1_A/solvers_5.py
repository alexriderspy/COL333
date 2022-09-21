import random
import itertools

class SentenceCorrector(object):
    def __init__(self, cost_fn, conf_matrix):
        self.conf_matrix = conf_matrix
        self.cost_fn = cost_fn

        # You should keep updating following variable with best string so far.
        self.best_state = None  
        self.original_state = None

    def initialise(self, start_state):
        words = start_state.split()
        weight = 2
        indices = [0]
        for i in range(len(words)):
            indices.append(indices[i] + len(words[i]) + 1)
        cost = self.cost_fn(start_state)
        sentence = start_state
        mat = self.get_conf_matrix()
        return words, indices, weight, sentence, cost, mat

    #The objective function in A*
    def fn(self, initial_cost, sentence, wt):
        cost = self.cost_fn(sentence)
        f = cost*wt + initial_cost-cost
        return f

    #Substitutes the letter in the sentence
    def get_sentence(self,i, sentence, letter):
        return sentence[:i] + letter + sentence[i+1:]

    def get_letters(self, mat, indices, ch):
        to_substitute = []
        for i in range(ch):
            to_substitute.append([])
            l = mat[self.original_state[indices[i]]]
            for elem in l:
                to_substitute[i].append(elem)
            to_substitute[i].append(self.original_state[indices[i]])
        letters = itertools.product(*to_substitute)
        return letters

    def find_best_letters(self, current_letters, original_sentence ,indices, wt, initial_cost, mat, ch):
        better = current_letters
        min = 10000000
        letters = self.get_letters(mat, indices, ch)
        for vals in letters:
            sentence = original_sentence
            for j in range(ch):
                sentence = self.get_sentence(indices[j], sentence, vals[j])
            f= self.fn(initial_cost, sentence, wt)
            if f<=min:
                better, min = vals, f
        return better , min

    def diff_letter(self, ch, new_letters, word, index):
        cond = False
        for j in range(ch):
            if new_letters[j]!=word[index[j]]:
                cond = True
                break
        return cond

    def stop_loop(self, ch, letters):
        cond = True
        for j in range(ch):
            if letters[0]!='':
                cond = False
                break
        return cond

    def init(self,ch, word, offset):
        test_list = []
        for i in range(len(word)):
            test_list.append(i)
        indices = list(itertools.combinations(test_list, ch))
        min = 1000000000
        letters = []
        positions = []
        letters_to_send = []
        positions_to_send = []
        for i in range(ch):
            letters.append('')
            positions.append(offset)
            letters_to_send.append("")
            positions_to_send.append(0)
        return indices, min, letters, positions, letters_to_send, positions_to_send

    #Finds the best letter to change in a word
    def find_best_word(self, word, offset, ch, weight, cost_init, mat, sentence):
        for j in range(len(word)):
            indices, min, letters, positions, letters_to_send, positions_to_send = self.init(ch, word, offset)
            for index in indices:
                for i in range(ch):
                    letters_to_send[i] = word[index[i]]
                    positions_to_send[i] = offset + index[i]

                new_letters, cost= self.find_best_letters(letters_to_send, sentence ,positions_to_send ,weight, cost_init, mat, ch)
                if self.diff_letter(ch, new_letters, word, index)==True and cost < min:
                    min = cost
                    letters = new_letters
                    positions = [offset + position for position in index]

            if self.stop_loop(ch, letters)==True:
                break 
            else:
                for i in range(ch):
                    sentence = self.get_sentence(positions[i], sentence, letters[i])
        return sentence

    #carries out the series of steps
    def iterative_deepening(self, word, offset, weight, cost_init, sentence, mat, ch):
        for i in range(ch):
            sentence = self.find_best_word(word, offset, i+1, weight, cost_init, mat, sentence)
        return sentence

    #converts the given conf matrix to the semantically opposite key value pairs
    def get_conf_matrix(self):
        mat = {}
        char = 'a'
        for i in range(26):
            mat[char] = []
            char = chr(ord(char) + 1)
        for key, value in self.conf_matrix.items():
            for letter in value:
                mat[letter].append(key)
        return mat

    
    def search(self, start_state):
        words, indices, weight, self.best_state, cost_init, mat = self.initialise(start_state)
        
        self.original_state = start_state
        lis_words = [a for a in range(len(words))]
        
        depth = 1
        while True:
            for j in lis_words:
                #self.best_state = self.iterative_deepening(words[j], indices[j], weight, cost_init, self.best_state, mat, depth)
                self.best_state = self.find_best_word(words[j], indices[j], depth, weight, cost_init, mat, self.best_state)
            depth+=1 