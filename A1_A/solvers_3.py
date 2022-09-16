class SentenceCorrector(object):
    def __init__(self, cost_fn, conf_matrix):
        self.conf_matrix = conf_matrix
        self.cost_fn = cost_fn

        # You should keep updating following variable with best string so far.
        self.best_state = None  

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
        f = (initial_cost-cost) + wt*cost
        return f

    #Substitutes the letter in the sentence
    def get_sentence(self,i, sentence, letter):
        return sentence[:i] + letter + sentence[i+1:]
    
    #Finds the best alternative for a letter at a given index
    def find_best_letter(self, current_letter, original_sentence ,index, wt, initial_cost, mat):
        better = current_letter
        min = 10000000000
        to_substitute = []
        l = mat[current_letter]
        for elem in l:
            to_substitute.append(elem)
        to_substitute.append(current_letter)

        for letter in to_substitute:
            sentence = self.get_sentence(index, original_sentence, letter)
            f= self.fn(initial_cost, sentence, wt)
            if f<=min:
                min, better = f, letter
        return better, min

    #Finds the best letter to change in a word
    def find_best_word(self, word, offset, weight, cost_init, sentence, mat):
        for j in range(len(word)):
            min = 1000000000
            letter = ''
            position = offset
            for i in range(len(word)):
                new_letter, cost= self.find_best_letter(word[i], sentence ,offset+i, weight, cost_init, mat)
                if new_letter!=word[i] and cost < min:
                    min = cost
                    letter = new_letter
                    position = offset + i
            if letter!='':
                sentence = self.get_sentence(position, sentence, letter)
            else:
                break
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
        for j in range(len(words)):
            self.best_state = self.find_best_word(words[j], indices[j], weight, cost_init, self.best_state, mat)
