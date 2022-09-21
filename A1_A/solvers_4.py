import random

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
    
    #Finds the best alternative for a letter at a given index
    def find_best_letter(self, current_letter, original_sentence ,index, wt, initial_cost, mat):
        better = current_letter
        min = 10000000
        to_substitute = []
        original_letter = self.original_state[index]
        l = mat[original_letter]
        for elem in l:
            to_substitute.append(elem)
        to_substitute.append(original_letter)

        for letter in to_substitute:
            sentence = self.get_sentence(index, original_sentence, letter)
            f= self.fn(initial_cost, sentence, wt)
            if f<=min:
                min, better = f, letter
        return better, min

    def find_2_best_letters(self, current_letter1, current_letter2, original_sentence ,index1, index2, wt, initial_cost, mat):
        better1 = current_letter1
        better2 = current_letter2
        min = 10000000
        to_substitute1 = []
        original_letter1 = self.original_state[index1]
        l = mat[original_letter1]
        for elem in l:
            to_substitute1.append(elem)
        to_substitute1.append(original_letter1)

        to_substitute2 = []
        original_letter2 = self.original_state[index2]
        l = mat[original_letter2]
        for elem in l:
            to_substitute2.append(elem)
        to_substitute2.append(original_letter2)

        for letter1 in to_substitute1:
            for letter2 in to_substitute2:
                sentence = self.get_sentence(index1, original_sentence, letter1)
                sentence = self.get_sentence(index2, sentence, letter2)
                f= self.fn(initial_cost, sentence, wt)
                if f<=min:
                    min, better1,better2 = f, letter1,letter2
        return better1, better2, min

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

        for j in range(len(word)):
            min = 1000000000
            letter1 = ''
            letter2 = ''
            position1 = offset
            position2 = offset
            #find 2 letters such that they jointly reduce cost function
            for i in range(len(word)):
                for k in range(i+1,len(word)):
                    new_letter1, new_letter2, cost= self.find_2_best_letters(word[i], word[k], sentence ,offset+i, offset+k,weight, cost_init, mat)
                    if (new_letter1!=word[i] or new_letter2!=word[k]) and cost < min:
                        min = cost
                        letter1 = new_letter1
                        letter2 = new_letter2
                        position1 = offset + i
                        position2 = offset + k
            if letter1=='' and letter2=='':
                break
                
            else:
                sentence = self.get_sentence(position1, sentence, letter1)
                sentence = self.get_sentence(position2,sentence,letter2)
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
        
        for j in lis_words:
            self.best_state = self.find_best_word(words[j], indices[j], weight, cost_init, self.best_state, mat)
        