class SentenceCorrector(object):
    def __init__(self, cost_fn, conf_matrix):
        self.conf_matrix = conf_matrix
        self.cost_fn = cost_fn

        # You should keep updating following variable with best string so far.
        self.best_state = None  

    def search(self, start_state):
        """
        :param start_state: str Input string with spelling errors
        """
        # You should keep updating self.best_state with best string so far.
        # self.best_state = start_state
        cost = self.cost_fn(start_state)
        end_state = 'the company said president and chief executive officer morris moliver had been chief operating officer as well'
        print(cost)
        cost = self.cost_fn(end_state)
        print(cost)
        self.best_state = end_state
