class Heuristic:
    def get_evaluation(self, state):
        pass


class ExampleHeuristic(Heuristic):
    def get_evaluation(self, state):
        return 0

class Hamming(Heuristic):
    def get_evaluation(self,state):
        broj_neodgovarajucih=0
        i=1
        for s in state:
            if int(s)!=i and int(s)!=0:
                broj_neodgovarajucih=broj_neodgovarajucih+1
            i=i+1
        return broj_neodgovarajucih
'''
class Manhattan(Heuristic):
    def get_evaluation(self,state):
        return sum(list( abs(int(s)-i-1) if int(s)!=0 else 0 for i,s in enumerate(state)))
'''

class Manhattan(Heuristic):
    def get_evaluation(self,state):
        n=len(state)**0.5
        return sum(list( abs((s-1)%n-i%n) + abs((s-1)//n-i//n) if int(s)!=0 else 0 for i,s in enumerate(state)))
