from collections import deque
import heapq
import random
import time

import config


class Algorithm:
    def __init__(self, heuristic=None):
        self.heuristic = heuristic
        self.nodes_evaluated = 0
        self.nodes_generated = 0

    def get_legal_actions(self, state):
        self.nodes_evaluated += 1
        max_index = len(state)
        zero_tile_ind = state.index(0)
        legal_actions = []
        if 0 <= (up_ind := (zero_tile_ind - config.N)) < max_index:
            legal_actions.append(up_ind)
        if 0 <= (right_ind := (zero_tile_ind + 1)) < max_index and right_ind % config.N:
            legal_actions.append(right_ind)
        if 0 <= (down_ind := (zero_tile_ind + config.N)) < max_index:
            legal_actions.append(down_ind)
        if 0 <= (left_ind := (zero_tile_ind - 1)) < max_index and (left_ind + 1) % config.N:
            legal_actions.append(left_ind)
        return legal_actions

    def apply_action(self, state, action):
        self.nodes_generated += 1
        copy_state = list(state)
        zero_tile_ind = state.index(0)
        copy_state[action], copy_state[zero_tile_ind] = copy_state[zero_tile_ind], copy_state[action]
        return tuple(copy_state)

    def get_steps(self, initial_state, goal_state):
        pass

    def get_solution_steps(self, initial_state, goal_state):
        begin_time = time.time()
        solution_actions = self.get_steps(initial_state, goal_state)
        print(f'Execution time in seconds: {(time.time() - begin_time):.2f} | '
              f'Nodes generated: {self.nodes_generated} | '
              f'Nodes evaluated: {self.nodes_evaluated}')
        return solution_actions


class ExampleAlgorithm(Algorithm):
    def get_steps(self, initial_state, goal_state):
        print("Poziv examplealgo")
        print("init"+str(initial_state))
        print("goal"+str(goal_state))
        state = initial_state
        solution_actions = []
        while state != goal_state:
            legal_actions = self.get_legal_actions(state)
            action = legal_actions[random.randint(0, len(legal_actions) - 1)]
            solution_actions.append(action)
            state = self.apply_action(state, action)
        print("solution ret"+str(solution_actions))
        return solution_actions


class BFS(Algorithm):
    def get_steps(self, initial_state, goal_state):
        # Set za praćenje posećenih stanja
        visited = set()
        # Red koji sadrži parove (stanje, putanja)
        queue = deque([(initial_state, [])])

        # Glavna petlja BFS algoritma
        while queue:
            # Uzimanje trenutnog stanja i putanje iz reda
            current_state, current_path = queue.popleft()
            # Provera da li smo dostigli ciljno stanje
            if current_state == goal_state:
                return current_path

            # Označavanje trenutnog stanja kao posećenog
            visited.add(current_state)
            # Dobijanje legalnih akcija koje se mogu primeniti na trenutnom stanju
            legal_actions = self.get_legal_actions(current_state)
            #print(legal_actions)
            # Iteriranje kroz sve legalne akcije
            for action in legal_actions:
                # Primenjivanje akcije na trenutnom stanju
                new_state = self.apply_action(current_state, action)
                # Provera da li je novo stanje već posećeno, optimizacija (na generisani cvor)
                if new_state not in visited:
                    # Ažuriranje putanje dodavanjem nove akcije
                    new_path = current_path + [action]
                    # Dodavanje novog stanja i putanje u red
                    queue.append((new_state, new_path))
                    # Označavanje novog stanja kao posećenog

        # Ako nema rešenja, vraćamo None
        return None

class BestFirst(Algorithm):
    def __init__(self, heuristic):
        super().__init__(heuristic)

    def get_steps(self, initial_state, goal_state):
        # Skup za praćenje posećenih stanja
        visited = set()
        # Lista čvorova u obliku trojki (vrednost prioriteta, trenutno stanje, putanja do trenutnog stanja)
        node_list = [(self.heuristic.get_evaluation(initial_state), initial_state, [])]
        # Glavna petlja Best-First pretrage
        while node_list:
            # Uzimanje čvora sa početka liste
            #_, current_state, current_path= node_list.pop(0)
            _, current_state, current_path = heapq.heappop(node_list)
            # Provera da li smo dostigli ciljno stanje
            if current_state == goal_state:
                return current_path
            # Označavanje trenutnog stanja kao posećenog
            visited.add(current_state)
            # Dobijanje legalnih akcija koje se mogu primeniti na trenutnom stanju
            legal_actions = self.get_legal_actions(current_state)

            # Lista u koju ćemo dodavati sledeće čvorove
            next_nodes = []

            # Iteriranje kroz sve legalne akcije
            for action in legal_actions:
                # Primenjivanje akcije na trenutnom stanju
                new_state = self.apply_action(current_state, action)
                # Provera da li je novo stanje već posećeno
                if new_state not in visited:
                    # Ažuriranje putanje dodavanjem nove akcije
                    new_path = current_path + [action]
                    # Izračunavanje prioriteta za novo stanje pomoću heuristike
                    priority = self.heuristic.get_evaluation(new_state)
                    # Dodavanje nove trojke u listu
                    heapq.heappush(node_list,(priority, new_state, new_path))

            # Sortiranje liste prema rastućoj vrednosti heuristike čvora
            #node_list.sort(key=lambda node: (node[0],node[1]))
            # Dodavanje sledećih čvorova u listu čvorova
            #node_list.extend(next_nodes)
            #node_list=next_nodes+node_list
        # Ako nema rešenja, vraćamo None
        return None

class Astar(Algorithm):
    def __init__(self,heuristic):
        super().__init__(heuristic)

    def get_steps(self, initial_state, goal_state):
        # Skup za praćenje posećenih stanja
        visited = set()
        # Prioritetni red u obliku trojki (vrednost funkcije proćene, trenutno stanje, putanja do trenutnog stanja)
        priority_queue = [(self.heuristic.get_evaluation(initial_state) + 0, initial_state, [])]

        # Glavna petlja A* algoritma
        while priority_queue:
            # Uzimanje trenutne trojke iz prioriteta
            cost1, current_state, current_path = heapq.heappop(priority_queue)
            # Provera da li smo dostigli ciljno stanje
            if current_state == goal_state:
                return current_path

            # Označavanje trenutnog stanja kao posećenog
            visited.add(current_state)
            # Dobijanje legalnih akcija koje se mogu primeniti na trenutnom stanju
            legal_actions = self.get_legal_actions(current_state)

            # Iteriranje kroz sve legalne akcije
            for action in legal_actions:
                # Primenjivanje akcije na trenutnom stanju
                new_state = self.apply_action(current_state, action)
                # Provera da li je novo stanje već posećeno
                if new_state not in visited:
                    # Ažuriranje putanje dodavanjem nove akcije
                    new_path = current_path + [action]
                    # Izračunavanje funkcije proćene za novo stanje
                    cost =  len(new_path)+self.heuristic.get_evaluation(new_state)
                    # Dodavanje nove trojke u prioritetni red
                    heapq.heappush(priority_queue, (cost, new_state, new_path))


        # Ako nema rešenja, vraćamo None
        return None


class BAB(Algorithm):
    def __init__(self,heuristic):
        super().__init__(heuristic)

    def get_steps(self, initial_state, goal_state):
        # Skup za praćenje posećenih stanja
        visited = set()
        # Prioritetni red u obliku trojki (vrednost funkcije proćene, trenutno stanje, putanja do trenutnog stanja)
        priority_queue = [(0, initial_state, [])]

        # Glavna petlja A* algoritma
        while priority_queue:
            # Uzimanje trenutne trojke iz prioriteta
            _, current_state, current_path = heapq.heappop(priority_queue)
            # Provera da li smo dostigli ciljno stanje
            if current_state == goal_state:
                return current_path

            # Označavanje trenutnog stanja kao posećenog
            visited.add(current_state)
            # Dobijanje legalnih akcija koje se mogu primeniti na trenutnom stanju
            legal_actions = self.get_legal_actions(current_state)

            # Iteriranje kroz sve legalne akcije
            for action in legal_actions:
                # Primenjivanje akcije na trenutnom stanju
                new_state = self.apply_action(current_state, action)
                # Provera da li je novo stanje već posećeno
                if new_state not in visited:
                    # Ažuriranje putanje dodavanjem nove akcije
                    new_path = current_path + [action]
                    # Izračunavanje funkcije proćene za novo stanje
                    cost = len(new_path)
                    # Dodavanje nove trojke u prioritetni red
                    heapq.heappush(priority_queue, (cost, new_state, new_path))

        # Ako nema rešenja, vraćamo None
        return None
'''
class GreedyDepthFirst(Algorithm):
        def __init__(self, heuristic):
            super().__init__(heuristic)

        def get_steps(self, initial_state, goal_state):
            # Skup za praćenje posećenih stanja
            visited = set()
            # StEK za čuvanje stanja koja treba posetiti (LIFO)
            stack = [(self.heuristic.get_evaluation(initial_state), initial_state, visited)]

            # Glavna petlja pohlepne pretrage po dubini
            while stack:
                # Uzimanje trenutne trojke sa stoga
                _, current_state, visited = stack.pop()
                # Provera da li smo dostigli ciljno stanje
                if current_state == goal_state:
                    # Ako jeste, možemo prekinuti pretragu i vratiti putanju
                    return self.reconstruct_path(initial_state, goal_state, visited)

                # Označavanje trenutnog stanja kao posećenog
                visited.add(current_state)
                # Dobijanje legalnih akcija koje se mogu primeniti na trenutnom stanju
                legal_actions = self.get_legal_actions(current_state)

                # Sortiranje legalnih akcija pre dodavanja na stog
                sorted_actions = sorted(legal_actions, key=lambda action: self.heuristic.get_evaluation(
                    self.apply_action(current_state, action)))

                # Iteriranje kroz sve legalne akcije
                for action in sorted_actions:
                    # Primenjivanje akcije na trenutnom stanju
                    new_state = self.apply_action(current_state, action)
                    # Provera da li je novo stanje već posećeno
                    if new_state not in visited:
                        # Dodavanje novog stanja na stog
                        stack.append((self.heuristic.get_evaluation(new_state), new_state, visited))

            # Ako nema rešenja, vraćamo None
            return None

        def reconstruct_path(self, initial_state, goal_state, visited):
            # Rekonstrukcija putanje od ciljnog stanja do početnog
            current_state = goal_state
            path = []
            while current_state != initial_state:
                for action in self.get_legal_actions(current_state):
                    next_state = self.apply_action(current_state, action)
                    if next_state in visited:
                        path.append(action)
                        current_state = next_state
                        break
            # Putanja je rekonstruisana unazad, pa je okrećemo
            return path[::-1]
'''

class GreedyDepthFirst(Algorithm):
    def __init__(self, heuristic):
        super().__init__(heuristic)

    def get_steps(self, initial_state, goal_state):
        # Skup za praćenje posećenih stanja
        visited = set()
        # Stek za čuvanje stanja koja treba posetiti (LIFO)
        stack = [(self.heuristic.get_evaluation(initial_state), initial_state, [])]

        # Glavna petlja pohlepne pretrage po dubini
        while stack:
            # Uzimanje trenutne trojke sa steka
            _, current_state, current_path = stack.pop()
            # Provera da li smo dostigli ciljno stanje
            if current_state == goal_state:
                return current_path

            # Označavanje trenutnog stanja kao posećenog
            visited.add(current_state)
            # Dobijanje legalnih akcija koje se mogu primeniti na trenutnom stanju
            legal_actions = self.get_legal_actions(current_state)

            # Sortiranje legalnih akcija pre dodavanja na stog
            #sorted_actions = sorted(legal_actions, key=lambda action: (self.heuristic.get_evaluation(self.apply_action(current_state, action)), action))
            print("Trenutno stanje je:"+str(current_state)+"njegova deca su.")
            # Iteriranje kroz sve legalne akcije
            for action in legal_actions:
                # Primenjivanje akcije na trenutnom stanju
                new_state = self.apply_action(current_state, action)
                # Provera da li je novo stanje već posećeno

                if new_state not in visited:
                    # Ažuriranje putanje dodavanjem nove akcije
                    new_path = current_path + [action]
                    if new_state==goal_state:
                        return new_path
                    # Dodavanje nove trojke na stog
                    print(str(new_state) + "\n" + "a heuristika je:" +str(self.heuristic.get_evaluation(new_state)))
                    stack.append((self.heuristic.get_evaluation(new_state), new_state, new_path))

        # Ako nema rešenja, vraćamo None
        return None

class BestFirstSearch(Algorithm):
    def __init__(self, heuristic):
        super().__init__(heuristic)

    def get_steps(self, initial_state, goal_state):
        # Skup za praćenje posećenih stanja
        visited = set()
        # Prioritetni red u obliku trojki (vrednost prioriteta, trenutno stanje, putanja do trenutnog stanja)
        priority_queue = [(self.heuristic.get_evaluation(initial_state), initial_state, [],set())]
        i=10000
        # Glavna petlja Best-First pretrage
        while len(priority_queue)!=0:
            #priority_queue = [sublist for sublist in priority_queue if len(sublist) > 0]
            # Uzimanje trenutne trojke iz prioriteta
            _, current_state, current_path,visited = heapq.heappop(priority_queue)


            #priority_queue=[]
            # Provera da li smo dostigli ciljno stanje
            if current_state == goal_state:
                return current_path

            # Označavanje trenutnog stanja kao posećenog
            visited.add(current_state)
            # Dobijanje legalnih akcija koje se mogu primeniti na trenutnom stanju
            legal_actions = self.get_legal_actions(current_state)

            #lista = []
            # Iteriranje kroz sve legalne akcije
            for action in legal_actions:

                # Primenjivanje akcije na trenutnom stanju
                new_state = self.apply_action(current_state, action)
                # Provera da li je novo stanje već posećeno
                if new_state not in visited:
                    # Ažuriranje putanje dodavanjem nove akcije
                    new_path = current_path + [action]
                    # Izračunavanje prioriteta za novo stanje pomoću heuristike
                    priority = self.heuristic.get_evaluation(new_state) #*(len(str(i)))+i
                    # Dodavanje nove trojke u prioritetni red
                    #lista.append((priority, new_state, new_path))
                    # Dodavanje nove trojke u prioritetni red
                    heapq.heappush(priority_queue, (priority, new_state, new_path,visited))
            #i=i-1

        # Ako nema rešenja, vraćamo None
        return None