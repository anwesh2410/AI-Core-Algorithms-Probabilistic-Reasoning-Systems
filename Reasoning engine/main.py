
import heapq

class Step:
    def __init__(self, formula, rule, source_steps):
        self.formula = formula
        self.rule = rule
        self.source_steps = source_steps

class Node:
    def __init__(self):
        self.steps = []
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

# Helper functions
def is_proposition(c):
    return 'A' <= c <= 'Z'

def remove_spaces(s):
    return s.replace(' ', '')

# 1. Modus Ponens
def modus_ponens(formula, antecedent):
    pos = formula.find(">")
    if pos != -1:
        left = formula[:pos]
        if remove_spaces(left) == antecedent:
            return True
    return False

def get_consequent(formula):
    pos = formula.find(">")
    if pos != -1:
        return remove_spaces(formula[pos + 1:])
    return ""

# 2. Contrapositive
def contrapositive(formula):
    pos = formula.find(">")
    if pos != -1:
        left = formula[:pos]
        right = formula[pos + 1:]
        # Avoid deep nesting of negations
        if '!' in left or '!' in right:
            return ""
        return "!" + remove_spaces(right) + ">" + "!" + remove_spaces(left)
    return ""

# 3. AND Elimination
def and_elimination(formula):
    pos = formula.find("&")
    if pos != -1:
        left = formula[:pos]
        right = formula[pos + 1:]
        return (remove_spaces(left), remove_spaces(right))
    return ("", "")

# 4. AND Introduction
def and_introduction(A, B):
    return f"({A}&{B})"

# 5. OR Introduction
def or_introduction(A, B):
    return f"({A}|{B})"

# 6. Hypothetical Syllogism
def hypothetical_syllogism(formula1, formula2):
    pos1 = formula1.find(">")
    pos2 = formula2.find(">")
    if pos1 != -1 and pos2 != -1:
        left1 = remove_spaces(formula1[:pos1])
        right1 = remove_spaces(formula1[pos1 + 1:])
        left2 = remove_spaces(formula2[:pos2])
        right2 = remove_spaces(formula2[pos2 + 1:])
        if right1 == left2:
            return left1 + ">" + right2
    return ""

# 7. Disjunctive Syllogism
def disjunctive_syllogism(disjunction, negation):
    pos = disjunction.find("|")
    if pos != -1:
        left = remove_spaces(disjunction[:pos])
        right = remove_spaces(disjunction[pos + 1:])
        neg = remove_spaces(negation)
        if neg == "!" + left:
            return right
        if neg == "!" + right:
            return left
    return ""

# 8. DeMorgan's Laws
def de_morgan(formula):
    if len(formula) < 2:
        return ""
    if formula[0] == '!' and "&" in formula:
        pos = formula.find("&")
        left = remove_spaces(formula[1:pos])
        right = remove_spaces(formula[pos + 1:])
        return f"(!{left}|!{right})"
    if formula[0] == '!' and "|" in formula:
        pos = formula.find("|")
        left = remove_spaces(formula[1:pos])
        right = remove_spaces(formula[pos + 1:])
        return f"(!{left}&!{right})"
    return ""

# 9. Double Negation
def double_negation(formula):
    if len(formula) >= 2 and formula[0] == '!' and formula[1] == '!':
        return remove_spaces(formula[2:])
    return ""

# 10. Modus Tollens
def modus_tollens(conditional, negation):
    pos = conditional.find(">")
    if pos != -1:
        left = conditional[:pos]
        right = conditional[pos + 1:]
        if negation == f"!{right}":
            return f"!{left}"
    return ""

# Heuristic function
def heuristic(current_formula, query):
    unmatched = 0
    query_symbols = set(query)
    current_symbols = set(current_formula)
    for symbol in query_symbols:
        if is_proposition(symbol) and symbol not in current_symbols:
            unmatched += 1
    return unmatched

# Reasoning algorithm
def reasoning_algorithm(KB, goal):
    pq = []
    visited = set()

    # Initialize the first node with the KB as premises
    initial = Node()
    for formula in KB:
        initial.steps.append(Step(formula, "[Given premise]", []))
        visited.add(formula)
    initial.g = 0
    initial.h = heuristic(KB[0], goal)
    initial.f = initial.g + initial.h

    heapq.heappush(pq, initial)

    while pq:
        current = heapq.heappop(pq)

        # Check if goal is reached
        for step in current.steps:
            if remove_spaces(step.formula) == goal:
                print("Proof found!")
                print("Steps:")
                for i, step in enumerate(current.steps):
                    print(f"S{i + 1}) {step.formula}\t\t{step.rule}", end="")
                    if step.source_steps:
                        print(" using S" + " and S".join(str(s + 1) for s in step.source_steps), end="")
                    print()
                print(f"Total steps: {len(current.steps)}")
                return

        # Apply inference rules
        for i in range(len(current.steps)):
            # Apply contrapositive (but prevent too many negations)
            new_formula = contrapositive(current.steps[i].formula)
            if new_formula and new_formula not in visited:
                new_node = Node()
                new_node.steps = current.steps.copy()
                new_node.steps.append(Step(new_formula, "[Contrapositive]", [i]))
                new_node.g = current.g + 1
                new_node.h = heuristic(new_formula, goal)
                new_node.f = new_node.g + new_node.h
                heapq.heappush(pq, new_node)
                visited.add(new_formula)

            # Apply AND Elimination
            and_parts = and_elimination(current.steps[i].formula)
            if and_parts[0] and and_parts[1]:
                for part in and_parts:
                    if part and part not in visited:
                        new_node = Node()
                        new_node.steps = current.steps.copy()
                        new_node.steps.append(Step(part, "[AND Elimination]", [i]))
                        new_node.g = current.g + 1
                        new_node.h = heuristic(part, goal)
                        new_node.f = new_node.g + new_node.h
                        heapq.heappush(pq, new_node)
                        visited.add(part)

            # Apply Modus Ponens
            for j in range(len(current.steps)):
                if modus_ponens(current.steps[i].formula, current.steps[j].formula):
                    consequent = get_consequent(current.steps[i].formula)
                    if consequent and consequent not in visited:
                        new_node = Node()
                        new_node.steps = current.steps.copy()
                        new_node.steps.append(Step(consequent, "[Modus Ponens]", [i, j]))
                        new_node.g = current.g + 1
                        new_node.h = heuristic(consequent, goal)
                        new_node.f = new_node.g + new_node.h
                        heapq.heappush(pq, new_node)
                        visited.add(consequent)

    print("Goal cannot be proved from the given knowledge base.")

# Input and Execution
if __name__ == "__main__":
    # Read n and m
    n, m = map(int, input().split())

    # Read the knowledge base
    KB = []
    for _ in range(n):
        KB.append(input().strip())

    # Read the goal (query)
    goal = input().strip()

    print("Knowledge Base:")
    for formula in KB:
        print(formula)
    print("Goal:", goal)
    print("\nStarting proof search...")

    reasoning_algorithm(KB, goal)
