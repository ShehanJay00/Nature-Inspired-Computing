import numpy as np
import random

# Problem data
exams = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
timeslots = ['9:00 AM to 11:00 AM', '12:00 PM to 2:00 PM']
exam_halls = ['Exam Hall 1', 'Exam Hall 2', 'Exam Hall 3']
examiners = ['Examiner 1', 'Examiner 2', 'Examiner 3', 'Examiner 4', 'Examiner 5']
students = 100

# Parameters for ACO
num_ants = 10
num_iterations = 100
alpha = 1  # Pheromone importance
beta = 2   # Heuristic importance
evaporation_rate = 0.5
pheromone_constant = 100

epsilon = 1e-6  # Small constant to prevent division by zero

# Initialize pheromone matrix and heuristic information
pheromone = np.ones((len(exams), len(timeslots), len(exam_halls), len(examiners)))
heuristic = np.random.rand(len(exams), len(timeslots), len(exam_halls), len(examiners))

# Helper function to calculate solution cost (lower is better)
def calculate_cost(solution):
    cost = 0
    assigned = set()
    for exam, (timeslot, exam_hall, examiner) in solution.items():
        if (timeslot, exam_hall, examiner) in assigned:
            cost += 10  # Penalize conflicts
        assigned.add((timeslot, exam_hall, examiner))
    return cost

# Ant Colony Optimization algorithm
def ant_colony_optimization():
    global pheromone
    best_solution = None
    best_cost = float('inf')

    for iteration in range(num_iterations):
        solutions = []
        costs = []

        for ant in range(num_ants):
            solution = {}
            for i, exam in enumerate(exams):
                probabilities = []
                for t in range(len(timeslots)):
                    for r in range(len(exam_halls)):
                        for e in range(len(examiners)):
                            prob = (pheromone[i][t][r][e] ** alpha) * (heuristic[i][t][r][e] ** beta)
                            probabilities.append((prob, t, r, e))
                probabilities = sorted(probabilities, key=lambda x: x[0], reverse=True)
                chosen = random.choices(probabilities, weights=[p[0] for p in probabilities], k=1)[0]
                solution[exam] = (timeslots[chosen[1]], exam_halls[chosen[2]], examiners[chosen[3]])

            cost = calculate_cost(solution)
            solutions.append(solution)
            costs.append(cost)

        # Update pheromones
        pheromone *= (1 - evaporation_rate)
        for solution, cost in zip(solutions, costs):
            for exam, (timeslot, exam_hall, examiner) in solution.items():
                i = exams.index(exam)
                t = timeslots.index(timeslot)
                r = exam_halls.index(exam_hall)
                e = examiners.index(examiner)
                pheromone[i][t][r][e] += pheromone_constant / (cost + epsilon)

        # Track the best solution
        min_cost_idx = np.argmin(costs)
        if costs[min_cost_idx] < best_cost:
            best_cost = costs[min_cost_idx]
            best_solution = solutions[min_cost_idx]

    return best_solution

# Run the algorithm
best_timetable = ant_colony_optimization()

# Display the result
def display_timetable(timetable):
    print("Exam\t--Timeslot--\t\t--Exam Hall--\t--Examiner--")
    for exam, (timeslot, exam_hall, examiner) in timetable.items():
        print(f"{exam}\t{timeslot}\t{exam_hall}\t{examiner}")

display_timetable(best_timetable)