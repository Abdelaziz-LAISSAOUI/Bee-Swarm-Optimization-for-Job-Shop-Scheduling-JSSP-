"""
data = [
 [25, 75, 75, 76, 38, 62, 38, 59, 14, 13, 46, 31, 57, 92,  3,],
 [67,  5, 11, 11, 40, 34, 77, 42, 35, 96, 22, 55, 21, 29, 16,],
 [22, 98,  8, 35, 59, 31, 13, 46, 52, 22, 18, 19, 64, 29, 70,],
 [99, 42,  2, 35, 11, 92, 88, 97, 21, 56, 17, 43, 27, 19, 23,],
 [50,  5, 59, 71, 47, 39, 82, 35, 12,  2, 39, 42, 52, 65, 35,],
 [48, 57,  5,  2, 60, 64, 86,  3, 51, 26, 34, 39, 45, 63, 54,],
 [40, 43, 50, 71, 46, 99, 67, 34,  6, 95, 67, 54, 29, 30, 60,],
 [59,  3, 85,  6, 46, 49,  5, 82, 18, 71, 48, 79, 62, 65, 76,],
 [65, 55, 81, 15, 32, 52, 97, 69, 82, 89, 69, 87, 22, 71, 63,],
 [70, 74, 52, 94, 14, 81, 24, 14, 32, 39, 67, 59, 18, 77, 50,],
 [18,  6, 96, 53, 35, 99, 39, 18, 14, 90, 64, 81, 89, 48, 80,],
 [44, 75, 12, 13, 74, 59, 71, 75, 30, 93, 26, 30, 84, 91, 93,],
 [39, 56, 13, 29, 55, 69, 26,  7, 55, 48, 22, 46, 50, 96, 17,],
 [57, 14,  8, 13, 95, 53, 78, 24, 92, 90, 68, 87, 43, 75, 94,],
 [93, 92, 18, 28, 27, 40, 56, 83, 51, 15, 97, 48, 53, 78, 39,],
 [47, 34, 42, 28, 11, 11, 30, 14, 10,  4, 20, 92, 19, 59, 28,],
 [69, 82, 64, 40, 27, 82, 27, 43, 56, 17, 18, 20, 98, 43, 68,],
 [84, 26, 87, 61, 95, 23, 88, 89, 49, 84, 12, 51,  3, 44, 20,],
 [43, 54, 18, 72, 70, 28, 20, 22, 59, 36, 85, 13, 73, 29, 45,],
 [ 7, 97,  4, 22, 74, 45, 62, 95, 66, 14, 40, 23, 79, 34,  8,],
]

machines = [
 [ 4, 12, 15,  2,  11,  3,  5,  8,  1, 13,  6, 10,   7,  14,   9,], 
 [ 6,  1,  4,  9,   5,  2, 13, 15,  7,  8, 11,  3,  10,  14,  12,], 
 [ 3,  4, 15,  1,  10, 13,  6,  5,  8, 11,  9, 12,  14,   2,   7,], 
 [ 9, 11,  2, 14,   4,  5, 15, 10,  3,  6, 12,  8,   1,   7,  13,], 
 [15,  9,  2,  3,  11, 10, 13,  5,  7,  6,  1, 14,   4,  12,   8,], 
 [ 4, 11,  2,  6,   7,  1,  9,  8, 12, 14,  3, 15,  13,  10,   5,], 
 [ 3, 11,  2, 13,   9,  1,  8,  7, 15, 14,  5,  4,   6,  10,  12,], 
 [ 2,  1,  3,  5,   8, 14, 12,  4, 13,  6,  7, 15,  10,   9,  11,], 
 [ 5,  6, 10, 11,   8,  7,  3,  2, 13,  4, 14,  1,   9,  15,  12,], 
 [ 2,  5,  4, 11,  15,  1,  7, 14, 12,  9,  6, 13,   8,  10,   3,], 
 [ 4, 11,  2,  1,  10,  9, 15,  7,  5,  8,  3, 13,   6,  12,  14,], 
 [ 3,  8,  7,  9,   4,  6, 15,  5,  2,  1, 10, 11,  14,  12,  13,], 
 [ 1,  8, 15,  9,  13, 11, 10,  4,  7,  2,  5,  3,  12,  14,   6,], 
 [13,  4, 10,  5,   2,  1, 11,  7,  6,  3, 15, 14,   8,   9,  12,], 
 [ 4, 15,  7,  6,  14, 10,  2,  1, 13,  8,  3,  5,  11,   9,  12,], 
 [ 6, 15,  7, 13,   9,  3,  5, 10, 12, 14,  4,  2,   8,   1,  11,], 
 [ 4,  8, 11, 15,   1,  9,  2, 12,  6, 14,  5, 13,   7,  10,   3,], 
 [11,  9,  3, 12,  14,  7, 15,  4, 10,  8,  5,  6,  13,   1,   2,], 
 [ 4,  3, 13, 14,   2,  7, 15,  6,  5,  9, 10, 12,   1,  11,   8,], 
 [12, 15,  6,  7,  11, 10, 14,  2,  5,  9,  1,  4,  13,   3,   8,],     
]
"""

jobs =[
 [(4, 25), (12, 75), (15, 75), (2, 76), (11, 38), (3, 62), (5, 38), (8, 59), (1, 14), (13, 13), (6, 46), (10, 31), (7, 57), (14, 92), (9, 3)],
 [(6, 67), (1, 5), (4, 11), (9, 11), (5, 40), (2, 34), (13, 77), (15, 42), (7, 35), (8, 96), (11, 22), (3, 55), (10, 21), (14, 29), (12, 16)],
 [(3, 22), (4, 98), (15, 8), (1, 35), (10, 59), (13, 31), (6, 13), (5, 46), (8, 52), (11, 22), (9, 18), (12, 19), (14, 64), (2, 29), (7, 70)],
 [(9, 99), (11, 42), (2, 2), (14, 35), (4, 11), (5, 92), (15, 88), (10, 97), (3, 21), (6, 56), (12, 17), (8, 43), (1, 27), (7, 19), (13, 23)],
 [(15, 50), (9, 5), (2, 59), (3, 71), (11, 47), (10, 39), (13, 82), (5, 35), (7, 12), (6, 2), (1, 39), (14, 42), (4, 52), (12, 65), (8, 35)],
 [(4, 48), (11, 57), (2, 5), (6, 2), (7, 60), (1, 64), (9, 86), (8, 3), (12, 51), (14, 26), (3, 34), (15, 39), (13, 45), (10, 63), (5, 54)],
 [(3, 40), (11, 43), (2, 50), (13, 71), (9, 46), (1, 99), (8, 67), (7, 34), (15, 6), (14, 95), (5, 67), (4, 54), (6, 29), (10, 30), (12, 60)],
 [(2, 59), (1, 3), (3, 85), (5, 6), (8, 46), (14, 49), (12, 5), (4, 82), (13, 18), (6, 71), (7, 48), (15, 79), (10, 62), (9, 65), (11, 76)],
 [(5, 65), (6, 55), (10, 81), (11, 15), (8, 32), (7, 52), (3, 97), (2, 69), (13, 82), (4, 89), (14, 69), (1, 87), (9, 22), (15, 71), (12, 63)],
 [(2, 70), (5, 74), (4, 52), (11, 94), (15, 14), (1, 81), (7, 24), (14, 14), (12, 32), (9, 39), (6, 67), (13, 59), (8, 18), (10, 77), (3, 50)],
 [(4, 18), (11, 6), (2, 96), (1, 53), (10, 35), (9, 99), (15, 39), (7, 18), (5, 14), (8, 90), (3, 64), (13, 81), (6, 89), (12, 48), (14, 80)],
 [(3, 44), (8, 75), (7, 12), (9, 13), (4, 74), (6, 59), (15, 71), (5, 75), (2, 30), (1, 93), (10, 26), (11, 30), (14, 84), (12, 91), (13, 93)],
 [(1, 39), (8, 56), (15, 13), (9, 29), (13, 55), (11, 69), (10, 26), (4, 7), (7, 55), (2, 48), (5, 22), (3, 46), (12, 50), (14, 96), (6, 17)],
 [(13, 57), (4, 14), (10, 8), (5, 13), (2, 95), (1, 53), (11, 78), (7, 24), (6, 92), (3, 90), (15, 68), (14, 87), (8, 43), (9, 75), (12, 94)],
 [(4, 93), (15, 92), (7, 18), (6, 28), (14, 27), (10, 40), (2, 56), (1, 83), (13, 51), (8, 15), (3, 97), (5, 48), (11, 53), (9, 78), (12, 39)],
 [(6, 47), (15, 34), (7, 42), (13, 28), (9, 11), (3, 11), (5, 30), (10, 14), (12, 10), (14, 4), (4, 20), (2, 92), (8, 19), (1, 59), (11, 28)],
 [(4, 69), (8, 82), (11, 64), (15, 40), (1, 27), (9, 82), (2, 27), (12, 43), (6, 56), (14, 17), (5, 18), (13, 20), (7, 98), (10, 43), (3, 68)],
 [(11, 84), (9, 26), (3, 87), (12, 61), (14, 95), (7, 23), (15, 88), (4, 89), (10, 49), (8, 84), (5, 12), (6, 51), (13, 3), (1, 44), (2, 20)],
 [(4, 43), (3, 54), (13, 18), (14, 72), (2, 70), (7, 28), (15, 20), (6, 22), (5, 59), (9, 36), (10, 85), (12, 13), (1, 73), (11, 29), (8, 45)],
 [(12, 7), (15, 97), (6, 4), (7, 22), (11, 74), (10, 45), (14, 62), (2, 95), (5, 66), (9, 14), (1, 40), (4, 23), (13, 79), (3, 34), (8, 8)]
 ]



import random
import math
import copy

# Flatten operations with job and op index
def get_all_operations(jobs):
    all_ops = []
    for job_id, ops in enumerate(jobs):
        for op_id, (machine, duration) in enumerate(ops):
            all_ops.append((job_id, op_id, machine, duration))
    return all_ops

# Generate a random operation sequence (with correct counts)
def generate_initial_solution(jobs):
    job_counts = [0] * len(jobs)
    total_ops = sum(len(job) for job in jobs)
    solution = []

    while len(solution) < total_ops:
        available = [i for i in range(len(jobs)) if job_counts[i] < len(jobs[i])]
        job_id = random.choice(available)
        solution.append(job_id)
        job_counts[job_id] += 1
    return solution

# Decode solution to makespan
def calculate_makespan(jobs, job_order):
    machine_available = {}
    job_next_op = [0] * len(jobs)
    job_end_time = [0] * len(jobs)

    machine_end_time = {}
    for job_id in job_order:
        op_index = job_next_op[job_id]
        if op_index >= len(jobs[job_id]):
            continue
        machine, duration = jobs[job_id][op_index]

        start_time = max(job_end_time[job_id], machine_end_time.get(machine, 0))
        end_time = start_time + duration

        job_end_time[job_id] = end_time
        machine_end_time[machine] = end_time
        job_next_op[job_id] += 1

    return max(job_end_time)

# Neighbor by swapping two jobs
def generate_neighbor(solution):
    neighbor = solution.copy()
    i, j = random.sample(range(len(neighbor)), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

# Perturbation when stuck
def perturb(solution):
    solution = solution.copy()
    for _ in range(3):
        i, j = random.sample(range(len(solution)), 2)
        solution[i], solution[j] = solution[j], solution[i]
    return solution
"""
def simulated_annealing_great(jobs, max_iterations=1000, initial_temp=1000, cooling_rate=0.995, stagnation_threshold=100):
    current_solution = generate_initial_solution(jobs)
    current_cost = calculate_makespan(jobs, current_solution)
    best_solution = current_solution[:]
    best_cost = current_cost

    temperature = initial_temp
    stagnation_counter = 0

    for iteration in range(1, max_iterations + 1):
        neighbor = generate_neighbor(current_solution)
        neighbor_cost = calculate_makespan(jobs, neighbor)

        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_solution = neighbor
            current_cost = neighbor_cost

            if current_cost < best_cost:
                best_solution = current_solution[:]
                best_cost = current_cost
                stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= stagnation_threshold:
            current_solution = perturb(current_solution)
            current_cost = calculate_makespan(jobs, current_solution)
            stagnation_counter = 0
            print(f"Iteration {iteration}: Perturbation triggered.")

        temperature *= cooling_rate

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best makespan = {best_cost}")

    print(f"\nBest job order: {best_solution}")
    print(f"Best makespan: {best_cost}")
    
    return best_solution, best_cost
"""



def simulated_annealing(
    jobs,
    max_iterations=5000,
    initial_temperature=1000,
    cooling_rate=0.003,
    reheat_interval=1000,
    reheat_temperature=800,
    checkpoint_callback=None
):
    current_solution = list(range(len(jobs))) * len(jobs[0])
    random.shuffle(current_solution)
    current_cost = calculate_makespan(jobs, current_solution)
    best_solution = current_solution[:]
    best_cost = current_cost
    temperature = initial_temperature

    for iteration in range(1, max_iterations + 1):
        if iteration % (max_iterations/10) == 0: 
            print (f"iteration {iteration}") 
        neighbor = generate_neighbor(current_solution)
        neighbor_cost = calculate_makespan(jobs, neighbor)

        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_solution = neighbor
            current_cost = neighbor_cost

            if current_cost < best_cost:
                best_solution = current_solution[:]
                best_cost = current_cost

        temperature *= 1 - cooling_rate

        # Reheat
        if reheat_interval and iteration % reheat_interval == 0:
            temperature = reheat_temperature

        # Save checkpoint
        if checkpoint_callback and iteration % 500 == 0:
            checkpoint_callback(iteration, best_cost)

    return best_solution, best_cost

import csv
import os

def grid_search_sa(jobs):
    initial_temperatures = [500, 1000]
    cooling_rates = [0.001, 0.003, 0.005]
    reheat_intervals = [0, 1000]
    reheat_temperatures = [600, 800]

    os.makedirs("results", exist_ok=True)
    summary_file = open("results/summary.csv", "w", newline='')
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(["Temp", "CoolingRate", "ReheatInt", "ReheatTemp", "FinalCost"])

    for temp in initial_temperatures:
        for rate in cooling_rates:
            for reheat_int in reheat_intervals:
                for reheat_temp in reheat_temperatures:
                    checkpoint_log = []

                    def checkpoint_callback(iteration, best_cost):
                        checkpoint_log.append((iteration, best_cost))

                    best_solution, best_cost = simulated_annealing(
                        jobs,
                        max_iterations=5000,
                        initial_temperature=temp,
                        cooling_rate=rate,
                        reheat_interval=reheat_int,
                        reheat_temperature=reheat_temp,
                        checkpoint_callback=checkpoint_callback
                    )

                    # Save detailed log
                    filename = f"results/T{temp}_R{rate}_RI{reheat_int}_RT{reheat_temp}.csv"
                    with open(filename, "w", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Iteration", "BestCost"])
                        writer.writerows(checkpoint_log)

                    # Save summary
                    summary_writer.writerow([temp, rate, reheat_int, reheat_temp, best_cost])
                    print(f"Done: T={temp}, Rate={rate}, RI={reheat_int}, RT={reheat_temp}, Cost={best_cost}")

    summary_file.close()



def decode_schedule(jobs, job_order):
    machine_end_time = {}
    job_end_time = [0] * len(jobs)
    job_next_op = [0] * len(jobs)

    schedule = []  # (job_id, op_id, machine, start_time, end_time)

    for job_id in job_order:
        op_index = job_next_op[job_id]
        if op_index >= len(jobs[job_id]):
            continue

        machine, duration = jobs[job_id][op_index]

        start_time = max(job_end_time[job_id], machine_end_time.get(machine, 0))
        end_time = start_time + duration

        job_end_time[job_id] = end_time
        machine_end_time[machine] = end_time
        job_next_op[job_id] += 1

        schedule.append((job_id, op_index, machine, start_time, end_time))

    return schedule

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_schedule(schedule, num_machines):
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = cm.get_cmap('tab20', max(job_id for job_id, *_ in schedule) + 1)

    for job_id, op_id, machine, start, end in schedule:
        ax.barh(machine, end - start, left=start, color=colors(job_id), edgecolor='black', height=0.4)
        ax.text((start + end) / 2, machine, f'J{job_id}-O{op_id}', 
                ha='center', va='center', fontsize=8, color='white')

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Job Shop Schedule (Gantt Chart)')
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'M{m}' for m in range(num_machines)])
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    best_solution, best_cost = simulated_annealing(jobs, 10000000, 500,0.003, 0, 600) 
    
    print(f"\nBest job order: {best_solution}")
    print(f"Best makespan: {best_cost}")
    
    
    
    # Assuming you already defined 'jobs' variable
    # grid_search_sa(jobs)

    # schedule = decode_schedule(jobs, best_solution)
    # plot_schedule(schedule, num_machines=15)
    
    
