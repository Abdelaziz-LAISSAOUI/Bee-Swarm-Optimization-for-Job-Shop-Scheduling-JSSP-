import random
import csv
from typing import List, Tuple

# --------------------- Parser ---------------------

def parse_multiple_job_shop_instances(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    benchmarks = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("Nb of jobs"):
            i += 1
            header = list(map(int, lines[i].split()))
            nb_jobs, nb_machines = header[0], header[1]
            i += 1

            while not lines[i].startswith("Times"):
                i += 1
            i += 1
            times_lines = lines[i:i + nb_jobs]
            i += nb_jobs
            times = [int(x) for line in times_lines for x in line.split()]

            while not lines[i].startswith("Machines"):
                i += 1
            i += 1
            machines_lines = lines[i:i + nb_jobs]
            i += nb_jobs
            machines = [int(x) - 1 for line in machines_lines for x in line.split()]

            jobs = []
            for j in range(nb_jobs):
                job_ops = []
                for m in range(nb_machines):
                    idx = j * nb_machines + m
                    job_ops.append((machines[idx], times[idx]))
                jobs.append(job_ops)

            benchmarks.append({
                'nb_jobs': nb_jobs,
                'nb_machines': nb_machines,
                'jobs': jobs
            })
        else:
            i += 1
    return benchmarks

# --------------------- JSSP Core ---------------------

def generate_operation_sequence(jobs: List[List[Tuple[int, int]]]) -> List[int]:
    nb_jobs = len(jobs)
    nb_ops = len(jobs[0])
    sequence = [j for j in range(nb_jobs) for _ in range(nb_ops)]
    random.shuffle(sequence)
    return sequence

def decode_schedule(sequence: List[int], jobs: List[List[Tuple[int, int]]]) -> Tuple[int, List[Tuple[int, int, int, int]]]:
    nb_machines = max(max(op[0] for op in job) for job in jobs) + 1
    job_indices = [0] * len(jobs)
    machine_available = [0] * nb_machines
    job_available = [0] * len(jobs)
    schedule = []

    for job_id in sequence:
        op_index = job_indices[job_id]
        if op_index >= len(jobs[job_id]):
            continue
        machine, duration = jobs[job_id][op_index]
        
        start_time = max(machine_available[machine], job_available[job_id])
        end_time = start_time + duration
        
        machine_available[machine] = end_time
        job_available[job_id] = end_time
        job_indices[job_id] += 1
        
        schedule.append((job_id, op_index, start_time, end_time))

    if not all(job_indices[job] == len(jobs[job]) for job in range(len(jobs))):
        return float('inf'), []
    makespan = max(end for _, _, _, end in schedule)
    return makespan, schedule

# --------------------- BSO with Checkpoints ---------------------

def generate_neighbors(seq: List[int], num_neighbors: int) -> List[List[int]]:
    neighbors = []
    for _ in range(num_neighbors):
        neighbor = seq[:]
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbors.append(neighbor)
    return neighbors

def local_search(seq: List[int], jobs: List[List[Tuple[int, int]]], depth: int = 20) -> Tuple[int, List[int]]:
    best_seq = seq[:]
    best_makespan, _ = decode_schedule(best_seq, jobs)
    for _ in range(depth):
        neighbor = best_seq[:]
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        mk, _ = decode_schedule(neighbor, jobs)
        if mk < best_makespan:
            best_makespan = mk
            best_seq = neighbor
    return best_makespan, best_seq

# def bee_swarm_optimization(jobs, max_iter, num_bees, neighbors_per_bee, local_depth, checkpoints, log_callback):
def bee_swarm_optimization(jobs, max_iter, num_bees, local_depth, checkpoints, log_callback):
    Sref = generate_operation_sequence(jobs)
    best_makespan, _ = decode_schedule(Sref, jobs)
    taboo = set()
    taboo.add(tuple(Sref))  

    for it in range(1, max_iter + 1):
        search_points = generate_neighbors(Sref, num_bees)
        dances = []
        for sp in search_points:
            mk, improved = local_search(sp, jobs, depth=local_depth)
            dances.append((mk, improved))
        dances.sort()
        best_mk, best_seq = dances[0]
        if tuple(best_seq) not in taboo:
            Sref = best_seq
            taboo.add(tuple(Sref))
            best_makespan = best_mk

        if it in checkpoints:
            log_callback(it, best_makespan) 

# --------------------- Grid Search Experiment ---------------------

def run_experiments():
    benchmarks = parse_multiple_job_shop_instances("data/tai20_15.txt")
    parameter_grid = {
        "num_bees": [5, 10, 20],
        # "neighbors_per_bee": [3, 5, 10],
        "local_depth": [10, 20]
    }
    checkpoints = list(range(1000, 10001, 1000))
    max_iter = 10000

    with open("bso_results2.csv", "w", newline="") as csvfile:
        fieldnames = ["benchmark", "num_bees", "neighbors_per_bee", "local_depth", "iteration", "makespan"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for bench_id, instance in enumerate(benchmarks):
            for nb in parameter_grid["num_bees"]:
                # for nnb in parameter_grid["neighbors_per_bee"]:
                    for ld in parameter_grid["local_depth"]:
                        def log_callback(it, mk):
                            writer.writerow({
                                "benchmark": bench_id + 1,
                                "num_bees": nb,
                                # "neighbors_per_bee": nnb,
                                "local_depth": ld,
                                "iteration": it,
                                "makespan": mk
                            })
                        bee_swarm_optimization(
                            instance["jobs"],
                            max_iter=max_iter,
                            num_bees=nb,
                            # neighbors_per_bee=nnb,
                            local_depth=ld,
                            checkpoints=checkpoints,
                            log_callback=log_callback
                        )

if __name__ == "__main__":
    run_experiments()
