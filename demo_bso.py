def parse_multiple_job_shop_instances(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    benchmarks = []
    i = 0
    while i < len(lines):
        # Header line: "Nb of jobs, Nb of Machines, ..."
        if lines[i].startswith("Nb of jobs"):
            i += 1
            header = list(map(int, lines[i].split()))
            nb_jobs, nb_machines = header[0], header[1]
            i += 1  # Move past header line

            # Find and parse Times
            while not lines[i].startswith("Times"):
                i += 1
            i += 1
            times_lines = lines[i:i + nb_jobs]
            i += nb_jobs
            times = [int(x) for line in times_lines for x in line.split()]

            # Find and parse Machines
            while not lines[i].startswith("Machines"):
                i += 1
            i += 1
            machines_lines = lines[i:i + nb_jobs]
            i += nb_jobs
            machines = [int(x)-1 for line in machines_lines for x in line.split()]  # 0-based indexing

            # Create job structure
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
            i += 1  # Skip lines until next header

    return benchmarks

# Example usage
benchmarks = parse_multiple_job_shop_instances("data/tai20_15.txt")
print(f"Parsed {len(benchmarks)} benchmarks.")
print(f"First job of first benchmark: {benchmarks[0]['jobs']}")

print()
print()
for j in benchmarks[0]['jobs']: 
    print(j) 
print()
print()


from collections import defaultdict
import heapq
import random
from typing import List, Tuple

def generate_operation_sequence(jobs: List[List[Tuple[int, int]]]) -> List[int]:
    """
    Generate a valid random operation-based representation for JSSP.
    Each job ID is repeated as many times as it has operations.
    """
    nb_jobs = len(jobs)
    nb_ops = len(jobs[0])
    sequence = [j for j in range(nb_jobs) for _ in range(nb_ops)]
    random.shuffle(sequence)
    return sequence

def decode_schedule(sequence: List[int], jobs: List[List[Tuple[int, int]]]) -> Tuple[int, List[Tuple[int, int, int, int]]]:
    """
    Build a schedule from a job operation sequence.
    Returns the makespan and the detailed operation start times.
    """
    nb_machines = max(max(op[0] for op in job) for job in jobs) + 1 # really wtf ur doin CHANGE THAT !   to 15 directly 
    job_indices = [0] * len(jobs)
    machine_available = [0] * nb_machines
    job_available = [0] * len(jobs)
    schedule = []  # (job_id, op_index, start_time, end_time)

    for job_id in sequence:
        op_index = job_indices[job_id]
        if op_index >= len(jobs[job_id]):
            continue  # invalid sequence, more operations than allowed (it is impossible to happen anyways)
        
        machine, duration = jobs[job_id][op_index]

        start_time = max(machine_available[machine], job_available[job_id])
        end_time = start_time + duration

        machine_available[machine] = end_time
        job_available[job_id] = end_time
        job_indices[job_id] += 1

        schedule.append((job_id, op_index, start_time, end_time))

    # Check validity
    valid = all(job_indices[job] == len(jobs[job]) for job in range(len(jobs)))
    if not valid:
        return float('inf'), []  # Invalid solution

    makespan = max(end for _, _, _, end in schedule)
    return makespan, schedule

# Test on the first benchmark
# sample_jobs = benchmarks[0]['jobs']
# sample_sequence = generate_operation_sequence(sample_jobs)
# makespan, schedule = decode_schedule(sample_sequence, sample_jobs)

# print(makespan, schedule[:5])  # show only first 5 operations for brevity



def generate_neighbors(seq: List[int], num_neighbors: int) -> List[List[int]]:
    neighbors = []
    for _ in range(num_neighbors):
        neighbor = seq[:]
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbors.append(neighbor)
    return neighbors

def local_search(seq: List[int], jobs: List[List[Tuple[int, int]]], max_iter: int = 20) -> Tuple[int, List[int]]:
    best_seq = seq[:]
    best_makespan, _ = decode_schedule(best_seq, jobs)
    for _ in range(max_iter):
        neighbor = best_seq[:]
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        mk, _ = decode_schedule(neighbor, jobs)
        if mk < best_makespan:
            best_makespan = mk
            best_seq = neighbor
    return best_makespan, best_seq

def bee_swarm_optimization(jobs, max_iter=1000, num_bees=50):
    Sref = generate_operation_sequence(jobs)
    best_makespan, _ = decode_schedule(Sref, jobs)
    taboo = set()
    taboo.add(tuple(Sref))

    for it in range(max_iter):
        search_points = generate_neighbors(Sref, num_bees)
        dances = []
        for sp in search_points:
            mk, improved = local_search(sp, jobs)
            dances.append((mk, improved))
        dances.sort()
        best_mk, best_seq = dances[0]
        if tuple(best_seq) in taboo:
            continue
        Sref = best_seq
        taboo.add(tuple(Sref))
        best_makespan = best_mk
        if it%(max_iter/10) == 0 : 
            print("we are in ", it)
    return best_makespan, Sref


# # --------------------- Gantt Chart ---------------------
import matplotlib.pyplot as plt

def generate_gantt_data(sequence, jobs):
    nb_jobs = len(jobs)
    nb_machines = max(max(op[0] for op in job) for job in jobs) + 1

    job_indices = [0] * nb_jobs
    job_available = [0] * nb_jobs
    machine_available = [0] * nb_machines

    gantt_data = []

    for job_id in sequence:
        op_idx = job_indices[job_id]
        if op_idx >= len(jobs[job_id]):
            continue

        machine_id, duration = jobs[job_id][op_idx]
        start_time = max(machine_available[machine_id], job_available[job_id])
        end_time = start_time + duration

        job_indices[job_id] += 1
        machine_available[machine_id] = end_time
        job_available[job_id] = end_time

        gantt_data.append({
            "Job": f"Job {job_id}",
            "Machine": machine_id,
            "Start": start_time,
            "Duration": duration,
            "End": end_time,
            "Color": job_id
        })

    return gantt_data

def plot_gantt(gantt_data):
    fig, ax = plt.subplots(figsize=(16, 8))
    yticks = []
    yticklabels = []
    colors = plt.cm.get_cmap('tab20', len(set([g['Job'] for g in gantt_data])))

    for g in gantt_data:
        y = g["Machine"]
        ax.barh(y, g["Duration"], left=g["Start"], color=colors(g["Color"]), edgecolor='black')
        ax.text(g["Start"] + g["Duration"] / 2, y, g["Job"], va='center', ha='center', fontsize=7, color='white')
        if y not in yticks:
            yticks.append(y)
            yticklabels.append(f"Machine {y}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time")
    ax.set_title("JSSP Gantt Chart (Matplotlib)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()



# # --------------------- Runner ---------------------

if __name__ == "__main__":
    benchmarks = parse_multiple_job_shop_instances("data/tai20_15.txt")
    # for idx, instance in enumerate(benchmarks):
    mk, best_seq = bee_swarm_optimization(benchmarks[0]['jobs'], max_iter=1000, num_bees=10)
    print(f"Benchmark {1}: Best Makespan = {mk}")
    
    plot_gantt(generate_gantt_data(best_seq, benchmarks[0]['jobs']))
