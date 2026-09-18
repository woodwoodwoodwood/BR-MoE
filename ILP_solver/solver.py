import json
import os
import argparse
import numpy as np
import re
from ortools.linear_solver import pywraplp
from tabulate import tabulate
from collections import defaultdict

def load_expert_freq(freq_file_path):
    with open(freq_file_path, 'r') as f:
        freq_data = json.load(f)
    
    expert_freq = {}
    
    if isinstance(freq_data, dict):
        if "frequency" in freq_data:
            freq_dict = freq_data["frequency"]
            for layer_key, experts_freq in freq_dict.items():
                layer_match = re.search(r'layers\.([0-9]+)', layer_key)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    for expert_idx, freq in enumerate(experts_freq):
                        expert_key = f"L{layer_idx}_E{expert_idx}"
                        expert_freq[expert_key] = float(freq)
        else:
            expert_pattern = re.compile(r'L\d+_E\d+')
            for key, value in freq_data.items():
                if expert_pattern.match(key):
                    expert_freq[key] = float(value)
                    
    elif isinstance(freq_data, list):
        for layer_idx, layer_freqs in enumerate(freq_data):
            for expert_idx, freq in enumerate(layer_freqs):
                expert_key = f"L{layer_idx+1}_E{expert_idx}"
                expert_freq[expert_key] = float(freq)
    
    return expert_freq

def extract_rank_from_config(config_name):
    match = re.search(r'rank(\d+)', config_name)
    if match:
        return int(match.group(1))
    return 0

def solve_quantization_config(l2_results, memory_usage, expert_freq, total_budget_mb, min_rank=16, max_rank=256):
    experts = list(l2_results.keys())
    all_configs = list(memory_usage.keys())
    
    first_expert = experts[0]
    available_configs = list(l2_results[first_expert].keys())
    
    configs = []
    filtered_configs_low = []
    filtered_configs_high = []
    for config in all_configs:
        if config in available_configs:
            rank = extract_rank_from_config(config)
            if rank < min_rank:
                filtered_configs_low.append(config)
            elif max_rank is not None and rank > max_rank:
                filtered_configs_high.append(config)
            else:
                configs.append(config)
    
    rank_range_str = f"rank >= {min_rank}" if max_rank is None else f"{min_rank} <= rank <= {max_rank}"
    print(f"Total configs: {len(all_configs)}, Filtered configs: {len(configs)} ({rank_range_str})")
    
    if not configs:
        print(f"Error: No configs satisfy {rank_range_str}!")
        return None
    
    print(f"Memory budget: {total_budget_mb:.2f} MB")

    values = {}
    weights = {}
    l2_values = {}
    
    bit_pattern = re.compile(r'(\d+)bit')

    for i, expert_key in enumerate(experts):
        freq_weight = expert_freq.get(expert_key, 1.0)

        for j, config_key in enumerate(configs):
            if config_key not in l2_results[expert_key]:
                print(f"Warning: Expert {expert_key} missing config {config_key}, skipping")
                continue
            
            l2_loss = l2_results[expert_key][config_key]
            l2_values[(i, j)] = l2_loss
            values[(i, j)] = l2_loss * freq_weight
            weights[(i, j)] = memory_usage[config_key]

    num_experts = len(experts)
    num_configs = len(configs)
    print(f"Total: {num_experts} experts, {num_configs} configs")

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("SCIP solver unavailable")
        return None

    x = {}
    for i in range(num_experts):
        for j in range(num_configs):
            x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')

    invalid_4bit_indices = []
    for j, cfg in enumerate(configs):
        bit_match = re.search(r'(\d+)bit', cfg)
        if bit_match and bit_match.group(1) == '4':
            if extract_rank_from_config(cfg) > 32:
                invalid_4bit_indices.append(j)
    for j in invalid_4bit_indices:
        solver.Add(solver.Sum([x[i, j] for i in range(num_experts)]) == 0)

    memory_constraint = solver.RowConstraint(0, total_budget_mb, 'memory_budget')
    for i in range(num_experts):
        for j in range(num_configs):
            memory_constraint.SetCoefficient(x[i, j], weights[i, j])

    for i in range(num_experts):
        choice_constraint = solver.RowConstraint(1, 1, f'expert_{i}_choice')
        for j in range(num_configs):
            choice_constraint.SetCoefficient(x[i, j], 1)

    objective = solver.Objective()
    for i in range(num_experts):
        for j in range(num_configs):
            objective.SetCoefficient(x[i, j], -values[i, j])
    objective.SetMinimization()

    print(f"Solving... (vars: {solver.NumVariables()}, constraints: {solver.NumConstraints()})")
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal solution found!")
        total_memory_used = 0
        total_loss = 0
        best_config_map = {}
        config_counts = defaultdict(int)

        for i in range(num_experts):
            for j in range(num_configs):
                if x[i, j].solution_value() > 0.5:
                    expert_key = experts[i]
                    config_key = configs[j]
                    best_config_map[expert_key] = config_key
                    total_memory_used += weights[i, j]
                    total_loss += values[i, j]
                    config_counts[config_key] += 1
        
        print(f"Total loss: {total_loss:.4f}")
        print(f"Total memory used: {total_memory_used:.4f} MB (budget: {total_budget_mb:.2f} MB)")
        
        bit_counts = defaultdict(int)
        for config in best_config_map.values():
            bit_match = re.search(r'(\d+)bit', config)
            if bit_match:
                bit = bit_match.group(1)
                bit_counts[f"{bit}bit"] += 1
        
        print("\nBit-width statistics:")
        bit_table = [(bit, count, f"{count/num_experts*100:.2f}%") 
                      for bit, count in sorted(bit_counts.items())]
        print(tabulate(bit_table, headers=["Bits", "Count", "Percentage"], tablefmt="grid"))
        
        return best_config_map
    else:
        print('No optimal solution found')
        return None

def main():
    parser = argparse.ArgumentParser(description="ILP solver for expert quantization configuration")
    parser.add_argument("--l2_file", type=str, 
                      default="./expert_impact_results.json")
    parser.add_argument("--freq_file", type=str,
                      default="./expert_freq_lo.json")
    parser.add_argument("--memory_file", type=str,
                      default="./memory_usage.json")
    parser.add_argument("--output_file", type=str, default="optimal_config.json")
    parser.add_argument("--budget", type=float, default=11880.0)
    parser.add_argument("--min_rank", type=int, default=16)
    parser.add_argument("--max_rank", type=int, default=256)
    
    args = parser.parse_args()
    
    for file_path in [args.l2_file, args.freq_file, args.memory_file]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist!")
            return
    
    print(f"Loading L2 data: {args.l2_file}")
    with open(args.l2_file, 'r') as f:
        l2_results = json.load(f)
    
    print(f"Loading memory data: {args.memory_file}")
    with open(args.memory_file, 'r') as f:
        memory_usage = json.load(f)
    
    print(f"Loading frequency data: {args.freq_file}")
    expert_freq = load_expert_freq(args.freq_file)
    
    best_configs = solve_quantization_config(
        l2_results, 
        memory_usage, 
        expert_freq, 
        args.budget,
        args.min_rank,
        args.max_rank
    )
    
    if best_configs:
        with open(args.output_file, 'w') as f:
            json.dump(best_configs, f, indent=4)
        print(f"\nOptimal config saved to: {args.output_file}")

if __name__ == "__main__":
    main()