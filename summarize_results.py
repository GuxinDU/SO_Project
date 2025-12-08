import os
import pandas as pd
import numpy as np
from rvo2.simulator import RVOSimulator
from rvo2.myFuncs import count_overlapping_balls

def get_radius(agent_count):
    if agent_count == 4:
        return 0.5
    elif agent_count == 8:
        return 1.0
    elif agent_count == 15:
        return 1.0
    return 0.5

def summarize():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/experiment')
    
    data = []

    # Define the loops as in example.py
    agent_counts = [4, 8, 15]
    example_ids = [1, 2]
    methods = [0, 1, 2] # 0: PE, 1: SAA, 2: RO

    for agent_count in agent_counts:
        for example_id in example_ids:
            for method in methods:
                configs = []
                if method == 0: # PE
                    for sb in [1, 10, 50]:
                        configs.append(({'sample_budget': sb}, f"_PE_{sb}"))
                elif method == 1: # SAA
                    for sb in [10, 50]:
                        configs.append(({'sample_budget': sb}, f"_SAA_{sb}"))
                elif method == 2: # RO
                    for rb in [0.2, 0.4]:
                        configs.append(({'radius_budget': rb}, f"_RO_{rb:.1f}"))
                
                for params, suffix in configs:
                    namestring = f"{agent_count}_{example_id}{suffix}"
                    
                    pos_file = os.path.join(results_dir, f'{namestring}_positions.txt')
                    relax_file = os.path.join(results_dir, f'{namestring}_relax_times.txt')
                    
                    if not os.path.exists(pos_file):
                        print(f"File not found: {pos_file}")
                        continue
                        
                    # Load data
                    sim = RVOSimulator()
                    sim.load_positions(pos_file)
                    sim.load_relax_times(relax_file)
                    
                    # Calculate metrics
                    steps = len(sim.position_list)
                    
                    radius = get_radius(agent_count)
                    total_collisions = 0
                    for pos_dict in sim.position_list:
                        # count_overlapping_balls returns number of agents in collision
                        total_collisions += count_overlapping_balls(pos_dict, radius)
                        
                    total_relaxations = sum(sim.relax_times.values())
                    
                    # Prepare row
                    row = {
                        'Agent Count': agent_count,
                        'Example ID': example_id,
                        'Method': ['PE', 'SAA', 'RO'][method],
                        'Parameter': list(params.values())[0],
                        'Steps': steps,
                        'Total Collisions (Agent-Steps)': total_collisions,
                        'Total Relaxations': total_relaxations
                    }
                    data.append(row)

    df = pd.DataFrame(data)
    
    output_file = os.path.join(results_dir, 'summary.xlsx')
    try:
        df.to_excel(output_file, index=False)
        print(f"Summary saved to {output_file}")
    except ImportError:
        print("pandas or openpyxl not installed. Saving to CSV instead.")
        output_csv = os.path.join(results_dir, 'summary.csv')
        df.to_csv(output_csv, index=False)
        print(f"Summary saved to {output_csv}")
    except Exception as e:
        print(f"Error saving summary: {e}")

if __name__ == "__main__":
    summarize()
