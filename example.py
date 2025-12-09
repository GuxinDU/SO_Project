#!/usr/bin/env python
import sys, os
import rvo2
import math
import random
# import subprocess
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

random.seed(42)
# sim = rvo2.PyRVOSimulator(1/60., 1.5, 5, 1.5, 0.4, 2)

def setup_scenario(agent_count, example_id):
    sim = rvo2.PyRVOSimulator(time_step=0.5)
    agents = []
    # Default parameters: neighbor_dist, max_neighbors, time_horizon, radius, max_speed, velocity
    # default_params = (6, 10, 3, 0.5, 2.0, (0, 0))

    if agent_count == 4:
        default_params = (6, 10, 3, 0.5, 2.0, (0, 0))

        if example_id == 1:
            # Existing 4-agent scenario
            agents.append(sim.addAgent((0, 0), (8.0, 8.0), *default_params))
            agents.append(sim.addAgent((10, 0), (2.0, 8.0), *default_params))
            agents.append(sim.addAgent((10, 10), (2.0, 2.0), *default_params))
            agents.append(sim.addAgent((0, 10), (8.0, 2.0), *default_params))
        elif example_id == 2:
            # 4 Agents crossing in a line
            agents.append(sim.addAgent((0, 0), (40, 0), *default_params))
            agents.append(sim.addAgent((10, 0), (30, 0), *default_params))
            agents.append(sim.addAgent((30, 0), (10, 0), *default_params))
            agents.append(sim.addAgent((40, 0), (0, 0), *default_params))
        elif example_id == 3:
            # 4 Agents in a diamond
            agents.append(sim.addAgent((0, 20), (0, -20), *default_params))
            agents.append(sim.addAgent((20, 0), (-20, 0), *default_params))
            agents.append(sim.addAgent((0, -20), (0, 20), *default_params))
            agents.append(sim.addAgent((-20, 0), (20, 0), *default_params))
            
    elif agent_count == 8:
        sim.set_time_step(1)
        default_params = (10, 10, 6, 1, 2, (0, 0))
        if example_id == 1:
            # 8 Agents Circle (Radius 4)
            # radius = 4.0
            radius = 40
            for i in range(8):
                angle = i * (2 * math.pi / 8)
                pos = (math.cos(angle) * radius, math.sin(angle) * radius)
                goal = (-pos[0], -pos[1])
                agents.append(sim.addAgent(pos, goal, *default_params))
        elif example_id == 2:
            # 8 Agents Two Lines
            for i in range(4):
                agents.append(sim.addAgent((-40, i*4 - 6), (40, i*4 - 6), *default_params))
                agents.append(sim.addAgent((40, i*4 - 6), (-40, i*4 - 6), *default_params))
        elif example_id == 3:
            # 8 Agents Circle (Radius 6, Rotated Goals)
            # radius = 6.0
            radius = 60
            for i in range(8):
                angle = i * (2 * math.pi / 8)
                pos = (math.cos(angle) * radius, math.sin(angle) * radius)
                goal = (math.cos(angle + math.pi/2) * radius, math.sin(angle + math.pi/2) * radius)
                agents.append(sim.addAgent(pos, goal, *default_params))

    elif agent_count == 15:
        sim.set_time_step(1)
        default_params = (10, 15, 6, 1, 2, (0, 0))
        if example_id == 1:
            # Existing 15-Agent Circle
            radius = 40
            for i in range(15):
                angle = i * (2 * math.pi / 15)
                pos = (math.cos(angle) * radius, math.sin(angle) * radius)
                goal = (-pos[0], -pos[1])
                agents.append(sim.addAgent(pos, goal, *default_params))
        elif example_id == 2:
            # 15 Agents Circle (Larger Radius)
            radius = 80.0
            for i in range(15):
                angle = i * (2 * math.pi / 15)
                pos = (math.cos(angle) * radius, math.sin(angle) * radius)
                goal = (-pos[0], -pos[1])
                agents.append(sim.addAgent(pos, goal, *default_params))
        elif example_id == 3:
            # 15 Agents Random Cluster
            for i in range(15):
                pos = (random.uniform(-50, 50), random.uniform(-50, 50))
                goal = (-pos[0], -pos[1])
                agents.append(sim.addAgent(pos, goal, *default_params))
    namestring = f"{agent_count}_{example_id}"
    return sim, agents, namestring

def run_simulation(agent_count, example_id, method, sample_budget=None, radius_budget=None):
    sim, agents, namestring = setup_scenario(agent_count, example_id)
    if method == 0:
        namestring += f"_PE_{sample_budget:d}"
    if method == 1:
        namestring += f"_SAA_{sample_budget:d}"
    if method == 2:
        namestring += f"_RO_{radius_budget:.1f}"
    if method == 3:
        namestring += f"_ER_{radius_budget:.1f}"
    is_arrived = False
    sim.init_relax_times()
    sim.init_arrive_dict()
    sim.check_positions()
 
    step = 0
    while not is_arrived and step < 9999:
        if method == 0:
            is_arrived = sim.do_step_point_estimation(sample_budget)
        elif method == 1:
            is_arrived = sim.do_step_saa(sample_budget)
        elif method == 2:
            is_arrived = sim.do_step_robust(radius_budget)
        elif method == 3:
            is_arrived = sim.do_step_extend_radius(radius_budget)
        step += 1
    print('Simulation finished')
    print('Relaxation times:', sum(sim.relax_times.values()))
    print('Total steps:', step)
    # print(sim.velocity_list)
    # print(sim.position_list)
    # for i in range(step):
    #     print(rvo2.count_overlapping_balls(sim.position_list[i], radius=0.5))
    sim.save_positions(os.path.join(current_dir, 'results/experiment', f'{namestring}_positions.txt'))
    sim.save_velocities(os.path.join(current_dir, 'results/experiment', f'{namestring}_velocities.txt'))
    sim.save_relax_times(os.path.join(current_dir, 'results/experiment', f'{namestring}_relax_times.txt'))
    rvo2.plot_trajectories(sim.position_list, output_file=os.path.join(current_dir, 'results/experiment', f'{namestring}_trajectories.png'),
                    title=f"{namestring}: Agent Trajectories")
    if step < 200:
        rvo2.create_animation(sim.position_list, sim.velocity_list, time_step=0.5, radius=sim.agents[0].radius, output_file=os.path.join(current_dir, 'results/experiment', f'{namestring}_animation.gif'),
                            title=f"{namestring}: Agent Animation")


if __name__ == "__main__":
    if not os.path.exists(os.path.join(current_dir, 'results/experiment')):
        os.makedirs(os.path.join(current_dir, 'results/experiment'))
    for agent_count in [4]:
        for example_id in [1, 2]:
            # for method in [0, 1, 2]:
            for method in [3]:
                if method == 0:
                    # for sb in [1, 10, 100]:
                    for sb in [1, 10, 50]:
                        run_simulation(agent_count, example_id, method, sample_budget=sb)
                if method == 1:
                    for sb in [10, 50]:
                        run_simulation(agent_count, example_id, method, sample_budget=sb)
                if method == 2:
                    for rb in [0.2, 0.4]:
                        run_simulation(agent_count, example_id, method, radius_budget=rb)
                if method == 3:
                    for rb in [0.2, 0.4]:
                        run_simulation(agent_count, example_id, method, radius_budget=rb)
    for agent_count in [8]:
        for example_id in [1, 2]:
            # for method in [0, 1, 2]:
            for method in [3]:
                if method == 0:
                    # for sb in [1, 10, 100]:
                    for sb in [1, 10, 50]:
                        run_simulation(agent_count, example_id, method, sample_budget=sb)
                if method == 1:
                    for sb in [10, 50]:
                        run_simulation(agent_count, example_id, method, sample_budget=sb)
                if method == 2:
                    for rb in [0.2, 0.4]:
                        run_simulation(agent_count, example_id, method, radius_budget=rb)
                if method == 3:
                    for rb in [0.2, 0.4]:
                        run_simulation(agent_count, example_id, method, radius_budget=rb)
    for agent_count in [15]:
        for example_id in [1,2]:
            # for method in [0, 1, 2]:
            for method in [3]:
                if method == 0:
                    # for sb in [1, 10, 100]:
                    for sb in [1, 10, 50]:
                        run_simulation(agent_count, example_id, method, sample_budget=sb)
                if method == 1:
                    for sb in [10, 50]:
                        run_simulation(agent_count, example_id, method, sample_budget=sb)
                if method == 2:
                    for rb in [0.2, 0.4]:
                        run_simulation(agent_count, example_id, method, radius_budget=rb)
                if method == 3:
                    for rb in [0.2, 0.4]:
                        run_simulation(agent_count, example_id, method, radius_budget=rb)
 
 
# # Interactive Input
# print("Available Scenarios: 4_1, 4_2, 4_3, 8_1, 8_2, 8_3, 15_1, 15_2, 15_3")
# scenario_input = input("Enter scenario (e.g. 8_3): ")
# print("Available Generation Methods: deterministic, Uniformdisk, biasedG, AsymetricG")
# gen_method_input = input("Enter generate sample method: ")

# try:
#     count_str, id_str = scenario_input.split('_')
#     agent_count = int(count_str)
#     example_id = int(id_str)
# except ValueError:
#     print("Invalid scenario format. Using default 4_1.")
#     agent_count = 4
#     example_id = 1
#     scenario_input = "4_1"

# if gen_method_input not in ['deterministic', 'Uniformdisk', 'biasedG', 'AsymetricG']:
#     print("Invalid method. Using deterministic.")
#     gen_method_input = 'deterministic'

# # Set seed for reproducibility

# # Set generation method in simulator for agents to use
# sim.gen_method = gen_method_input

# agents = setup_scenario(sim, agent_count, example_id)

# # Construct filenames
# base_name = f"{scenario_input}_{gen_method_input}"
# traj_file = f"trajectory_{base_name}.txt"
# vel_file = f"Velocity_{base_name}.txt"
# perf_file = f"Performance_{base_name}.txt"

# with open(traj_file, 'w') as f_traj, \
#      open(vel_file, 'w') as f_vel, \
#      open(perf_file, 'w') as f_perf:
    
#     print('Simulation has %i agents and %i obstacle vertices in it.' %
#           (sim.getNumAgents(), sim.getNumObstacleVertices()))
#     f_traj.write('Simulation has %i agents and %i obstacle vertices in it.\n' %
#           (sim.getNumAgents(), sim.getNumObstacleVertices()))

#     print('Running simulation')
#     f_traj.write('Running simulation\n')

#     for step in range(2000):
#         sim.doStep()

#         # Trajectory
#         positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(agent_no)
#                      for agent_no in agents]
#         line_traj = 'step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(positions))
#         print(line_traj)
#         f_traj.write(line_traj + '\n')

#         # Velocity
#         velocities = ['(%5.3f, %5.3f)' % sim.getAgentVelocity(agent_no)
#                       for agent_no in agents]
#         line_vel = 'step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(velocities))
#         f_vel.write(line_vel + '\n')

#         # Performance
#         perf_info = []
#         for agent_no in agents:
#             runtime = sim.getAgentSaaRuntime(agent_no)
#             statuses = sim.getAgentQpStatuses(agent_no)
#             perf_info.append('A%d: %.4fs %s' % (agent_no, runtime, str(statuses)))
        
#         line_perf = 'step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(perf_info))
#         f_perf.write(line_perf + '\n')

#         if sim.have_all_agents_reached_goal():
#             print("All agents reached their goals!")
#             f_traj.write("All agents reached their goals!\n")
#             break

# # Run visualization
# print(f"Running visualization for {traj_file}...")
# subprocess.run([sys.executable, 'draw_trajectory.py', traj_file])

