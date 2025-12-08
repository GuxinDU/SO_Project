#!/usr/bin/env python
import sys, os
import rvo2
from argparse import ArgumentParser

paser = ArgumentParser()
paser.add_argument('--method', '-m', type=int, default=0,)
paser.add_argument('--sample_budget', '-sb', type=int, default=10)
paser.add_argument('--radius_budget', '-rb', type=float, default=0.2)
args = paser.parse_args()
method = args.method
sample_budget = args.sample_budget
radius_budget = args.radius_budget
current_dir = os.path.dirname(os.path.abspath(__file__))

def exp1(method, sb=None, rb=None):
    if method == 0:
        namestring = 'Point Estimation {:d} samples'.format(sb)
    if method == 1:
        namestring = 'SAA {:d} samples'.format(sb)
    if method == 2:
        namestring = 'Robust Optimization {:.1f} budget'.format(rb)
    sim = rvo2.PyRVOSimulator(time_step=0.5)

    a0 = sim.addAgent((0, 0), (8, 8), 6, 5, 3, 0.5, 2, (0, 0))
    a1 = sim.addAgent((10, 0), (2, 8), 6, 5, 3, 0.5, 2, (0, 0))
    a2 = sim.addAgent((10, 10), (2, 2), 6, 5, 3, 0.5, 2, (0, 0))
    a3 = sim.addAgent((0, 10), (8, 2), 6, 5, 3, 0.5, 2, (0, 0))


    # print('Simulation has %i agents and %i obstacle vertices in it.' %
    #       (sim.getNumAgents(), sim.getNumObstacleVertices()))


    print('Running simulation')
    sim.init_relax_times()
    sim.init_arrive_dict()
    sim.check_positions()
    is_arrived = False
    step = 0
    while not is_arrived and step < 100:
    # for step in range(50):
    #     is_arrived = sim.doStep()
        is_arrived = sim.do_step_point_estimation(10)
        # is_arrived = sim.do_step_saa(10)
        # is_arrived = sim.do_step_robust(0.4)
        
        positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(agent_no)
                    for agent_no in (a0, a1, a2, a3)]
                #      for agent_no in (a0, a1, a2)]
        print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(positions)))
        step += 1

    print('Simulation finished')
    print('Relaxation times:', sum(sim.relax_times.values()))
    print('Total steps:', step)
    # print(sim.velocity_list)
    # print(sim.position_list)
    for i in range(step):
        print(rvo2.count_overlapping_balls(sim.position_list[i], radius=0.5))
    sim.save_positions(os.path.join(current_dir, 'results/example1', f'eg1_{namestring}_positions.txt'))
    sim.save_velocities(os.path.join(current_dir, 'results/example1', f'eg1_{namestring}_velocities.txt'))
    sim.save_relax_times(os.path.join(current_dir, 'results/example1', f'eg1_{namestring}_relax_times.txt'))
    rvo2.plot_trajectories(sim.position_list, output_file=os.path.join(current_dir, 'results/example1', f'eg1_{namestring}_trajectories.png'),
                    title="Example 1: Agent Trajectories")
    rvo2.create_animation(sim.position_list, sim.velocity_list, time_step=0.5, radius=0.5, output_file=os.path.join(current_dir, 'results/example1', f'eg1_{namestring}_animation.gif'),
                        title="Example 1: Agent Animation")

if __name__ == "__main__":
    for method in [0, 1, 2]:
        if method == 0:
            for sb in [1, 100]:
                exp1(method, sb=sb)
        if method == 1:
            for sb in [10, 100]:
                exp1(method, sb=sb)
        if method == 2:
            for rb in [0.2, 0.4]:
                exp1(method, rb=rb)