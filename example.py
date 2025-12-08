#!/usr/bin/env python

import rvo2

# sim = rvo2.PyRVOSimulator(1/60., 1.5, 5, 1.5, 0.4, 2)
sim = rvo2.PyRVOSimulator(time_step=0.5)
# sim = rvo2.PyRVOSimulator(time_step=1.0)

# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
# a0 = sim.addAgent((0, 0), (0.8, 0.8), 1.5, 5, 1.5, 0.1, 2, (0, 0))
# a1 = sim.addAgent((1, 0), (0.2, 0.8), 1.5, 5, 1.5, 0.1, 2, (0, 0))
# a2 = sim.addAgent((1, 1), (0.2, 0.2), 1.5, 5, 1.5, 0.1, 2, (0, 0))
# a3 = sim.addAgent((0, 1), (0.8, 0.2), 1.5, 5, 1.5, 0.1, 2, (0, 0))

a0 = sim.addAgent((0, 0), (8, 8), 5, 5, 3, 0.5, 2, (0, 0))
a1 = sim.addAgent((10, 0), (2, 8), 5, 5, 3, 0.5, 2, (0, 0))
a2 = sim.addAgent((10, 10), (2, 2), 5, 5, 3, 0.5, 2, (0, 0))
a3 = sim.addAgent((0, 10), (8, 2), 5, 5, 3, 0.5, 2, (0, 0))

# Obstacles are also supported.
# o1 = sim.addObstacle([(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1)])
# sim.processObstacles()

# sim.setAgentPrefVelocity(a0, (1, 1))
# sim.setAgentPrefVelocity(a1, (-1, 1))
# sim.setAgentPrefVelocity(a2, (-1, -1))
# sim.setAgentPrefVelocity(a3, (1, -1))

print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))

print('Running simulation')
sim.init_relax_times()
sim.init_arrive_dict()
sim.check_positions()
is_arrived = False
step = 0
while not is_arrived and step < 100:
# for step in range(50):
#     is_arrived = sim.doStep()
#     is_arrived = sim.do_step_point_estimation(1)
#     is_arrived = sim.do_step_saa(10)
    is_arrived = sim.do_step_robust(0.3)
    
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
