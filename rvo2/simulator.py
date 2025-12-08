from .vector import Vector2, normalize, abs_vector, abs_sq
from .agent import Agent
from .obstacle import Obstacle
from .kdtree import KdTree
from .utils import leftOf, sqr

RVO_ERROR = -1

def to_vector(v):
    if isinstance(v, Vector2):
        return v
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return Vector2(v[0], v[1])
    return v

class RVOSimulator:
    def __init__(self, time_step=0.0, neighbor_dist=0.0, max_neighbors=0, time_horizon=0.0, radius=0.0, max_speed=0.0, velocity=None):
        self.time_step = time_step
        self.agents = []
        self.default_agent = None
        self.global_time = 0.0
        self.kd_tree = KdTree(self)
        self.obstacles = []
        self.agents_on_the_way = []
        self.position_list = []
        self.velocity_list = []
        self.relax_times = {}
        self.arrive_dict = {}

        velocity = to_vector(velocity)
        if velocity is None:
            velocity = Vector2()

        if neighbor_dist > 0 or max_neighbors > 0 or time_horizon > 0 or radius > 0 or max_speed > 0:
             self.default_agent = Agent(self)
             self.default_agent.max_neighbors = max_neighbors
             self.default_agent.max_speed = max_speed
             self.default_agent.neighbor_dist = neighbor_dist
             self.default_agent.radius = radius
             self.default_agent.time_horizon = time_horizon
             self.default_agent.velocity = velocity
    
    def add_agent(self, position, goal, neighbor_dist=None, max_neighbors=None, time_horizon=None, radius=1.0, max_speed=2.0, velocity=None):
        position = to_vector(position)
        velocity = to_vector(velocity)
        goal = to_vector(goal)
        
        if neighbor_dist is None:
            if self.default_agent is None:
                return RVO_ERROR
            
            agent = Agent(self)
            agent.position = position
            agent.max_neighbors = self.default_agent.max_neighbors
            agent.max_speed = self.default_agent.max_speed
            agent.neighbor_dist = self.default_agent.neighbor_dist
            agent.radius = self.default_agent.radius
            agent.time_horizon = self.default_agent.time_horizon
            agent.velocity = self.default_agent.velocity
            agent.goal_position = goal # Set goal position
            agent.id = len(self.agents)
            agent._init_qp_model() # Initialize QP model
            agent._init_error_generator((-0.15, 0.15), (-0.15, 0.15), (-0.07, 0.07))
            self.agents.append(agent)
            self.agents_on_the_way.append(agent)
            return len(self.agents) - 1
        else:
            agent = Agent(self)
            agent.position = position
            agent.max_neighbors = max_neighbors
            agent.max_speed = max_speed
            agent.neighbor_dist = neighbor_dist
            agent.radius = radius
            agent.time_horizon = time_horizon
            agent.velocity = velocity if velocity is not None else Vector2()
            agent.goal_position = goal # Set goal position
            agent.id = len(self.agents)
            agent._init_qp_model() # Initialize QP model
            agent._init_error_generator((-0.15, 0.15), (-0.15, 0.15), (-0.07, 0.07))
            self.agents.append(agent)
            self.agents_on_the_way.append(agent)
            return len(self.agents) - 1

    # def add_obstacle(self, vertices):
    #     # Each obstacle is defined by a list of vertices (at least 2) and the last vertex is connected to the first
    #     # The self.obstacles list contains all obstacle vertices in order
    #     # All the vertices are later stored in the KdTree for obstacle neighbor queries,
    #     # and the connection between vertices is stored in the prev_obstacle and next_obstacle attributes of each Obstacle
    #     if len(vertices) < 2:
    #         return RVO_ERROR
        
    #     vertices = [to_vector(v) for v in vertices]
        
    #     obstacle_no = len(self.obstacles)

    #     for i in range(len(vertices)):
    #         obstacle = Obstacle()
    #         obstacle.point = vertices[i]

    #         if i != 0:
    #             obstacle.prev_obstacle = self.obstacles[-1]
    #             obstacle.prev_obstacle.next_obstacle = obstacle
            
    #         if i == len(vertices) - 1:
    #             obstacle.next_obstacle = self.obstacles[obstacle_no]
    #             obstacle.next_obstacle.prev_obstacle = obstacle
            
    #         obstacle.unit_dir = normalize(vertices[(0 if i == len(vertices) - 1 else i + 1)] - vertices[i])

    #         if len(vertices) == 2:
    #             obstacle.is_convex = True
    #         else:
    #             obstacle.is_convex = (leftOf(vertices[(len(vertices) - 1 if i == 0 else i - 1)], vertices[i], vertices[(0 if i == len(vertices) - 1 else i + 1)]) >= 0.0)
            
    #         obstacle.id = len(self.obstacles)
    #         self.obstacles.append(obstacle)
        
    #     return obstacle_no

    def check_positions(self):
        pos_dict = {i: agent.position.to_array() for i, agent in enumerate(self.agents)}
        self.position_list.append(pos_dict)
    
    def init_relax_times(self):
        self.relax_times = {i:0 for i in range(len(self.agents))}

    def init_arrive_dict(self):
        self.arrive_dict = {i:False for i in range(len(self.agents))}

    def check_collisions(self):
        self.kd_tree.build_agent_tree()
        max_radius = 0.0
        for agent in self.agents:
            if agent.radius > max_radius:
                max_radius = agent.radius
        
        collisions = []
        for agent in self.agents:
            range_sq = sqr(agent.radius + max_radius)
            potential_collisions = []
            self.kd_tree.query_potential_collisions(agent, range_sq, potential_collisions)
            
            for other in potential_collisions:
                if agent.id < other.id:
                    dist_sq = abs_sq(agent.position - other.position)
                    if dist_sq < sqr(agent.radius + other.radius):
                        collisions.append((agent.id, other.id))
        return collisions

    def do_step(self):
        self.kd_tree.build_agent_tree()
        velocity_dict = {i:0.0 for i in range(len(self.agents))}
        for i, agent in enumerate(self.agents):
            if not self.arrive_dict[i]:
                agent._update_pref_velocity()
                agent._compute_neighbors()
                agent._deterministic_orca_lines()
                status = agent.compute_new_velocity_copt()
                self.relax_times[agent.id] += status
                velocity_dict[agent.id] = agent.new_velocity
        self.velocity_list.append(velocity_dict)
        
        # for i in range(len(self.agents_on_the_way)-1, -1, -1):
        for i, agent in enumerate(self.agents):
            if not self.arrive_dict[i]:
                agent.update()
                if agent._check_arrival():
                    self.arrive_dict[i] = True
        self.check_positions()
        self.global_time += self.time_step
        if all(self.arrive_dict.values()):
            print("All agents have reached their goals.")
            return True
        return False

    def do_step_point_estimation(self, sample_budget=10):
        self.kd_tree.build_agent_tree()
        velocity_dict = {i:0.0 for i in range(len(self.agents))}
        for i, agent in enumerate(self.agents_on_the_way):
            agent._update_pref_velocity()
            agent._compute_neighbors()
            # agent._deterministic_orca_lines()
            agent._orca_lines_point_estimation(sample_budget)
            status = agent.compute_new_velocity_copt()
            self.relax_times[agent.id] += status
            velocity_dict[agent.id] = agent.new_velocity.to_array()
        self.velocity_list.append(velocity_dict)
        
        for i in range(len(self.agents_on_the_way)-1, -1, -1):
            agent = self.agents_on_the_way[i]
            agent.update()
            if agent._check_arrival():
                self.agents_on_the_way.pop(i)
        self.check_positions()
        self.global_time += self.time_step
        if len(self.agents_on_the_way) == 0:
            print("All agents have reached their goals.")
            return True
        return False

    def do_step_saa(self, sample_budget=10):
        self.kd_tree.build_agent_tree()
        velocity_dict = {i:0.0 for i in range(len(self.agents))}
        for i, agent in enumerate(self.agents_on_the_way):
            agent._update_pref_velocity()
            agent._compute_neighbors()
            # agent._deterministic_orca_lines()
            agent._orca_lines_saa(sample_budget)
            status = agent.compute_new_velocity_copt()
            self.relax_times[agent.id] += status
            velocity_dict[agent.id] = agent.new_velocity.to_array()
        self.velocity_list.append(velocity_dict)
        
        for i in range(len(self.agents_on_the_way)-1, -1, -1):
            agent = self.agents_on_the_way[i]
            agent.update()
            if agent._check_arrival():
                self.agents_on_the_way.pop(i)
        self.check_positions()
        self.global_time += self.time_step
        if len(self.agents_on_the_way) == 0:
            print("All agents have reached their goals.")
            return True
        return False

    def do_step_robust(self, sample_budget=0.15):
        self.kd_tree.build_agent_tree()
        velocity_dict = {i:0.0 for i in range(len(self.agents))}
        for i, agent in enumerate(self.agents_on_the_way):
            agent._update_pref_velocity()
            agent._compute_neighbors()
            # agent._deterministic_orca_lines()
            agent._orca_lines_robust(sample_budget)
            status = agent.compute_new_velocity_copt()
            self.relax_times[agent.id] += status
            velocity_dict[agent.id] = agent.new_velocity.to_array()
        self.velocity_list.append(velocity_dict)
        
        for i in range(len(self.agents_on_the_way)-1, -1, -1):
            agent = self.agents_on_the_way[i]
            agent.update()
            if agent._check_arrival():
                self.agents_on_the_way.pop(i)
        self.check_positions()
        self.global_time += self.time_step
        if len(self.agents_on_the_way) == 0:
            print("All agents have reached their goals.")
            return True
        return False



    def get_agent_agent_neighbor(self, agent_no, neighbor_no):
        return self.agents[agent_no].agent_neighbors[neighbor_no][1].id

    def get_agent_max_neighbors(self, agent_no):
        return self.agents[agent_no].max_neighbors

    def get_agent_max_speed(self, agent_no):
        return self.agents[agent_no].max_speed

    def get_agent_neighbor_dist(self, agent_no):
        return self.agents[agent_no].neighbor_dist

    def get_agent_num_agent_neighbors(self, agent_no):
        return len(self.agents[agent_no].agent_neighbors)

    def get_agent_num_obstacle_neighbors(self, agent_no):
        return len(self.agents[agent_no].obstacle_neighbors)

    def get_agent_num_orca_lines(self, agent_no):
        return len(self.agents[agent_no].orca_lines)

    def get_agent_obstacle_neighbor(self, agent_no, neighbor_no):
        return self.agents[agent_no].obstacle_neighbors[neighbor_no][1].id

    def get_agent_orca_line(self, agent_no, line_no):
        return self.agents[agent_no].orca_lines[line_no]

    def get_agent_position(self, agent_no):
        return self.agents[agent_no].position

    def get_agent_pref_velocity(self, agent_no):
        return self.agents[agent_no].pref_velocity

    def get_agent_radius(self, agent_no):
        return self.agents[agent_no].radius

    def get_agent_time_horizon(self, agent_no):
        return self.agents[agent_no].time_horizon

    def get_agent_time_horizon_obst(self, agent_no):
        return self.agents[agent_no].time_horizon_obst

    def get_agent_velocity(self, agent_no):
        return self.agents[agent_no].velocity

    def get_global_time(self):
        return self.global_time

    def get_num_agents(self):
        return len(self.agents)

    def get_num_obstacle_vertices(self):
        return len(self.obstacles)

    def get_obstacle_vertex(self, vertex_no):
        return self.obstacles[vertex_no].point

    def get_next_obstacle_vertex_no(self, vertex_no):
        return self.obstacles[vertex_no].next_obstacle.id

    def get_prev_obstacle_vertex_no(self, vertex_no):
        return self.obstacles[vertex_no].prev_obstacle.id

    def get_time_step(self):
        return self.time_step

    def process_obstacles(self):
        self.kd_tree.build_obstacle_tree()

    def query_visibility(self, point1, point2, radius=0.0):
        point1 = to_vector(point1)
        point2 = to_vector(point2)
        return self.kd_tree.query_visibility(point1, point2, radius)

    def set_agent_defaults(self, neighbor_dist, max_neighbors, time_horizon, time_horizon_obst, radius, max_speed, velocity=None):
        velocity = to_vector(velocity)
        if self.default_agent is None:
            self.default_agent = Agent(self)
        
        self.default_agent.max_neighbors = max_neighbors
        self.default_agent.max_speed = max_speed
        self.default_agent.neighbor_dist = neighbor_dist
        self.default_agent.radius = radius
        self.default_agent.time_horizon = time_horizon
        self.default_agent.time_horizon_obst = time_horizon_obst
        self.default_agent.velocity = velocity if velocity is not None else Vector2()

    def set_agent_max_neighbors(self, agent_no, max_neighbors):
        self.agents[agent_no].max_neighbors = max_neighbors

    def set_agent_max_speed(self, agent_no, max_speed):
        self.agents[agent_no].max_speed = max_speed

    def set_agent_neighbor_dist(self, agent_no, neighbor_dist):
        self.agents[agent_no].neighbor_dist = neighbor_dist

    def set_agent_position(self, agent_no, position):
        self.agents[agent_no].position = to_vector(position)

    def set_agent_goal_position(self, agent_no, goal_position):
        self.agents[agent_no].goal_position = to_vector(goal_position)

    def set_agent_pref_velocity(self, agent_no, pref_velocity):
        self.agents[agent_no].pref_velocity = to_vector(pref_velocity)

    def set_agent_radius(self, agent_no, radius):
        self.agents[agent_no].radius = radius

    def set_agent_time_horizon(self, agent_no, time_horizon):
        self.agents[agent_no].time_horizon = time_horizon

    def set_agent_time_horizon_obst(self, agent_no, time_horizon_obst):
        self.agents[agent_no].time_horizon_obst = time_horizon_obst

    def set_agent_velocity(self, agent_no, velocity):
        self.agents[agent_no].velocity = to_vector(velocity)

    def set_time_step(self, time_step):
        self.time_step = time_step

    def compute_total_distance_to_goals(self):
        total_distance = 0.0
        for agent in self.agents:
            total_distance += abs_vector(agent.position - agent.goal_position)
        return total_distance

    # Aliases for compatibility
    addAgent = add_agent
    # addObstacle = add_obstacle
    doStep = do_step
    getAgentAgentNeighbor = get_agent_agent_neighbor
    getAgentMaxNeighbors = get_agent_max_neighbors
    getAgentMaxSpeed = get_agent_max_speed
    getAgentNeighborDist = get_agent_neighbor_dist
    getAgentNumAgentNeighbors = get_agent_num_agent_neighbors
    getAgentNumObstacleNeighbors = get_agent_num_obstacle_neighbors
    getAgentNumORCALines = get_agent_num_orca_lines
    getAgentObstacleNeighbor = get_agent_obstacle_neighbor
    getAgentORCALine = get_agent_orca_line
    
    def getAgentPosition(self, agent_no):
        pos = self.agents[agent_no].position
        return (pos.x, pos.y)

    def getAgentPrefVelocity(self, agent_no):
        vel = self.agents[agent_no].pref_velocity
        return (vel.x, vel.y)

    getAgentRadius = get_agent_radius
    getAgentTimeHorizon = get_agent_time_horizon
    getAgentTimeHorizonObst = get_agent_time_horizon_obst
    
    def getAgentVelocity(self, agent_no):
        vel = self.agents[agent_no].velocity
        return (vel.x, vel.y)

    getGlobalTime = get_global_time
    getNumAgents = get_num_agents
    getNumObstacleVertices = get_num_obstacle_vertices
    
    def getObstacleVertex(self, vertex_no):
        v = self.obstacles[vertex_no].point
        return (v.x, v.y)

    getNextObstacleVertexNo = get_next_obstacle_vertex_no
    getPrevObstacleVertexNo = get_prev_obstacle_vertex_no
    getTimeStep = get_time_step
    processObstacles = process_obstacles
    queryVisibility = query_visibility
    setAgentDefaults = set_agent_defaults
    setAgentMaxNeighbors = set_agent_max_neighbors
    setAgentMaxSpeed = set_agent_max_speed
    setAgentNeighborDist = set_agent_neighbor_dist
    setAgentPosition = set_agent_position
    setAgentPrefVelocity = set_agent_pref_velocity
    setAgentRadius = set_agent_radius
    setAgentTimeHorizon = set_agent_time_horizon
    setAgentTimeHorizonObst = set_agent_time_horizon_obst
    setAgentVelocity = set_agent_velocity
    setTimeStep = set_time_step
