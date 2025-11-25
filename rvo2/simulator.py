from .vector import Vector2, normalize
from .agent import Agent
from .obstacle import Obstacle
from .kdtree import KdTree
from .utils import leftOf

RVO_ERROR = -1

def to_vector(v):
    if isinstance(v, Vector2):
        return v
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return Vector2(v[0], v[1])
    return v

class RVOSimulator:
    def __init__(self, time_step=0.0, neighbor_dist=0.0, max_neighbors=0, time_horizon=0.0, time_horizon_obst=0.0, radius=0.0, max_speed=0.0, velocity=None):
        self.time_step = time_step
        self.agents = []
        self.default_agent = None
        self.global_time = 0.0
        self.kd_tree = KdTree(self)
        self.obstacles = []

        velocity = to_vector(velocity)
        if velocity is None:
            velocity = Vector2()

        if neighbor_dist > 0 or max_neighbors > 0 or time_horizon > 0 or time_horizon_obst > 0 or radius > 0 or max_speed > 0:
             self.default_agent = Agent(self)
             self.default_agent.max_neighbors = max_neighbors
             self.default_agent.max_speed = max_speed
             self.default_agent.neighbor_dist = neighbor_dist
             self.default_agent.radius = radius
             self.default_agent.time_horizon = time_horizon
             self.default_agent.time_horizon_obst = time_horizon_obst
             self.default_agent.velocity = velocity

    def add_agent(self, position, neighbor_dist=None, max_neighbors=None, time_horizon=None, time_horizon_obst=None, radius=None, max_speed=None, velocity=None):
        position = to_vector(position)
        velocity = to_vector(velocity)
        
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
            agent.time_horizon_obst = self.default_agent.time_horizon_obst
            agent.velocity = self.default_agent.velocity
            agent.id = len(self.agents)
            self.agents.append(agent)
            return len(self.agents) - 1
        else:
            agent = Agent(self)
            agent.position = position
            agent.max_neighbors = max_neighbors
            agent.max_speed = max_speed
            agent.neighbor_dist = neighbor_dist
            agent.radius = radius
            agent.time_horizon = time_horizon
            agent.time_horizon_obst = time_horizon_obst
            agent.velocity = velocity if velocity is not None else Vector2()
            agent.id = len(self.agents)
            self.agents.append(agent)
            return len(self.agents) - 1

    def add_obstacle(self, vertices):
        if len(vertices) < 2:
            return RVO_ERROR
        
        vertices = [to_vector(v) for v in vertices]
        
        obstacle_no = len(self.obstacles)

        for i in range(len(vertices)):
            obstacle = Obstacle()
            obstacle.point = vertices[i]

            if i != 0:
                obstacle.prev_obstacle = self.obstacles[-1]
                obstacle.prev_obstacle.next_obstacle = obstacle
            
            if i == len(vertices) - 1:
                obstacle.next_obstacle = self.obstacles[obstacle_no]
                obstacle.next_obstacle.prev_obstacle = obstacle
            
            obstacle.unit_dir = normalize(vertices[(0 if i == len(vertices) - 1 else i + 1)] - vertices[i])

            if len(vertices) == 2:
                obstacle.is_convex = True
            else:
                obstacle.is_convex = (leftOf(vertices[(len(vertices) - 1 if i == 0 else i - 1)], vertices[i], vertices[(0 if i == len(vertices) - 1 else i + 1)]) >= 0.0)
            
            obstacle.id = len(self.obstacles)
            self.obstacles.append(obstacle)
        
        return obstacle_no

    def do_step(self):
        self.kd_tree.build_agent_tree()

        for agent in self.agents:
            agent.compute_neighbors()
            agent.compute_new_velocity()
        
        for agent in self.agents:
            agent.update()
        
        self.global_time += self.time_step

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

    # Aliases for compatibility
    addAgent = add_agent
    addObstacle = add_obstacle
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
