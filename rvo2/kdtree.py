from .vector import Vector2, abs_sq, det
from .utils import RVO_EPSILON, leftOf, sqr
from .obstacle import Obstacle

MAX_LEAF_SIZE = 10

class AgentTreeNode:
    def __init__(self):
        self.begin = 0
        self.end = 0
        self.left = 0
        self.max_x = 0.0
        self.max_y = 0.0
        self.min_x = 0.0
        self.min_y = 0.0
        self.right = 0

class ObstacleTreeNode:
    def __init__(self):
        self.left = None
        self.obstacle = None
        self.right = None

class KdTree:
    def __init__(self, sim):
        self.sim = sim
        self.agents = []
        self.agent_tree = [] # List of AgentTreeNode
        self.obstacle_tree = None # ObstacleTreeNode

    def build_agent_tree(self):
        if len(self.agents) < len(self.sim.agents):
            self.agents = list(self.sim.agents) # Copy
            self.agent_tree = [AgentTreeNode() for _ in range(2 * len(self.agents) - 1)]
        
        if self.agents:
            self.build_agent_tree_recursive(0, len(self.agents), 0)

    def build_agent_tree_recursive(self, begin, end, node):
        self.agent_tree[node].begin = begin
        self.agent_tree[node].end = end
        self.agent_tree[node].min_x = self.agent_tree[node].max_x = self.agents[begin].position.x
        self.agent_tree[node].min_y = self.agent_tree[node].max_y = self.agents[begin].position.y

        for i in range(begin + 1, end):
            self.agent_tree[node].max_x = max(self.agent_tree[node].max_x, self.agents[i].position.x)
            self.agent_tree[node].min_x = min(self.agent_tree[node].min_x, self.agents[i].position.x)
            self.agent_tree[node].max_y = max(self.agent_tree[node].max_y, self.agents[i].position.y)
            self.agent_tree[node].min_y = min(self.agent_tree[node].min_y, self.agents[i].position.y)

        if end - begin > MAX_LEAF_SIZE:
            is_vertical = (self.agent_tree[node].max_x - self.agent_tree[node].min_x > self.agent_tree[node].max_y - self.agent_tree[node].min_y)
            split_value = 0.5 * (self.agent_tree[node].max_x + self.agent_tree[node].min_x) if is_vertical else 0.5 * (self.agent_tree[node].max_y + self.agent_tree[node].min_y)

            left = begin
            right = end

            while left < right:
                # The maximum index of the left subtree
                while left < right and (self.agents[left].position.x if is_vertical else self.agents[left].position.y) < split_value:
                    left += 1
                # The minimum index of the right subtree
                while right > left and (self.agents[right - 1].position.x if is_vertical else self.agents[right - 1].position.y) >= split_value:
                    right -= 1
                
                # The agents are not sorted, swap them
                if left < right:
                    self.agents[left], self.agents[right - 1] = self.agents[right - 1], self.agents[left]
                    left += 1
                    right -= 1
            
            if left == begin:
                left += 1
                right += 1
            
            self.agent_tree[node].left = node + 1
            self.agent_tree[node].right = node + 2 * (left - begin)

            self.build_agent_tree_recursive(begin, left, self.agent_tree[node].left)
            self.build_agent_tree_recursive(left, end, self.agent_tree[node].right)

    def build_obstacle_tree(self):
        self.obstacle_tree = None # Python GC handles deletion
        obstacles = list(self.sim.obstacles)
        self.obstacle_tree = self.build_obstacle_tree_recursive(obstacles)

    def build_obstacle_tree_recursive(self, obstacles):
        if not obstacles:
            return None
        
        node = ObstacleTreeNode()
        
        optimal_split = 0
        # History of best split, the number of left and right
        # smaller is the max value of left and right, more balance is the tree, and better
        min_left = len(obstacles)
        min_right = len(obstacles)

        for i in range(len(obstacles)):
            left_size = 0
            right_size = 0

            obstacle_i1 = obstacles[i]
            obstacle_i2 = obstacle_i1.next_obstacle

            for j in range(len(obstacles)):
                if i == j:
                    continue
                
                obstacle_j1 = obstacles[j]
                obstacle_j2 = obstacle_j1.next_obstacle

                j1_left_of_i = leftOf(obstacle_i1.point, obstacle_i2.point, obstacle_j1.point)
                j2_left_of_i = leftOf(obstacle_i1.point, obstacle_i2.point, obstacle_j2.point)

                if j1_left_of_i >= -RVO_EPSILON and j2_left_of_i >= -RVO_EPSILON:
                    left_size += 1
                elif j1_left_of_i <= RVO_EPSILON and j2_left_of_i <= RVO_EPSILON:
                    right_size += 1
                else:
                    left_size += 1
                    right_size += 1
                
                # Early exit if this split is worse than the best found so far
                if max(left_size, right_size) >= max(min_left, min_right) and min(left_size, right_size) >= min(min_left, min_right):
                    break
            
            if max(left_size, right_size) < max(min_left, min_right) or (max(left_size, right_size) == max(min_left, min_right) and min(left_size, right_size) < min(min_left, min_right)):
                min_left = left_size
                min_right = right_size
                optimal_split = i
        
        left_obstacles = []
        right_obstacles = []

        i = optimal_split
        obstacle_i1 = obstacles[i]
        obstacle_i2 = obstacle_i1.next_obstacle

        for j in range(len(obstacles)):
            if i == j:
                continue
            
            obstacle_j1 = obstacles[j]
            obstacle_j2 = obstacle_j1.next_obstacle

            # Determine which side of i1-i2 the j1 and j2 points are
            # nonnegative if on the left side
            j1_left_of_i = leftOf(obstacle_i1.point, obstacle_i2.point, obstacle_j1.point)
            j2_left_of_i = leftOf(obstacle_i1.point, obstacle_i2.point, obstacle_j2.point)
            
            # Both points on left side or right side
            # if j1_left_of_i >= -RVO_EPSILON and j2_left_of_i >= -RVO_EPSILON:
            if j1_left_of_i >= 0.0 and j2_left_of_i >= 0.0:
                left_obstacles.append(obstacles[j])
            # elif j1_left_of_i <= RVO_EPSILON and j2_left_of_i <= RVO_EPSILON:
            elif j1_left_of_i <= 0.0 and j2_left_of_i <= 0.0:
                right_obstacles.append(obstacles[j])
            # If the obstacle straddles the line, split it
            else:
                t = det(obstacle_i2.point - obstacle_i1.point, obstacle_j1.point - obstacle_i1.point) / det(obstacle_i2.point - obstacle_i1.point, obstacle_j1.point - obstacle_j2.point)

                split_point = obstacle_j1.point + t * (obstacle_j2.point - obstacle_j1.point)

                new_obstacle = Obstacle()
                new_obstacle.point = split_point
                new_obstacle.prev_obstacle = obstacle_j1
                new_obstacle.next_obstacle = obstacle_j2
                new_obstacle.is_convex = True
                new_obstacle.unit_dir = obstacle_j1.unit_dir
                new_obstacle.id = len(self.sim.obstacles)

                self.sim.obstacles.append(new_obstacle)

                obstacle_j1.next_obstacle = new_obstacle
                obstacle_j2.prev_obstacle = new_obstacle

                if j1_left_of_i > 0.0:
                    left_obstacles.append(obstacle_j1)
                    right_obstacles.append(new_obstacle)
                else:
                    right_obstacles.append(obstacle_j1)
                    left_obstacles.append(new_obstacle)

        node.obstacle = obstacle_i1
        node.left = self.build_obstacle_tree_recursive(left_obstacles)
        node.right = self.build_obstacle_tree_recursive(right_obstacles)
        return node

    def compute_agent_neighbors(self, agent, range_sq):
        self.query_agent_tree_recursive(agent, range_sq, 0)

    def compute_obstacle_neighbors(self, agent, range_sq):
        self.query_obstacle_tree_recursive(agent, range_sq, self.obstacle_tree)

    def query_agent_tree_recursive(self, agent, range_sq, node): # The node is the index of agent_tree
        # Find the agents within range_sq of the agent
        # and then insert them into the agent's neighbor list
        if self.agent_tree[node].end - self.agent_tree[node].begin <= MAX_LEAF_SIZE:
            for i in range(self.agent_tree[node].begin, self.agent_tree[node].end):
                range_sq = agent.insert_agent_neighbor(self.agents[i], range_sq)
        else:
            dist_sq_left = sqr(max(0.0, self.agent_tree[self.agent_tree[node].left].min_x - agent.position.x)) + \
                           sqr(max(0.0, agent.position.x - self.agent_tree[self.agent_tree[node].left].max_x)) + \
                           sqr(max(0.0, self.agent_tree[self.agent_tree[node].left].min_y - agent.position.y)) + \
                           sqr(max(0.0, agent.position.y - self.agent_tree[self.agent_tree[node].left].max_y))
            
            dist_sq_right = sqr(max(0.0, self.agent_tree[self.agent_tree[node].right].min_x - agent.position.x)) + \
                            sqr(max(0.0, agent.position.x - self.agent_tree[self.agent_tree[node].right].max_x)) + \
                            sqr(max(0.0, self.agent_tree[self.agent_tree[node].right].min_y - agent.position.y)) + \
                            sqr(max(0.0, agent.position.y - self.agent_tree[self.agent_tree[node].right].max_y))

            if dist_sq_left < dist_sq_right:
                if dist_sq_left < range_sq:
                    range_sq = self.query_agent_tree_recursive(agent, range_sq, self.agent_tree[node].left)
                    if dist_sq_right < range_sq:
                        range_sq = self.query_agent_tree_recursive(agent, range_sq, self.agent_tree[node].right)
            else:
                if dist_sq_right < range_sq:
                    range_sq = self.query_agent_tree_recursive(agent, range_sq, self.agent_tree[node].right)
                    if dist_sq_left < range_sq:
                        range_sq = self.query_agent_tree_recursive(agent, range_sq, self.agent_tree[node].left)
        return range_sq

    def query_obstacle_tree_recursive(self, agent, range_sq, node): # The node is an ObstacleTreeNode
        if node is None:
            return range_sq
        
        obstacle1 = node.obstacle
        obstacle2 = obstacle1.next_obstacle

        agent_left_of_line = leftOf(obstacle1.point, obstacle2.point, agent.position)

        range_sq = self.query_obstacle_tree_recursive(agent, range_sq, node.left if agent_left_of_line >= 0.0 else node.right)

        # Squared distance from agent to obstacle line
        dist_sq_line = sqr(agent_left_of_line) / abs_sq(obstacle2.point - obstacle1.point)

        if dist_sq_line < range_sq:
            if agent_left_of_line < 0.0:
                range_sq = agent.insert_obstacle_neighbor(node.obstacle, range_sq)
            
            range_sq = self.query_obstacle_tree_recursive(agent, range_sq, node.right if agent_left_of_line >= 0.0 else node.left)
        return range_sq

    # True if q1 and q2 are visible to each other within radius
    def query_visibility(self, q1, q2, radius):
        return self.query_visibility_recursive(q1, q2, radius, self.obstacle_tree)

    # Recursively dicide wehter q1 and q2 are visible to each considering each 
    # segment of obstacles.
    def query_visibility_recursive(self, q1, q2, radius, node):
        if node is None:
            return True
        
        obstacle1 = node.obstacle
        obstacle2 = obstacle1.next_obstacle

        q1_left_of_i = leftOf(obstacle1.point, obstacle2.point, q1)
        q2_left_of_i = leftOf(obstacle1.point, obstacle2.point, q2)
        inv_length_i = 1.0 / abs_sq(obstacle2.point - obstacle1.point)

        # Both q1 and q2 are on the left side of the obstacle line
        if q1_left_of_i >= 0.0 and q2_left_of_i >= 0.0:
            # Visible if both are sufficiently far from the line (so that not affected by the obstacles in the right subtree),
            # or recursively check the right subtree
            return self.query_visibility_recursive(q1, q2, radius, node.left) and \
                   ((sqr(q1_left_of_i) * inv_length_i >= sqr(radius) and sqr(q2_left_of_i) * inv_length_i >= sqr(radius)) or \
                    self.query_visibility_recursive(q1, q2, radius, node.right))
        elif q1_left_of_i <= 0.0 and q2_left_of_i <= 0.0:
            # Visible if both are sufficiently far from the line (so that not affected by the obstacles in the left subtree), 
            # or recursively check the left subtree
            return self.query_visibility_recursive(q1, q2, radius, node.right) and \
                   ((sqr(q1_left_of_i) * inv_length_i >= sqr(radius) and sqr(q2_left_of_i) * inv_length_i >= sqr(radius)) or \
                    self.query_visibility_recursive(q1, q2, radius, node.left))
        # One point on each side of the obstacle line
        elif q1_left_of_i >= 0.0 and q2_left_of_i <= 0.0:
            # Recursively check both subtrees
            return self.query_visibility_recursive(q1, q2, radius, node.left) and self.query_visibility_recursive(q1, q2, radius, node.right)
        else:
            point1_left_of_q = leftOf(q1, q2, obstacle1.point)
            point2_left_of_q = leftOf(q1, q2, obstacle2.point)
            inv_length_q = 1.0 / abs_sq(q2 - q1)
            return (point1_left_of_q * point2_left_of_q >= 0.0 and \
                    sqr(point1_left_of_q) * inv_length_q > sqr(radius) and \
                    sqr(point2_left_of_q) * inv_length_q > sqr(radius) and \
                    self.query_visibility_recursive(q1, q2, radius, node.left) and \
                    self.query_visibility_recursive(q1, q2, radius, node.right))

    def query_potential_collisions(self, agent, range_sq, results):
        self.query_potential_collisions_recursive(agent, range_sq, 0, results)

    def query_potential_collisions_recursive(self, agent, range_sq, node, results):
        if self.agent_tree[node].end - self.agent_tree[node].begin <= MAX_LEAF_SIZE:
            for i in range(self.agent_tree[node].begin, self.agent_tree[node].end):
                if self.agents[i].id != agent.id:
                    dist_sq = abs_sq(self.agents[i].position - agent.position)
                    if dist_sq < range_sq:
                        results.append(self.agents[i])
        else:
            dist_sq_left = sqr(max(0.0, self.agent_tree[self.agent_tree[node].left].min_x - agent.position.x)) + \
                           sqr(max(0.0, agent.position.x - self.agent_tree[self.agent_tree[node].left].max_x)) + \
                           sqr(max(0.0, self.agent_tree[self.agent_tree[node].left].min_y - agent.position.y)) + \
                           sqr(max(0.0, agent.position.y - self.agent_tree[self.agent_tree[node].left].max_y))
            
            dist_sq_right = sqr(max(0.0, self.agent_tree[self.agent_tree[node].right].min_x - agent.position.x)) + \
                            sqr(max(0.0, agent.position.x - self.agent_tree[self.agent_tree[node].right].max_x)) + \
                            sqr(max(0.0, self.agent_tree[self.agent_tree[node].right].min_y - agent.position.y)) + \
                            sqr(max(0.0, agent.position.y - self.agent_tree[self.agent_tree[node].right].max_y))

            if dist_sq_left < range_sq:
                self.query_potential_collisions_recursive(agent, range_sq, self.agent_tree[node].left, results)
            
            if dist_sq_right < range_sq:
                self.query_potential_collisions_recursive(agent, range_sq, self.agent_tree[node].right, results)

