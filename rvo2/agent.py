from .vector import Vector2, abs_sq, det, normalize, abs_vector
from .utils import RVO_EPSILON, distSqPointLineSegment, sqr
from .line import Line
from .random_generator import RandomGenerator
import math
import numpy as np
from coptpy import *
from .agent_point_estimation import AgentPointEstimation
from .agent_saa import AgentSAA
from .agent_robust import AgentRobust

class Agent(AgentPointEstimation, AgentSAA, AgentRobust):
    def __init__(self, sim):
        self.agent_neighbors = [] # list of (distSq, agent)
        self.max_neighbors = 0
        self.max_speed = 0.0
        self.neighbor_dist = 0.0
        self.new_velocity = Vector2()
        self.orca_lines = []
        self.position = Vector2()
        self.goal_position = Vector2()
        self.pref_velocity = Vector2()
        self.radius = 0.0
        self.sim = sim
        self.time_horizon = 0.0
        self.time_horizon_obst = 0.0
        self.velocity = Vector2()
        self.id = 0

        self.env = None
        self.qp = None
        self.varx = None
        self.vary = None

        self.error_generator = None
        # self._init_error_generator((-0.15, 0.15), (-0.15, 0.15), (-0.07, 0.07))
        # self._init_qp_model()

    def _init_error_generator(self, x_range, y_range, mean_range):
        x_mean = np.random.uniform(mean_range[0], mean_range[1])
        # x_mean = np.random.uniform(-0.07, 0.07)
        y_mean = np.random.uniform(mean_range[0], mean_range[1])
        # y_mean = np.random.uniform(-0.07, 0.07)
        self.error_generator = RandomGenerator(x_range, y_range, (x_mean, y_mean))

    def _init_qp_model(self):
        self.env = Envr()
        self.qp = self.env.createModel()
        # self.qp = Model()
        self.qp.setParam('Logging', 0)  # Suppress output
        self.qp.setParam('threads', 1)     # Single thread for deterministic behavior
        self.qp.setParam('TimeLimit', 10)  # Time limit per solve in seconds
        self.varx = self.qp.addVar(lb = -COPT.INFINITY, ub = COPT.INFINITY, vtype=COPT.CONTINUOUS)
        self.vary = self.qp.addVar(lb = -COPT.INFINITY, ub = COPT.INFINITY, vtype=COPT.CONTINUOUS)
        self.qp.addQConstr(self.varx ** 2 + self.vary ** 2 <= self.max_speed * self.max_speed)

    def _check_arrival(self):
        dist_to_goal_sq = abs_sq(self.goal_position - self.position)
        if dist_to_goal_sq < 1e-4:
            return True
        return False

    def _update_pref_velocity(self):
        pref_velocity = (self.goal_position - self.position)/self.sim.time_step
        if abs_sq(pref_velocity) > sqr(self.max_speed):
            pref_velocity = normalize(pref_velocity) * self.max_speed
        self.pref_velocity = pref_velocity

    def _compute_neighbors(self):
        self.agent_neighbors = []
        if self.max_neighbors > 0:
            range_sq = sqr(self.neighbor_dist)
            self.sim.kd_tree.compute_agent_neighbors(self, range_sq)

    def _deterministic_orca_lines(self):
        self.orca_lines = []
        inv_time_horizon = 1.0 / self.time_horizon

        # Create agent ORCA lines
        for dist_sq, other in self.agent_neighbors:
            relative_position = other.position - self.position
            relative_velocity = self.velocity - other.velocity
            dist_sq = abs_sq(relative_position)
            combined_radius = self.radius + other.radius
            combined_radius_sq = sqr(combined_radius)

            line = Line()
            u = Vector2()

            if dist_sq > combined_radius_sq:
                w = relative_velocity - inv_time_horizon * relative_position
                w_length_sq = abs_sq(w)
                dot_product1 = w * relative_position

                if dot_product1 < 0.0 and sqr(dot_product1) > combined_radius_sq * w_length_sq:
                    w_length = math.sqrt(w_length_sq)
                    unit_w = w / w_length
                    line.direction = Vector2(unit_w.y, -unit_w.x)
                    u = (combined_radius * inv_time_horizon - w_length) * unit_w
                else:
                    leg = math.sqrt(dist_sq - combined_radius_sq)
                    if det(relative_position, w) > 0.0:
                        line.direction = Vector2(relative_position.x * leg - relative_position.y * combined_radius, relative_position.x * combined_radius + relative_position.y * leg) / dist_sq
                    else:
                        line.direction = -Vector2(relative_position.x * leg + relative_position.y * combined_radius, -relative_position.x * combined_radius + relative_position.y * leg) / dist_sq
                    
                    dot_product2 = relative_velocity * line.direction
                    u = dot_product2 * line.direction - relative_velocity
            else:
                inv_time_step = 1.0 / self.sim.time_step
                w = relative_velocity - inv_time_step * relative_position
                w_length = abs_vector(w)
                unit_w = w / w_length
                line.direction = Vector2(unit_w.y, -unit_w.x)
                u = (combined_radius * inv_time_step - w_length) * unit_w

            line.point = self.velocity + 0.5 * u
            self.orca_lines.append(line)
        return 0

    def compute_new_velocity_copt(self):
        """
        New implementation using COPT solver to replace linear_program 1, 2, and 3.
        """
        
        # 1. Try to solve the standard RVO problem:
        # Minimize ||v - v_pref||^2 subject to ORCA lines and Max Speed
        success, result_vel = self.solve_copt_qp()
        print('---Agent {}---'.format(self.id), success, result_vel, self.pref_velocity)
        
        if success:
            self.new_velocity = result_vel
            return 0
        else:
            # 2. If infeasible (equivalent to LP3), solve for minimum constraint violation
            # Minimize z (slack) subject to relaxed ORCA lines and Max Speed
            success_relax, result_vel_relax = self.solve_copt_relaxation()
            print('---Agent {} Relaxed---'.format(self.id), success_relax, result_vel_relax)
            if success_relax:
                self.new_velocity = result_vel_relax
                return 1
            # If both fail, keep previous velocity or set to zero (fallback)


    def solve_copt_qp(self):
        constrs = {}
        for i in range(len(self.orca_lines)):
            line = self.orca_lines[i]
            # print(line)
            constr_expr = (-line.direction.y * (self.varx - line.point.x) + line.direction.x * (self.vary - line.point.y))
            constrs[i] = self.qp.addConstr(constr_expr >= 0.0, name=f"orca_line_{i}")
        self.qp.setObjective((self.varx - self.pref_velocity.x) ** 2 + (self.vary - self.pref_velocity.y) ** 2, COPT.MINIMIZE)
        self.qp.solve()
        if self.qp.Status == COPT.OPTIMAL:
            x_val = self.varx.X
            y_val = self.vary.X
            for constr in constrs.values():
                self.qp.remove(constr)
            return True, Vector2(x_val, y_val)
        # elif self.qp.Status == COPT.INFEASIBLE:
        else:
            for constr in constrs.values():
                self.qp.remove(constr)
            return False, None
        # else:
        #     for constr in constrs.values():
        #         self.qp.remove(constr)
        #     raise Exception("COPT optimization failed with status: {}".format(self.qp.Status))
        
    def solve_copt_relaxation(self):
        violation_var = self.qp.addVar(0.0, COPT.INFINITY, vtype=COPT.CONTINUOUS, name="violation")
        self.qp.setObjective(violation_var, COPT.MINIMIZE)
        constrs = {}
        for i in range(len(self.orca_lines)):
            line = self.orca_lines[i]
            constr_expr = (-line.direction.y * (self.varx - line.point.x) + line.direction.x * (self.vary - line.point.y))
            constrs[i] = self.qp.addConstr(constr_expr + violation_var >= 0.0, name=f"relaxed_orca_line_{i}")
        self.qp.solve()
        if self.qp.Status == COPT.OPTIMAL:
            x_val = self.varx.X
            y_val = self.vary.X
            self.qp.remove(violation_var)
            for constr in constrs.values():
                self.qp.remove(constr)
            return True, Vector2(x_val, y_val)
        else:
            self.qp.remove(violation_var)
            for constr in constrs.values():
                self.qp.remove(constr)
            raise Exception("COPT relaxation optimization failed with status: {}".format(self.qp.Status))



    # def compute_new_velocity(self):
    #     line_fail, self.new_velocity = linear_program2(self.orca_lines, self.max_speed, self.pref_velocity, False, self.new_velocity)

    #     if line_fail < len(self.orca_lines):
    #         # self.new_velocity = linear_program3(self.orca_lines, num_obst_lines, line_fail, self.max_speed, self.new_velocity)
    #         self.new_velocity = linear_program3(self.orca_lines, line_fail, self.max_speed, self.new_velocity)



    def insert_agent_neighbor(self, agent, range_sq):
        if self != agent:
            dist_sq = abs_sq(self.position - agent.position)
            if dist_sq < range_sq:
                if len(self.agent_neighbors) < self.max_neighbors:
                    self.agent_neighbors.append((dist_sq, agent))
                
                # Insert in sorted order by dist_sq
                i = len(self.agent_neighbors) - 1
                while i != 0 and dist_sq < self.agent_neighbors[i-1][0]:
                    self.agent_neighbors[i] = self.agent_neighbors[i-1]
                    i -= 1
                self.agent_neighbors[i] = (dist_sq, agent)

                if len(self.agent_neighbors) == self.max_neighbors:
                    range_sq = self.agent_neighbors[-1][0]
        return range_sq

    def insert_obstacle_neighbor(self, obstacle, range_sq):
        next_obstacle = obstacle.next_obstacle
        dist_sq = distSqPointLineSegment(obstacle.point, next_obstacle.point, self.position)

        if dist_sq < range_sq:
            self.obstacle_neighbors.append((dist_sq, obstacle))
            
            i = len(self.obstacle_neighbors) - 1
            while i != 0 and dist_sq < self.obstacle_neighbors[i-1][0]:
                self.obstacle_neighbors[i] = self.obstacle_neighbors[i-1]
                i -= 1
            self.obstacle_neighbors[i] = (dist_sq, obstacle)
        return range_sq

    def update(self):
        self.velocity = self.new_velocity
        self.position += self.velocity * self.sim.time_step


def linear_program1(lines, line_no, radius, opt_velocity, direction_opt, result):
    dot_product = lines[line_no].point * lines[line_no].direction
    discriminant = sqr(dot_product) + sqr(radius) - abs_sq(lines[line_no].point)

    if discriminant < 0.0:
        return False, result

    sqrt_discriminant = math.sqrt(discriminant)
    t_left = -dot_product - sqrt_discriminant
    t_right = -dot_product + sqrt_discriminant

    for i in range(line_no):
        denominator = det(lines[line_no].direction, lines[i].direction)
        numerator = det(lines[i].direction, lines[line_no].point - lines[i].point)

        if abs(denominator) <= RVO_EPSILON:
            if numerator < 0.0:
                return False, result
            else:
                continue

        t = numerator / denominator

        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)

        if t_left > t_right:
            return False, result

    if direction_opt:
        if opt_velocity * lines[line_no].direction > 0.0:
            result = lines[line_no].point + t_right * lines[line_no].direction
        else:
            result = lines[line_no].point + t_left * lines[line_no].direction
    else:
        t = lines[line_no].direction * (opt_velocity - lines[line_no].point)

        if t < t_left:
            result = lines[line_no].point + t_left * lines[line_no].direction
        elif t > t_right:
            result = lines[line_no].point + t_right * lines[line_no].direction
        else:
            result = lines[line_no].point + t * lines[line_no].direction

    return True, result

# line_fail, self.new_velocity = linear_program2(self.orca_lines, self.max_speed, self.pref_velocity, False, self.new_velocity)
def linear_program2(lines, radius, opt_velocity, direction_opt, result):
    if direction_opt:  # If an optimal direction is to be taken
        result = opt_velocity * radius
    elif abs_sq(opt_velocity) > sqr(radius): # If the optimal velocity is larger than the max speed
        result = normalize(opt_velocity) * radius
    else:   # Otherwise, the optimal velocity is within the max speed. A trivial value is (0.0, 0.0)
        result = opt_velocity

    for i in range(len(lines)):
        if det(lines[i].direction, lines[i].point - result) > 0.0:
            temp_result = result
            success, result = linear_program1(lines, i, radius, opt_velocity, direction_opt, result)
            if not success:
                result = temp_result
                return i, result

    return len(lines), result

# if line_fail < len(self.orca_lines):
#     self.new_velocity = linear_program3(self.orca_lines, num_obst_lines, line_fail, self.max_speed, self.new_velocity)
def linear_program3(lines, begin_line, radius, result):
    distance = 0.0

    for i in range(begin_line, len(lines)):
        if det(lines[i].direction, lines[i].point - result) > distance:
            # proj_lines = lines[:num_obst_lines]
            proj_lines = []
            # for j in range(num_obst_lines, i):
            for j in range(i):
                line = Line()
                determinant = det(lines[i].direction, lines[j].direction)

                if abs(determinant) <= RVO_EPSILON:
                    if lines[i].direction * lines[j].direction > 0.0:
                        continue
                    else:
                        line.point = 0.5 * (lines[i].point + lines[j].point)
                else:
                    line.point = lines[i].point + (det(lines[j].direction, lines[i].point - lines[j].point) / determinant) * lines[i].direction

                line.direction = normalize(lines[j].direction - lines[i].direction)
                proj_lines.append(line)

            temp_result = result
            count, result = linear_program2(proj_lines, radius, Vector2(-lines[i].direction.y, lines[i].direction.x), True, result)
            if count < len(proj_lines):
                result = temp_result

            distance = det(lines[i].direction, lines[i].point - result)
    return result
