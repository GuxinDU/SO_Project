from .vector import Vector2, abs_sq, det, normalize, abs_vector
from .utils import RVO_EPSILON, distSqPointLineSegment, sqr
from .line import Line
import math

class Agent:
    def __init__(self, sim):
        self.agent_neighbors = [] # list of (distSq, agent)
        self.max_neighbors = 0
        self.max_speed = 0.0
        self.neighbor_dist = 0.0
        self.new_velocity = Vector2()
        self.obstacle_neighbors = [] # list of (distSq, obstacle)
        self.orca_lines = []
        self.position = Vector2()
        self.pref_velocity = Vector2()
        self.radius = 0.0
        self.sim = sim
        self.time_horizon = 0.0
        self.time_horizon_obst = 0.0
        self.velocity = Vector2()
        self.id = 0

    def compute_neighbors(self):
        self.obstacle_neighbors = []
        range_sq = sqr(self.time_horizon_obst * self.max_speed + self.radius)
        self.sim.kd_tree.compute_obstacle_neighbors(self, range_sq)

        self.agent_neighbors = []
        if self.max_neighbors > 0:
            range_sq = sqr(self.neighbor_dist)
            self.sim.kd_tree.compute_agent_neighbors(self, range_sq)

    def compute_new_velocity(self):
        self.orca_lines = []
        
        inv_time_horizon_obst = 1.0 / self.time_horizon_obst

        # Create obstacle ORCA lines
        for dist_sq, obstacle1 in self.obstacle_neighbors:
            obstacle2 = obstacle1.next_obstacle

            relative_position1 = obstacle1.point - self.position
            relative_position2 = obstacle2.point - self.position

            # Check if velocity obstacle of obstacle is already taken care of by
            # previously constructed obstacle ORCA lines
            already_covered = False
            for line in self.orca_lines:
                if det(inv_time_horizon_obst * relative_position1 - line.point, line.direction) - inv_time_horizon_obst * self.radius >= -RVO_EPSILON and \
                   det(inv_time_horizon_obst * relative_position2 - line.point, line.direction) - inv_time_horizon_obst * self.radius >= -RVO_EPSILON:
                    already_covered = True
                    break
            
            if already_covered:
                continue

            # Not yet covered. Check for collisions.
            dist_sq1 = abs_sq(relative_position1)
            dist_sq2 = abs_sq(relative_position2)
            radius_sq = sqr(self.radius)

            obstacle_vector = obstacle2.point - obstacle1.point
            s = (-relative_position1 * obstacle_vector) / abs_sq(obstacle_vector)
            dist_sq_line = abs_sq(-relative_position1 - s * obstacle_vector)

            line = Line()

            if s < 0.0 and dist_sq1 <= radius_sq:
                if obstacle1.is_convex:
                    line.point = Vector2(0.0, 0.0)
                    line.direction = normalize(Vector2(-relative_position1.y, relative_position1.x))
                    self.orca_lines.append(line)
                continue
            elif s > 1.0 and dist_sq2 <= radius_sq:
                if obstacle2.is_convex and det(relative_position2, obstacle2.unit_dir) >= 0.0:
                    line.point = Vector2(0.0, 0.0)
                    line.direction = normalize(Vector2(-relative_position2.y, relative_position2.x))
                    self.orca_lines.append(line)
                continue
            elif s >= 0.0 and s < 1.0 and dist_sq_line <= radius_sq:
                line.point = Vector2(0.0, 0.0)
                line.direction = -obstacle1.unit_dir
                self.orca_lines.append(line)
                continue

            # No collision.
            # Compute legs.
            left_leg_direction = Vector2()
            right_leg_direction = Vector2()

            if s < 0.0 and dist_sq_line <= radius_sq:
                if not obstacle1.is_convex:
                    continue
                
                obstacle2 = obstacle1
                leg1 = math.sqrt(dist_sq1 - radius_sq)
                left_leg_direction = Vector2(relative_position1.x * leg1 - relative_position1.y * self.radius, relative_position1.x * self.radius + relative_position1.y * leg1) / dist_sq1
                right_leg_direction = Vector2(relative_position1.x * leg1 + relative_position1.y * self.radius, -relative_position1.x * self.radius + relative_position1.y * leg1) / dist_sq1
            elif s > 1.0 and dist_sq_line <= radius_sq:
                if not obstacle2.is_convex:
                    continue
                
                obstacle1 = obstacle2
                leg2 = math.sqrt(dist_sq2 - radius_sq)
                left_leg_direction = Vector2(relative_position2.x * leg2 - relative_position2.y * self.radius, relative_position2.x * self.radius + relative_position2.y * leg2) / dist_sq2
                right_leg_direction = Vector2(relative_position2.x * leg2 + relative_position2.y * self.radius, -relative_position2.x * self.radius + relative_position2.y * leg2) / dist_sq2
            else:
                if obstacle1.is_convex:
                    leg1 = math.sqrt(dist_sq1 - radius_sq)
                    left_leg_direction = Vector2(relative_position1.x * leg1 - relative_position1.y * self.radius, relative_position1.x * self.radius + relative_position1.y * leg1) / dist_sq1
                else:
                    left_leg_direction = -obstacle1.unit_dir
                
                if obstacle2.is_convex:
                    leg2 = math.sqrt(dist_sq2 - radius_sq)
                    right_leg_direction = Vector2(relative_position2.x * leg2 + relative_position2.y * self.radius, -relative_position2.x * self.radius + relative_position2.y * leg2) / dist_sq2
                else:
                    right_leg_direction = obstacle1.unit_dir

            left_neighbor = obstacle1.prev_obstacle
            is_left_leg_foreign = False
            is_right_leg_foreign = False

            if obstacle1.is_convex and det(left_leg_direction, -left_neighbor.unit_dir) >= 0.0:
                left_leg_direction = -left_neighbor.unit_dir
                is_left_leg_foreign = True

            if obstacle2.is_convex and det(right_leg_direction, obstacle2.unit_dir) <= 0.0:
                right_leg_direction = obstacle2.unit_dir
                is_right_leg_foreign = True

            left_cutoff = inv_time_horizon_obst * (obstacle1.point - self.position)
            right_cutoff = inv_time_horizon_obst * (obstacle2.point - self.position)
            cutoff_vec = right_cutoff - left_cutoff

            t = 0.5 if obstacle1 == obstacle2 else ((self.velocity - left_cutoff) * cutoff_vec) / abs_sq(cutoff_vec)
            t_left = (self.velocity - left_cutoff) * left_leg_direction
            t_right = (self.velocity - right_cutoff) * right_leg_direction

            if (t < 0.0 and t_left < 0.0) or (obstacle1 == obstacle2 and t_left < 0.0 and t_right < 0.0):
                unit_w = normalize(self.velocity - left_cutoff)
                line.direction = Vector2(unit_w.y, -unit_w.x)
                line.point = left_cutoff + self.radius * inv_time_horizon_obst * unit_w
                self.orca_lines.append(line)
                continue
            elif t > 1.0 and t_right < 0.0:
                unit_w = normalize(self.velocity - right_cutoff)
                line.direction = Vector2(unit_w.y, -unit_w.x)
                line.point = right_cutoff + self.radius * inv_time_horizon_obst * unit_w
                self.orca_lines.append(line)
                continue

            dist_sq_cutoff = float('inf') if (t < 0.0 or t > 1.0 or obstacle1 == obstacle2) else abs_sq(self.velocity - (left_cutoff + t * cutoff_vec))
            dist_sq_left = float('inf') if t_left < 0.0 else abs_sq(self.velocity - (left_cutoff + t_left * left_leg_direction))
            dist_sq_right = float('inf') if t_right < 0.0 else abs_sq(self.velocity - (right_cutoff + t_right * right_leg_direction))

            if dist_sq_cutoff <= dist_sq_left and dist_sq_cutoff <= dist_sq_right:
                line.direction = -obstacle1.unit_dir
                line.point = left_cutoff + self.radius * inv_time_horizon_obst * Vector2(-line.direction.y, line.direction.x)
                self.orca_lines.append(line)
                continue
            elif dist_sq_left <= dist_sq_right:
                if is_left_leg_foreign:
                    continue
                line.direction = left_leg_direction
                line.point = left_cutoff + self.radius * inv_time_horizon_obst * Vector2(-line.direction.y, line.direction.x)
                self.orca_lines.append(line)
                continue
            else:
                if is_right_leg_foreign:
                    continue
                line.direction = -right_leg_direction
                line.point = right_cutoff + self.radius * inv_time_horizon_obst * Vector2(-line.direction.y, line.direction.x)
                self.orca_lines.append(line)
                continue

        num_obst_lines = len(self.orca_lines)
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

        line_fail, self.new_velocity = linear_program2(self.orca_lines, self.max_speed, self.pref_velocity, False, self.new_velocity)

        if line_fail < len(self.orca_lines):
            self.new_velocity = linear_program3(self.orca_lines, num_obst_lines, line_fail, self.max_speed, self.new_velocity)

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

def linear_program2(lines, radius, opt_velocity, direction_opt, result):
    if direction_opt:
        result = opt_velocity * radius
    elif abs_sq(opt_velocity) > sqr(radius):
        result = normalize(opt_velocity) * radius
    else:
        result = opt_velocity

    for i in range(len(lines)):
        if det(lines[i].direction, lines[i].point - result) > 0.0:
            temp_result = result
            success, result = linear_program1(lines, i, radius, opt_velocity, direction_opt, result)
            if not success:
                result = temp_result
                return i, result

    return len(lines), result

def linear_program3(lines, num_obst_lines, begin_line, radius, result):
    distance = 0.0

    for i in range(begin_line, len(lines)):
        if det(lines[i].direction, lines[i].point - result) > distance:
            proj_lines = lines[:num_obst_lines]

            for j in range(num_obst_lines, i):
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
