from .vector import Vector2, abs_sq, det, normalize, abs_vector
from .utils import RVO_EPSILON, distSqPointLineSegment, sqr
from .line import Line
from .random_generator import RandomGenerator
import math
import numpy as np
from coptpy import *

class AgentRobust:
    def _orca_lines_robust(self, budget_radius):
        self.orca_lines = []
        inv_time_horizon = 1.0 / self.time_horizon

        # Create agent ORCA lines
        for dist_sq, other in self.agent_neighbors:
            relative_position = other.position - self.position
            relative_velocity = self.velocity - other.velocity
            dist_sq = abs_sq(relative_position)
            combined_radius = self.radius + other.radius
            combined_radius_sq = sqr(combined_radius)

            collide_velocity = (relative_position) * inv_time_horizon
            combined_v_radius = combined_radius * inv_time_horizon

            line = Line()
            u = Vector2()

            if dist_sq > combined_radius_sq:
                w = relative_velocity - collide_velocity
                # w_direction = normalize(w)
                w_length_sq = abs_sq(w)
                dot_product1 = w * relative_position

                if dot_product1 < 0.0 and sqr(dot_product1) > combined_radius_sq * w_length_sq:
                    w_length = math.sqrt(w_length_sq)
                    unit_w = w / w_length
                    line.direction = Vector2(unit_w.y, -unit_w.x)
                    u = (combined_radius * inv_time_horizon - w_length) * unit_w

                    # Adjust u based on budget_radius, and obtain the robust ORCA line
                    v_minus_w = relative_velocity - (budget_radius * unit_w)
                    v_plus_w = relative_velocity + (budget_radius * unit_w)
                    closest_dis = abs_vector(v_minus_w - collide_velocity)
                    if closest_dis <= combined_v_radius:
                        u += budget_radius * unit_w
                    else:
                        u -= budget_radius * unit_w
                else:
                    leg = math.sqrt(dist_sq - combined_radius_sq)
                    if det(relative_position, w) > 0.0:
                        line.direction = Vector2(relative_position.x * leg - relative_position.y * combined_radius, relative_position.x * combined_radius + relative_position.y * leg) / dist_sq
                    else:
                        line.direction = -Vector2(relative_position.x * leg + relative_position.y * combined_radius, -relative_position.x * combined_radius + relative_position.y * leg) / dist_sq
                    # Adjust u based on budget_radius, and obtain the robust ORCA line
                    normal_direction = Vector2(-line.direction.y, line.direction.x)
                    most_dangerous_point = relative_velocity - budget_radius * normal_direction
                    dot_product2 = most_dangerous_point * line.direction
                    u = dot_product2 * line.direction - most_dangerous_point
                    # dot_product2 = relative_velocity * line.direction
                    # u = dot_product2 * line.direction - relative_velocity
            else:
                inv_time_step = 1.0 / self.sim.time_step
                w = relative_velocity - inv_time_step * relative_position
                w_length = abs_vector(w)
                unit_w = w / w_length
                line.direction = Vector2(unit_w.y, -unit_w.x)
                u = (combined_radius * inv_time_step - w_length) * unit_w
                # Adjust u based on budget_radius, and obtain the robust ORCA line
                most_dangerous_point = relative_velocity - budget_radius * unit_w
                dist_to_collide = abs_vector(most_dangerous_point - (inv_time_step * relative_position))
                if dist_to_collide <= (combined_radius * inv_time_step):
                    u += budget_radius * unit_w
                else:
                    u -= budget_radius * unit_w

            line.point = self.velocity + 0.5 * u
            self.orca_lines.append(line)
        return 0

