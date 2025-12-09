from .vector import Vector2, abs_sq, det, normalize, abs_vector
from .utils import RVO_EPSILON, distSqPointLineSegment, sqr
from .line import Line
from .random_generator import RandomGenerator
import math
import numpy as np
from coptpy import *

class AgentExtendRadius:
    def _orca_lines_extend_radius(self, budget_radius=0.2):
        """
        Generate ORCA lines using point estimation for uncertain velocities.
        """
        self.orca_lines = []
        inv_time_horizon = 1.0 / self.time_horizon

        # # Sample velocities based on the error generator
        # sampled_velocities = [self.error_generator.sample() for _ in range(sample_budget)]

        for dist_sq, other in self.agent_neighbors:
            relative_position = other.position - self.position
            relative_velocity = self.velocity - other.velocity
            combined_radius = self.radius + other.radius + (budget_radius * self.time_horizon)
            combined_radius_sq = sqr(combined_radius)
            
            dist_sq = abs_sq(relative_position)

            # Compute ORCA line based on mean velocity
            # (Implementation of ORCA line computation goes here)
            # This is a placeholder for the actual ORCA line calculation.
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

