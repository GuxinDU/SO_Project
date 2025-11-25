from .vector import Vector2

class Obstacle:
    def __init__(self):
        self.is_convex = False
        self.next_obstacle = None
        self.point = Vector2()
        self.prev_obstacle = None
        self.unit_dir = Vector2()
        self.id = 0
