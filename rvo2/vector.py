import math
import numpy as np

class Vector2:
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def __neg__(self):
        return Vector2(-self.x, -self.y)
    
    def to_array(self):
        return np.array([self.x, self.y])

    # Inner product
    def __mul__(self, other):
        if isinstance(other, Vector2):
            return self.x * other.x + self.y * other.y
        else:
            return Vector2(self.x * other, self.y * other)

    # Scalar multiplication
    def __rmul__(self, other):
        return Vector2(self.x * other, self.y * other)

    # Scalar division
    def __truediv__(self, other):
        return Vector2(self.x / other, self.y / other)

    # Vector addition
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    # Vector subtraction
    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y

    def __str__(self):
        return f"({self.x},{self.y})"
    
    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Vector2 index out of range")

def abs_vector(vector):
    return math.sqrt(vector * vector)

def abs_sq(vector):
    return vector * vector

def det(vector1, vector2):
    return vector1.x * vector2.y - vector1.y * vector2.x

def normalize(vector):
    return vector / abs_vector(vector)
