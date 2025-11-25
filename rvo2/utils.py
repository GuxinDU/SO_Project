from .vector import Vector2, abs_sq, det

RVO_EPSILON = 0.00001

def distSqPointLineSegment(a, b, c):
    r = ((c - a) * (b - a)) / abs_sq(b - a)

    if r < 0.0:
        return abs_sq(c - a)
    elif r > 1.0:
        return abs_sq(c - b)
    else:
        return abs_sq(c - (a + r * (b - a)))

def leftOf(a, b, c):
    return det(a - c, b - a)

def sqr(a):
    return a * a
