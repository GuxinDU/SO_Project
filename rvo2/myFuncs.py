import numpy as np

def count_overlapping_balls(centers, radius):
    """
    Checks how many of the n balls have overlapped with some other balls.
    
    Args:
        centers (dict): Dictionary with index as key and np.array center (x, y) as value.
        radius (float): Radius of the balls.
        
    Returns:
        int: Number of balls that overlap with at least one other ball.
    """
    if not centers:
        return 0
        
    # Extract coordinates from the dictionary values
    # We convert to a list first, then to a numpy array
    coords = np.array(list(centers.values()))
    
    n = len(coords)
    if n < 2:
        return 0
        
    # Calculate squared distance matrix using broadcasting
    # shape of coords: (n, 2)
    # delta shape: (n, n, 2)
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(delta**2, axis=-1)
    
    # Two balls overlap if the distance between their centers is less than 2 * radius
    # We compare squared distances to avoid square roots
    threshold_sq = (2 * radius) ** 2
    
    # Check for overlaps
    # dist_sq < threshold_sq
    overlaps = dist_sq < threshold_sq
    
    # A ball always "overlaps" with itself (distance 0), so we must ignore the diagonal
    np.fill_diagonal(overlaps, False)
    
    # Check which balls have at least one overlap (True in their row)
    balls_with_overlap = np.any(overlaps, axis=1)
    
    return np.sum(balls_with_overlap)
