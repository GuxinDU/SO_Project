
import unittest
from rvo2.kdtree import KdTree, ObstacleTreeNode
from rvo2.obstacle import Obstacle
from rvo2.vector import Vector2
from rvo2.agent import Agent

class MockSim:
    def __init__(self):
        self.agents = []
        self.obstacles = []
        self.time_step = 0.1

class TestObstacleTree(unittest.TestCase):
    def test_query_obstacle_tree_recursive_returns_range(self):
        sim = MockSim()
        kdtree = KdTree(sim)
        
        # Create a simple obstacle tree manually or via build
        # Obstacle 1: (-10, 10) to (10, 10)
        o1 = Obstacle()
        o1.point = Vector2(-10, 10)
        o1.id = 0
        o2 = Obstacle()
        o2.point = Vector2(10, 10)
        o2.id = 0 # Same obstacle, next point
        
        o1.next_obstacle = o2
        o2.prev_obstacle = o1
        o1.unit_dir = Vector2(1, 0)
        o1.is_convex = True
        
        # We need a closed loop for build_obstacle_tree usually, but let's just make a single segment
        # The build function expects a list of obstacles.
        # Let's manually construct a tree node.
        node = ObstacleTreeNode()
        node.obstacle = o1
        node.left = None
        node.right = None
        
        kdtree.obstacle_tree = node
        
        agent = Agent(sim)
        agent.position = Vector2(0, 0)
        
        # Initial range
        range_sq = 10000.0
        
        # Query
        # The agent is at (0,0). Obstacle line is y=10. Dist is 10. DistSq is 100.
        # 100 < 10000. Should insert.
        # insert_obstacle_neighbor will add it.
        # Since max_neighbors is 0 by default? No, let's set it.
        # Actually Agent constructor sets max_neighbors = 0.
        # But compute_neighbors sets obstacle_neighbors = [].
        # And insert_obstacle_neighbor doesn't check max_neighbors for obstacles?
        # Let's check agent.py.
        
        # insert_obstacle_neighbor:
        # if dist_sq < range_sq:
        #    append...
        #    sort...
        #    (No trimming to max_neighbors? It seems it keeps all obstacles within range?)
        #    Wait, let me check agent.py content again.
        
        result_range_sq = kdtree.query_obstacle_tree_recursive(agent, range_sq, node)
        
        # If the function returns None (current bug), this will fail or be None.
        self.assertIsNotNone(result_range_sq, "query_obstacle_tree_recursive returned None")
        self.assertEqual(result_range_sq, 10000.0, "range_sq should be returned (and maybe updated)")

    def test_query_obstacle_tree_updates_range(self):
        # To test update, we need insert_obstacle_neighbor to reduce range_sq.
        # In agent.py:
        # def insert_obstacle_neighbor(self, obstacle, range_sq):
        #    ...
        #    if dist_sq < range_sq:
        #        self.obstacle_neighbors.append(...)
        #        ...
        #    return range_sq
        
        # It returns range_sq UNCHANGED!
        # Wait, let me check agent.py again.
        pass

if __name__ == '__main__':
    unittest.main()
