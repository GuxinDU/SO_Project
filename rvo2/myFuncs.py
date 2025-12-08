import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

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

def plot_trajectories(position_list, output_file=None, title="Agent Trajectories"):
    """
    Plots the trajectories of agents from the position list.

    Args:
        position_list (list): List of dictionaries where each dictionary contains agent positions at a time step.
                              Format: [{agent_id: position, ...}, ...]
        output_file (str, optional): If provided, saves the plot to this file.
        title (str, optional): Title of the plot.
    """
    if not position_list:
        print("No positions to plot.")
        return

    # Reorganize data into trajectories per agent
    # Assume all agents present in the first step are relevant
    if not position_list[0]:
        return

    agent_ids = sorted(position_list[0].keys())
    trajectories = {agent_id: [] for agent_id in agent_ids}

    for pos_dict in position_list:
        for agent_id in agent_ids:
            if agent_id in pos_dict:
                pos = pos_dict[agent_id]
                # Handle different position formats (Vector2, tuple, list, np.array)
                if hasattr(pos, 'x') and hasattr(pos, 'y'):
                    x, y = pos.x, pos.y
                elif hasattr(pos, '__getitem__'):
                    x, y = pos[0], pos[1]
                else:
                    continue # Skip invalid
                trajectories[agent_id].append((x, y))

    plt.figure(figsize=(10, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(agent_ids)))

    for agent_id, color in zip(agent_ids, colors):
        traj = trajectories[agent_id]
        if not traj:
            continue
        
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        
        plt.plot(xs, ys, color=color, label=f'Agent {agent_id}')
        
        # Mark start and end
        plt.plot(xs[0], ys[0], 'o', color=color, markersize=8) # Start
        plt.plot(xs[-1], ys[-1], '*', color=color, markersize=12) # End

    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def create_animation(position_list, velocity_list, output_file, time_step=0.5, radius=0.5, title="Agent Animation"):
    """
    Creates a GIF animation of the agents' movement.

    Args:
        position_list (list): List of dictionaries with agent positions.
        velocity_list (list): List of dictionaries with agent velocities.
        output_file (str): Path to save the GIF.
        time_step (float): Time step between frames (seconds).
        radius (float): Radius of the agents for visualization.
        title (str): Title of the animation.
    """
    if not position_list:
        print("No positions to animate.")
        return

    agent_ids = sorted(position_list[0].keys())
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True)
    
    # Determine bounds
    all_x = []
    all_y = []
    for pos_dict in position_list:
        for pos in pos_dict.values():
            if hasattr(pos, 'x'):
                all_x.append(pos.x)
                all_y.append(pos.y)
            elif hasattr(pos, '__getitem__'):
                all_x.append(pos[0])
                all_y.append(pos[1])
    
    if all_x:
        margin = 1.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    else:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

    colors = cm.rainbow(np.linspace(0, 1, len(agent_ids)))
    circles = {}
    quivers = {}
    
    for agent_id, color in zip(agent_ids, colors):
        # Initialize circle
        circle = Circle((0, 0), radius, color=color, fill=False, linewidth=2)
        ax.add_patch(circle)
        circles[agent_id] = circle
        
        # Initialize velocity vector (quiver)
        # We use a single vector quiver for each agent to easily update it
        quiver = ax.quiver([0], [0], [0], [0], color=color, scale=0.5, scale_units='xy', angles='xy')
        quivers[agent_id] = quiver

    def update(frame):
        pos_dict = position_list[frame]
        vel_dict = velocity_list[frame] if frame < len(velocity_list) else {}
        
        artists = []
        for agent_id in agent_ids:
            if agent_id in pos_dict:
                pos = pos_dict[agent_id]
                if hasattr(pos, 'x'):
                    x, y = pos.x, pos.y
                elif hasattr(pos, '__getitem__'):
                    x, y = pos[0], pos[1]
                else:
                    continue
                
                circles[agent_id].center = (x, y)
                artists.append(circles[agent_id])
                
                if agent_id in vel_dict:
                    vel = vel_dict[agent_id]
                    if hasattr(vel, 'x'):
                        vx, vy = vel.x, vel.y
                    elif hasattr(vel, '__getitem__'):
                        vx, vy = vel[0], vel[1]
                    else:
                        vx, vy = 0, 0
                    
                    # Update quiver
                    quivers[agent_id].set_offsets([[x, y]])
                    quivers[agent_id].set_UVC([vx], [vy])
                    artists.append(quivers[agent_id])
            
        return artists

    anim = FuncAnimation(fig, update, frames=len(position_list), blit=True, interval=time_step*1000)
    
    print(f"Saving animation to {output_file}...")
    try:
        anim.save(output_file, writer=PillowWriter(fps=int(1/time_step)))
        print("Done.")
    except Exception as e:
        print(f"Error saving animation: {e}")
    finally:
        plt.close(fig)


