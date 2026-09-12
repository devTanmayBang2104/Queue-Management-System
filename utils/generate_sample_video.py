"""
Sample Video Generator
Creates a synthetic test video with simulated people moving in queue patterns.
Uses OpenCV to draw colored rectangles representing people.
"""

import cv2
import numpy as np
import os
import math
import random

def generate_sample_video(output_path="data/sample_queue.mp4", duration=60, fps=25, width=1280, height=720):
    """
    Generate a sample video with simulated persons in a queue.
    
    People are drawn as colored rectangles that:
    - Enter from the right side
    - Queue up in a line
    - Slowly move forward
    - Exit from the left side
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    # Person representation
    class SimPerson:
        def __init__(self, pid):
            self.id = pid
            self.x = width + random.randint(0, 100)
            self.y = random.randint(250, 450)
            self.w = random.randint(40, 60)
            self.h = random.randint(100, 140)
            self.target_x = random.randint(100, width - 200)
            self.speed = random.uniform(1.0, 3.0)
            self.color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255),
            )
            self.in_queue = False
            self.sway = random.uniform(0, 2 * math.pi)
            self.served = False
            self.serve_timer = random.randint(fps * 5, fps * 20)
    
    persons = []
    next_id = 0
    spawn_timer = 0
    spawn_interval = fps * 2  # New person every 2 seconds
    
    # Queue positions (slots from left to right)
    queue_x_positions = [150 + i * 80 for i in range(12)]
    
    for frame_idx in range(total_frames):
        # Dark background with subtle gradient
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        bg[:] = (30, 25, 20)
        
        # Floor
        cv2.rectangle(bg, (0, 500), (width, height), (50, 45, 40), -1)
        cv2.line(bg, (0, 500), (width, 500), (80, 70, 60), 2)
        
        # Queue zone indicator
        cv2.rectangle(bg, (80, 200), (width - 100, 520), (40, 50, 40), 2)
        cv2.putText(bg, "QUEUE ZONE", (85, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 80, 40), 2)
        
        # Service counter
        cv2.rectangle(bg, (30, 300), (100, 480), (70, 70, 120), -1)
        cv2.putText(bg, "SERVICE", (25, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 200), 1)
        
        # Spawn new persons
        spawn_timer += 1
        if spawn_timer >= spawn_interval and len(persons) < 15:
            persons.append(SimPerson(next_id))
            next_id += 1
            spawn_timer = 0
            spawn_interval = random.randint(fps, fps * 4)
        
        # Assign queue positions
        active = [p for p in persons if not p.served]
        active.sort(key=lambda p: p.x)
        
        for i, p in enumerate(active):
            if i < len(queue_x_positions):
                p.target_x = queue_x_positions[i]
                p.in_queue = True
        
        # Update and draw persons
        to_remove = []
        for p in persons:
            # Move towards target
            if p.x > p.target_x + 2:
                p.x -= p.speed
            elif p.x < p.target_x - 2:
                p.x += p.speed * 0.5
            else:
                p.in_queue = True
                # First person in queue gets served
                if p.target_x == queue_x_positions[0]:
                    p.serve_timer -= 1
                    if p.serve_timer <= 0:
                        p.served = True
                        to_remove.append(p)
                        continue
            
            # Natural sway
            p.sway += 0.05
            sway_y = math.sin(p.sway) * 2
            
            # Draw person (stylized rectangle with head)
            px, py = int(p.x), int(p.y + sway_y)
            
            # Body
            cv2.rectangle(bg, (px, py), (px + p.w, py + p.h), p.color, -1)
            cv2.rectangle(bg, (px, py), (px + p.w, py + p.h), (200, 200, 200), 1)
            
            # Head (circle)
            head_cx = px + p.w // 2
            head_cy = py - 15
            cv2.circle(bg, (head_cx, head_cy), 12, p.color, -1)
            cv2.circle(bg, (head_cx, head_cy), 12, (200, 200, 200), 1)
            
            # ID label
            cv2.putText(bg, f"P{p.id}", (px, py - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        for p in to_remove:
            persons.remove(p)
        
        # Random abandonment (rare)
        if random.random() < 0.002 and len(persons) > 3:
            abandon_p = random.choice(persons[2:])
            persons.remove(abandon_p)
        
        # Frame counter
        cv2.putText(bg, f"Frame: {frame_idx}/{total_frames}", (width - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        writer.write(bg)
    
    writer.release()
    print(f"[SampleVideo] Generated: {output_path} ({duration}s @ {fps}fps)")
    return output_path


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_queue.mp4")
    generate_sample_video(output)
