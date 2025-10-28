import itertools
import math
from typing import List, Tuple

from aido_schemas import Context, FriendlyPose
from dt_protocols import (
    Circle,
    CollisionCheckQuery,
    CollisionCheckResult,
    MapDefinition,
    PlacedPrimitive,
    Rectangle,
)

class CollisionChecker:
    params: MapDefinition

    def init(self, context: Context):
        context.info("init()")

    def on_received_set_params(self, context: Context, data: MapDefinition):
        context.info("initialized")
        self.params = data

    def on_received_query(self, context: Context, data: CollisionCheckQuery):
        collided = check_collision(
            environment=self.params.environment, robot_body=self.params.body, robot_pose=data.pose
        )
        result = CollisionCheckResult(collided)
        context.write("response", result)

def check_collision(
    environment: List[PlacedPrimitive], robot_body: List[PlacedPrimitive], robot_pose: FriendlyPose
) -> bool:
    # Rototranslate the robot body
    rototranslated_robot: List[PlacedPrimitive] = [
        rototranslate(primitive, robot_pose) for primitive in robot_body
    ]
    
    # Check for collisions with the environment
    return check_collision_list(rototranslated_robot, environment)

def rotate_point(x: float, y: float, theta: float) -> Tuple[float, float]:
    """Rotate a point around origin by theta radians"""
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return (x * cos_theta - y * sin_theta, x * sin_theta + y * cos_theta)

def get_rectangle_corners(rect: PlacedPrimitive) -> List[Tuple[float, float]]:
    """Get the four corners of a rectangle considering its position and rotation"""
    theta = math.radians(rect.pose.theta_deg)
    corners = [
        (rect.primitive.xmin, rect.primitive.ymin),
        (rect.primitive.xmax, rect.primitive.ymin),
        (rect.primitive.xmax, rect.primitive.ymax),
        (rect.primitive.xmin, rect.primitive.ymax),
    ]
    # Rotate corners
    rotated_corners = [rotate_point(x, y, theta) for x, y in corners]
    # Translate corners
    return [(x + rect.pose.x, y + rect.pose.y) for x, y in rotated_corners]

def rototranslate(placed_primitive: PlacedPrimitive, pose: FriendlyPose) -> PlacedPrimitive:
    # Calculate the new position and orientation
    x, y, base_theta = pose.x, pose.y, math.radians(pose.theta_deg)
    primitive_theta = math.radians(placed_primitive.pose.theta_deg)
    
    # Rotate the primitive position around origin
    rotated_x, rotated_y = rotate_point(placed_primitive.pose.x, placed_primitive.pose.y, base_theta)
    
    # Translate to final position
    new_x = rotated_x + x
    new_y = rotated_y + y
    
    # Add the rotations
    new_theta_deg = (math.degrees(base_theta) + placed_primitive.pose.theta_deg) % 360
    
    new_pose = FriendlyPose(new_x, new_y, new_theta_deg)
    return PlacedPrimitive(new_pose, placed_primitive.primitive)

def check_collision_list(
    rototranslated_robot: List[PlacedPrimitive], environment: List[PlacedPrimitive]
) -> bool:
    return any(
        check_collision_shape(robot, env_obj)
        for robot, env_obj in itertools.product(rototranslated_robot, environment)
    )

def check_collision_shape(a: PlacedPrimitive, b: PlacedPrimitive) -> bool:
    if isinstance(a.primitive, Circle) and isinstance(b.primitive, Circle):
        return check_circle_circle(a, b)
    elif isinstance(a.primitive, Rectangle) and isinstance(b.primitive, Circle):
        return check_rectangle_circle(a, b)
    elif isinstance(a.primitive, Circle) and isinstance(b.primitive, Rectangle):
        return check_rectangle_circle(b, a)
    elif isinstance(a.primitive, Rectangle) and isinstance(b.primitive, Rectangle):
        return check_rectangle_rectangle(a, b)
    return False

def check_circle_circle(a: PlacedPrimitive, b: PlacedPrimitive) -> bool:
    distance = math.hypot(a.pose.x - b.pose.x, a.pose.y - b.pose.y)
    return distance <= (a.primitive.radius + b.primitive.radius)

def point_to_line_distance(point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
    """Calculate the shortest distance from a point to a line segment"""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line start to end
    line_vec = (x2 - x1, y2 - y1)
    # Vector from line start to point
    point_vec = (px - x1, py - y1)
    
    # Length of line segment squared
    line_len_sq = line_vec[0]**2 + line_vec[1]**2
    
    if line_len_sq == 0:
        return math.hypot(px - x1, py - y1)
    
    # Project point vector onto line vector to get relative position along line
    t = max(0, min(1, (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_len_sq))
    
    # Get the closest point on the line segment
    closest_x = x1 + t * line_vec[0]
    closest_y = y1 + t * line_vec[1]
    
    return math.hypot(px - closest_x, py - closest_y)

def check_rectangle_circle(rect: PlacedPrimitive, circle: PlacedPrimitive) -> bool:
    # Get the corners of the rotated rectangle
    corners = get_rectangle_corners(rect)
    
    # Check if circle center is inside the rectangle
    # Implementation of point-in-polygon algorithm
    inside = False
    n = len(corners)
    j = n - 1
    for i in range(n):
        if (((corners[i][1] > circle.pose.y) != (corners[j][1] > circle.pose.y)) and
            (circle.pose.x < (corners[j][0] - corners[i][0]) * (circle.pose.y - corners[i][1]) /
             (corners[j][1] - corners[i][1]) + corners[i][0])):
            inside = not inside
        j = i
    
    if inside:
        return True
    
    # Check distance to each edge
    for i in range(len(corners)):
        j = (i + 1) % len(corners)
        distance = point_to_line_distance(
            (circle.pose.x, circle.pose.y),
            corners[i],
            corners[j]
        )
        if distance <= circle.primitive.radius:
            return True
    
    return False

def check_rectangle_rectangle(a: PlacedPrimitive, b: PlacedPrimitive) -> bool:
    # Get corners for both rectangles
    corners_a = get_rectangle_corners(a)
    corners_b = get_rectangle_corners(b)
    
    # Separating Axis Theorem (SAT)
    def get_axes(corners):
        axes = []
        for i in range(len(corners)):
            j = (i + 1) % len(corners)
            edge = (corners[j][0] - corners[i][0], corners[j][1] - corners[i][1])
            normal = (-edge[1], edge[0])  # Perpendicular vector
            length = math.hypot(normal[0], normal[1])
            if length > 0:
                axes.append((normal[0]/length, normal[1]/length))
        return axes
    
    def project(corners, axis):
        dots = [axis[0]*x + axis[1]*y for x, y in corners]
        return min(dots), max(dots)
    
    # Get all axes to test
    axes = get_axes(corners_a) + get_axes(corners_b)
    
    # Test projection onto each axis
    for axis in axes:
        proj_a = project(corners_a, axis)
        proj_b = project(corners_b, axis)
        
        # Check for separation
        if proj_a[1] < proj_b[0] or proj_b[1] < proj_a[0]:
            return False
            
    return True