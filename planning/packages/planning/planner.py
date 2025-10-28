from typing import List, Optional, Tuple
import math
import networkx as nx
from aido_schemas import Context, FriendlyPose
from dt_protocols import (
    PlacedPrimitive,
    PlanningQuery,
    PlanningResult,
    PlanningSetup,
    PlanStep,
    Circle,
    Rectangle,
)

def normalize_angle_deg(angle_deg: float) -> float:
    """Normalize angle to [0, 360) range."""
    return angle_deg % 360

def angle_diff_deg(a: float, b: float) -> float:
    """Compute the smallest angle difference in degrees."""
    diff = normalize_angle_deg(b - a)
    if diff > 180:
        diff -= 360
    return diff

def compute_heading_deg(dx: float, dy: float) -> float:
    """Compute heading angle in degrees from delta x and y."""
    return normalize_angle_deg(math.degrees(math.atan2(dy, dx)))

class Planner:
    def init(self, context: Context):
        context.info("init()")

    def on_received_set_params(self, context: Context, data: PlanningSetup):
        self.params = data
        self.environment = self.params.environment
        self.graph = nx.Graph()
        self.build_graph()
        context.info("Planner initialized with Dijkstra")

    def check_collision(self, pose: FriendlyPose) -> bool:
        """Check if a given pose collides with any obstacles."""
        for obstacle in self.environment:
            if isinstance(obstacle.primitive, Circle):
                dist = math.hypot(obstacle.pose.x - pose.x, obstacle.pose.y - pose.y)
                if dist < obstacle.primitive.radius:
                    return True
            elif isinstance(obstacle.primitive, Rectangle):
                if self.is_pose_in_rectangle(pose, obstacle):
                    return True
        return False

    def is_pose_in_rectangle(self, pose: FriendlyPose, rect: PlacedPrimitive) -> bool:
        """Check if a pose is inside a rectangle."""
        theta = math.radians(rect.pose.theta_deg)
        x_rel = pose.x - rect.pose.x
        y_rel = pose.y - rect.pose.y
        x_rot = x_rel * math.cos(-theta) - y_rel * math.sin(-theta)
        y_rot = x_rel * math.sin(-theta) + y_rel * math.cos(-theta)
        return (
            rect.primitive.xmin <= x_rot <= rect.primitive.xmax and
            rect.primitive.ymin <= y_rot <= rect.primitive.ymax
        )

    def build_graph(self, resolution: float = 1.0):
        """Build a roadmap graph with collision checking."""
        x_min, x_max = self.params.bounds.xmin, self.params.bounds.xmax
        y_min, y_max = self.params.bounds.ymin, self.params.bounds.ymax
        x_range = range(int(x_min), int(x_max) + 1, int(resolution))
        y_range = range(int(y_min), int(y_max) + 1, int(resolution))

        # Add nodes
        for x in x_range:
            for y in y_range:
                pose = FriendlyPose(x, y, 0)
                if not self.check_collision(pose):
                    self.graph.add_node((x, y), pose=pose)

        # Connect neighboring nodes with edges
        for (x1, y1) in self.graph.nodes():
            for (x2, y2) in self.graph.nodes():
                if (x1, y1) != (x2, y2):
                    distance = math.hypot(x2 - x1, y2 - y1)
                    if distance <= resolution * 1.5:
                        self.graph.add_edge((x1, y1), (x2, y2), weight=distance)

    def connect_poses(self, start: FriendlyPose, goal: FriendlyPose) -> List[PlanStep]:
        """Generate a plan connecting two poses."""
        dx = goal.x - start.x
        dy = goal.y - start.y
        distance = math.hypot(dx, dy)
        heading_deg = compute_heading_deg(dx, dy)

        initial_turn_deg = angle_diff_deg(start.theta_deg, heading_deg)
        final_turn_deg = angle_diff_deg(heading_deg, goal.theta_deg)

        plan = []

        # Initial turn
        if abs(initial_turn_deg) > self.params.tolerance_theta_deg:
            turn_duration = abs(initial_turn_deg) / self.params.max_angular_velocity_deg_s
            turn_velocity = math.copysign(self.params.max_angular_velocity_deg_s, initial_turn_deg)
            plan.append(PlanStep(
                duration=turn_duration,
                velocity_x_m_s=0.0,
                angular_velocity_deg_s=turn_velocity
            ))

        # Straight drive
        if distance > self.params.tolerance_xy_m:
            forward_duration = distance / self.params.max_linear_velocity_m_s
            plan.append(PlanStep(
                duration=forward_duration,
                velocity_x_m_s=self.params.max_linear_velocity_m_s,
                angular_velocity_deg_s=0.0
            ))

        # Final turn
        if abs(final_turn_deg) > self.params.tolerance_theta_deg:
            turn_duration = abs(final_turn_deg) / self.params.max_angular_velocity_deg_s
            turn_velocity = math.copysign(self.params.max_angular_velocity_deg_s, final_turn_deg)
            plan.append(PlanStep(
                duration=turn_duration,
                velocity_x_m_s=0.0,
                angular_velocity_deg_s=turn_velocity
            ))

        return plan

    def on_received_query(self, context: Context, data: PlanningQuery):
        """Handle planning query and return a collision-free plan."""
        start = data.start
        goal = data.target

        # Find nearest graph nodes to start and goal
        def closest_node(pose: FriendlyPose):
            return min(self.graph.nodes(), key=lambda n: math.hypot(n[0] - pose.x, n[1] - pose.y))

        start_node = closest_node(start)
        goal_node = closest_node(goal)

        try:
            # Find shortest path using Dijkstra's algorithm
            path_nodes = nx.shortest_path(self.graph, start_node, goal_node, weight="weight")
            path_poses = [self.graph.nodes[node]["pose"] for node in path_nodes]

            # Generate a plan by connecting consecutive poses
            plan = []
            for i in range(1, len(path_poses)):
                plan.extend(self.connect_poses(path_poses[i - 1], path_poses[i]))

            # Connect final pose to goal
            plan.extend(self.connect_poses(path_poses[-1], goal))

            context.write("response", PlanningResult(feasible=True, plan=plan))
        except nx.NetworkXNoPath:
            context.write("response", PlanningResult(feasible=False, plan=None))
