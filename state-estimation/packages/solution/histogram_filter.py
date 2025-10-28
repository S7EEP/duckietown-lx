# start by importing some things we will need
import numpy as np
from math import floor, sqrt, sin
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import multivariate_normal

# Now let's define the prior function. In this case, we choose
# to initialize the histogram based on a Gaussian distribution around [0,0]
def histogram_prior(belief, grid_spec, mean_0, cov_0):
    pos = np.empty(belief.shape + (2,))
    pos[:, :, 0] = grid_spec["d"]
    pos[:, :, 1] = grid_spec["phi"]
    RV = multivariate_normal(mean_0, cov_0)
    belief = RV.pdf(pos)
    return belief

# Now let's define the predict function
def histogram_predict(belief, left_encoder_ticks, right_encoder_ticks, grid_spec, robot_spec, cov_mask):
    belief_in = belief

    # Calculate v and w from ticks using kinematics
    wheel_radius = robot_spec['wheel_radius']
    baseline = robot_spec['wheel_baseline']
    delta_l = left_encoder_ticks * (2 * np.pi * wheel_radius / robot_spec['encoder_resolution'])
    delta_r = right_encoder_ticks * (2 * np.pi * wheel_radius / robot_spec['encoder_resolution'])

    # Linear and angular velocity
    v = (delta_r + delta_l) / 2
    w = (delta_r - delta_l) / baseline

    # You may find the following code useful to find the current best heading estimate:
    maxids = np.unravel_index(belief_in.argmax(), belief_in.shape)
    phi_max = grid_spec['phi_min'] + (maxids[1] + 0.5) * grid_spec['delta_phi']

    # Propagate each centroid
    d_t = grid_spec["d"] + v * grid_spec['delta_d'] * sin(phi_max)
    phi_t = grid_spec["phi"] + w * grid_spec['delta_phi']

    p_belief = np.zeros(belief.shape)

    # Accumulate the mass for each cell as a result of the propagation step
    for i in range(belief.shape[0]):
        for j in range(belief.shape[1]):
            if belief[i, j] > 0:
                if (
                    d_t[i, j] > grid_spec["d_max"]
                    or d_t[i, j] < grid_spec["d_min"]
                    or phi_t[i, j] < grid_spec["phi_min"]
                    or phi_t[i, j] > grid_spec["phi_max"]
                ):
                    continue

                i_new = int(floor((d_t[i, j] - grid_spec["d_min"]) / grid_spec["delta_d"]))
                j_new = int(floor((phi_t[i, j] - grid_spec["phi_min"]) / grid_spec["delta_phi"]))

                p_belief[i_new, j_new] += belief[i, j]

    s_belief = np.zeros(belief.shape)
    gaussian_filter(p_belief, cov_mask, output=s_belief, mode="constant")

    if np.sum(s_belief) == 0:
        return belief_in
    belief = s_belief / np.sum(s_belief)
    return belief

# Processing on the segments to remove anything that is behind the robot or not yellow/white
def prepare_segments(segments, grid_spec):
    filtered_segments = []
    for segment in segments:
        if segment.color != segment.WHITE and segment.color != segment.YELLOW:
            continue
        if segment.points[0].x < 0 or segment.points[1].x < 0:
            continue

        point_range = getSegmentDistance(segment)
        if grid_spec["range_est"] > point_range > 0:
            filtered_segments.append(segment)
    return filtered_segments

def generate_vote(segment, road_spec):
    p1 = np.array([segment.points[0].x, segment.points[0].y])
    p2 = np.array([segment.points[1].x, segment.points[1].y])
    t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)
    n_hat = np.array([-t_hat[1], t_hat[0]])

    d1 = np.inner(n_hat, p1)
    d2 = np.inner(n_hat, p2)
    l1 = np.inner(t_hat, p1)
    l2 = np.inner(t_hat, p2)
    l_i = (l1 + l2) / 2
    d_i = (d1 + d2) / 2
    phi_i = np.arcsin(t_hat[1])

    if segment.color == segment.WHITE:
        if p1[0] > p2[0]:
            d_i -= road_spec["linewidth_white"]
        else:
            d_i -= road_spec["linewidth_white"]
            d_i = road_spec["lanewidth"] * 2 + road_spec["linewidth_yellow"] - d_i
            phi_i = -phi_i
        d_i -= road_spec["lanewidth"] / 2

    elif segment.color == segment.YELLOW:
        if p2[0] > p1[0]:
            d_i -= road_spec["linewidth_yellow"]
            d_i = road_spec["lanewidth"] / 2 - d_i
            phi_i = -phi_i
        else:
            d_i += road_spec["linewidth_yellow"]
            d_i -= road_spec["lanewidth"] / 2

    return d_i, phi_i

def generate_measurement_likelihood(segments, road_spec, grid_spec):
    measurement_likelihood = np.zeros(grid_spec["d"].shape)

    for segment in segments:
        d_i, phi_i = generate_vote(segment, road_spec)

        if (
            d_i > grid_spec["d_max"]
            or d_i < grid_spec["d_min"]
            or phi_i < grid_spec["phi_min"]
            or phi_i > grid_spec["phi_max"]
        ):
            continue

        i = int(floor((d_i - grid_spec["d_min"]) / grid_spec["delta_d"]))
        j = int(floor((phi_i - grid_spec["phi_min"]) / grid_spec["delta_phi"]))

        measurement_likelihood[i, j] += 1

    if np.linalg.norm(measurement_likelihood) == 0:
        return None
    measurement_likelihood /= np.sum(measurement_likelihood)
    return measurement_likelihood

def histogram_update(belief, segments, road_spec, grid_spec):
    segmentsArray = prepare_segments(segments, grid_spec)
    measurement_likelihood = generate_measurement_likelihood(segmentsArray, road_spec, grid_spec)

    if measurement_likelihood is not None:
        belief = np.multiply(belief,measurement_likelihood)
        if np.sum(belief) == 0:
                belief = measurement_likelihood
        else:
            belief /= np.sum(belief)
    return measurement_likelihood, belief

def getSegmentDistance(segment):
    x_c = (segment.points[0].x + segment.points[1].x) / 2
    y_c = (segment.points[0].y + segment.points[1].y) / 2
    return sqrt(x_c**2 + y_c**2)