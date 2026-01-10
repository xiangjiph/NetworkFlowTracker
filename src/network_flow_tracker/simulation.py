from collections import defaultdict
import numpy as np 
from numba import njit, prange
from scipy.spatial import cKDTree

from .utils import util
#region Cylinder particle packing simulation 
def sample_free_particles_in_cylinder(N, R, L):
    r = R * np.sqrt(np.random.rand(N))
    theta = 2 * 3.1415926 * np.random.rand(N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = L * np.random.rand(N)
    return np.column_stack((x, y, z))

def sample_hard_spheres_in_cylinder_v1(N, R, L, r0, max_attempts=100000):
    """Generate N non-overlapping hard-sphere particles inside a cylinder.
        This is too slow for N > 10^4. The main cost probably comes from 
        checking constructing the kdtree for every particle inserted. 
    """
    positions = []
    attempts = 0
    while len(positions) < N and attempts < max_attempts:
        attempts += 1

        # Propose a random point
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, (R - r0)**2))  # ensure full sphere fits inside
        z = np.random.uniform(r0, L - r0)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        new_pos = np.array([x, y, z])

        # Check for overlaps
        if positions:
            tree = cKDTree(positions)
            neighbors = tree.query_ball_point(new_pos, 2 * r0)
            if neighbors:
                continue  # Overlap found, reject

        positions.append(new_pos)

        if attempts % 100 == 1: 
            print(f"\rFinish placing {len(positions)} particles after {attempts} attemps.", end='', flush=True)
            

    if len(positions) < N:
        print(f"Warning: Only placed {len(positions)} particles out of {N} after {attempts} attempts.")
    else: 
        print(f"Successfully placed {N} particles after {attempts} iterations.")

    return np.array(positions)

def sample_hard_spheres_cylinder_v2(N, r0, R, L, cell_size=None, max_attempts=None):
    """
    Efficient sampling of non-overlapping spheres in a cylinder using cell lists.

    Parameters:
    - N: Number of spheres
    - r0: Sphere radius
    - R: Cylinder radius
    - L: Cylinder length
    - cell_size: Size of spatial grid cell (default = 2*r0)
    - max_attempts: Max sampling attempts

    Returns:
    - positions: (N, 3) NumPy array of accepted sphere positions
    """
    if cell_size is None:
        cell_size = 2 * r0
    if max_attempts is None: 
        max_attempts = 10 * N

    # Define valid bounds (accounting for r0 margin)
    r_effective = R - r0
    # z_min, z_max = r0, L - r0
    z_min, z_max = 0, L
    r0_squared = (2 * r0) ** 2

    # Cell indexing function
    def cell_index(pos):
        return tuple((pos // cell_size).astype(int))

    # Neighboring cell offsets (27 neighbors in 3D)
    neighbor_offsets = np.array([
        [dx, dy, dz] for dx in (-1, 0, 1)
                     for dy in (-1, 0, 1)
                     for dz in (-1, 0, 1)
    ])

    # Store sphere centers and spatial hash
    positions = []
    grid = defaultdict(list)

    attempts = 0
    while len(positions) < N and attempts < max_attempts:
        # Batch sampling
        batch_size = min(1000, N - len(positions))

        rho = r_effective * np.sqrt(np.random.rand(batch_size))
        phi = 2 * np.pi * np.random.rand(batch_size)
        z = np.random.uniform(z_min, z_max, batch_size)

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        new_points = np.stack((x, y, z), axis=1)

        for point in new_points:
            attempts += 1
            c_idx = cell_index(point)
            neighbors = []

            # Check 3x3x3 neighboring cells
            for offset in neighbor_offsets:
                neighbor_idx = tuple(np.array(c_idx) + offset)
                neighbors.extend(grid[neighbor_idx])

            # Only check against nearby spheres
            if len(neighbors) == 0 or np.all(np.sum((point - np.vstack([positions[i] for i in neighbors])) ** 2, axis=1) >= r0_squared):
                idx = len(positions)
                positions.append(point)
                grid[c_idx].append(idx)

                if len(positions) >= N:
                    break

            # if attempts % 100 == 1: 
            #     print(f"\rFinish placing {len(positions)} particles after {attempts} attemps.", end='', flush=True)

    if len(positions) < N:
        raise RuntimeError(f"Only placed {len(positions)} out of {N} spheres after {max_attempts} attempts.")
    else: 
        print(f"\rSuccessfully placed {N} particles after {attempts} iterations.")

    return np.array(positions)

@njit(parallel=True)
def compute_directional_distances(points, r, num_bins:int):
    """Average distnace between particles - not very useful for the 
    axial direction as it scale with the cylinder length 
    
    
    """ 
    N = points.shape[0]
    r_bin_width = r / num_bins
    radial_sums = np.zeros(num_bins)
    axial_sums = np.zeros(num_bins)
    d_sums = np.zeros(num_bins)

    radial_sums2 = np.zeros(num_bins)
    axial_sums2 = np.zeros(num_bins)
    d_sums2 = np.zeros(num_bins)

    counts = np.zeros(num_bins)

    for i in prange(N):
        xi, yi, zi = points[i]
        ri = np.sqrt(xi**2 + yi**2)
        bin_index = int(ri // r_bin_width)

        sum_dr = 0.0
        sum_dz = 0.0
        sum_dd = 0.0

        sum_dr2 = 0.0
        sum_dz2 = 0.0
        sum_dd2 = 0.0
        for j in range(N):
            if i == j:
                continue
            xj, yj, zj = points[j]
            dx2 = (xi - xj) ** 2
            dy2 = (yi - yj) ** 2
            dz = abs(zi - zj)
            dr2= dx2 + dy2
            dd2 = dr2 + dz ** 2

            sum_dr += np.sqrt(dr2)
            sum_dz += dz
            sum_dd += np.sqrt(dd2)

            sum_dr2 += dr2
            sum_dz2 += dz ** 2
            sum_dd2 += dd2

        d_sums[bin_index] += sum_dd
        radial_sums[bin_index] += sum_dr 
        axial_sums[bin_index] += sum_dz
        
        d_sums2[bin_index] += sum_dd2
        radial_sums2[bin_index] += sum_dr2
        axial_sums2[bin_index] += sum_dz2

        counts[bin_index] += 1

    counts *= N

    d_sums /= counts
    radial_sums /= counts
    axial_sums /= counts

    d_sums2 /= counts
    radial_sums2 /= counts
    axial_sums2 /= counts

    dd_std = np.sqrt(d_sums2 - d_sums ** 2)
    r_std = np.sqrt(radial_sums2 - radial_sums ** 2)
    z_std = np.sqrt(axial_sums2 - axial_sums ** 2)

    bin_centers = (np.arange(num_bins) + 0.5) * r_bin_width
    result = {'bin_val': bin_centers, 
              'dd': d_sums, 'dd_std': dd_std, 
              'dr': radial_sums, 'dr_std': r_std, 
              'dz': axial_sums, 'dz_std': z_std
              }
    return result

def compute_avg_min_distances(positions, r, num_bins):
    """Compute average minimum radial, axial, and 3D distances between particles."""
    r_bin_width = r / num_bins

    tree = cKDTree(positions)
    # Query the nearest neighbor (k=2 because first is itself)
    dist_3d, indices = tree.query(positions, k=2)
    nearest = indices[:, 1]
    deltas = positions - positions[nearest]
    dist_val = {}
    dist_val['r'] = np.linalg.norm(deltas[:, :2], axis=1)
    dist_val['z'] = np.abs(deltas[:, 2])
    dist_val['d'] = dist_3d[:, 1]

    p_r_idx = np.round(np.sqrt(np.sum(positions[:, :2] ** 2, axis=1)) // r_bin_width).astype(np.uint8)
    idx, r_idx = util.bin_data_to_idx_list(p_r_idx)
    r_dist_mean = {}
    r_dist_std = {}
    r_dist_se = {}
    for k, v in dist_val.items():
        tmp_mean, tmp_std, tmp_se = [np.full(num_bins, np.nan, dtype=np.float64) for _ in range(3)]
        for i, i_r in zip(idx, r_idx):
            tmp_v = v[i]
            tmp_m = np.mean(tmp_v)
            tmp_m2 = np.mean(tmp_v ** 2)
            tmp_s = np.sqrt(tmp_m2 - tmp_m ** 2)
            
            tmp_mean[i_r] = tmp_m
            tmp_std[i_r] = tmp_s
            tmp_se[i_r] = tmp_s / np.sqrt(tmp_v.size)
        
        r_dist_mean[k] = tmp_mean
        r_dist_std[k] = tmp_std
        r_dist_se[k] = tmp_se

    bin_val = (np.arange(num_bins) + 0.5) * r_bin_width

    return bin_val, r_dist_mean, r_dist_std, r_dist_se

#endregion