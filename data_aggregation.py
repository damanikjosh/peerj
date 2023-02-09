import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import networkx as nx
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm

# Generate argument parser
parser = ArgumentParser(description='Data aggregation', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--std', type=float, default=1.0, help='Standar deviation of generated data')
parser.add_argument('--n_samples', type=int, default=100, help='Number of samples')
parser.add_argument('--n_features', type=int, default=2, help='Number of features')
parser.add_argument('--n_clusters', type=int, default=4, help='Number of clusters')
parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations')
parser.add_argument('--n_neighbors', type=int, default=5, help='Number of neighbors')
parser.add_argument('--eps', type=float, default=1., help='Epsilon')
parser.add_argument('--seed', type=int, default=20, help='Random seed')
parser.add_argument('--fig_size_x', type=int, default=6, help='Figure size x')
parser.add_argument('--fig_size_y', type=int, default=4, help='Figure size y')
parser.add_argument('--save_path', type=str, default='result', help='Figure save path')
parser.add_argument('--save_name', type=str, default=None, help='Figure save name')
args = parser.parse_args()

# Get arguments
STD = args.std
N_SAMPLES = args.n_samples
N_FEATURES = args.n_features
N_CLUSTERS = args.n_clusters
MAX_ITER = args.max_iter
N_NEIGHBORS = args.n_neighbors
EPS = args.eps
SEED = args.seed
FIG_SIZE = (args.fig_size_x, args.fig_size_y)
SAVE_PATH = os.getcwd() if args.save_path is None else args.save_path
SAVE_NAME = args.save_name if args.save_name is not None else 'data_aggregation'
ANNEALING = False

# Define colors
colors = 'bgrcmykw'

# Define legend handles
handles = []
for i in range(4):
    handles.append(mpatches.Patch(color=colors[i], label='Cluster %d' % i))

# Create figure save path if it does not exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Set random seed
np.random.seed(SEED)

# Define gaussian function
def gaussian(x, mu, sigma):
    return np.exp(-np.linalg.norm(x - mu) / (2 * sigma ** 2)) * (np.linalg.norm(x - mu) < 2 * sigma)
    # return np.linalg.norm(x - mu) < 2 * sigma

# Generate sample data
X, y = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CLUSTERS, cluster_std=STD, random_state=SEED)

# Sort data based on cluster labels
y_sorted = np.argsort(y)
X_sorted = np.zeros(X.shape)
for i in range(X.shape[0]):
    X_sorted[i, :] = X[y_sorted[i], :]
X, y = X_sorted, y[y_sorted]


# Create graph
nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='ball_tree', radius=STD).fit(X)
distances, indices = nbrs.kneighbors(X)
G = nx.Graph()
for i in range(X.shape[0]):
    for j in range(1, indices.shape[1]):
        G.add_edge(i, indices[i, j], weight=np.exp(-distances[i, j] / (2 * STD ** 2)))

# Add random edges to graph to increase connectivity
for i in range(X.shape[0]):
    for j in range(X.shape[0]):
        if i != j and np.random.rand() < 0.001:
            G.add_edge(i, j, weight=np.exp(-np.linalg.norm(X[i, :] - X[j, :]) / (2 * STD ** 2)))

# Plot graph with nodes at sample data points
pos = {i: X[i, :] for i in range(X.shape[0])}
fig, ax = plt.subplots(figsize=FIG_SIZE)
nx.draw(G, pos, node_size=20, width=0.1, node_color=[colors[y[node]] for node in G.nodes], ax=ax)
plt.axis('on')
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.legend(handles=handles)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(os.path.join(SAVE_PATH, SAVE_NAME + '_graph.png'), bbox_inches='tight', dpi=300)

# Plot adjacency matrix
plt.figure(figsize=FIG_SIZE)
plt.imshow(nx.adjacency_matrix(G, nodelist=np.arange(N_SAMPLES)).toarray(), cmap='viridis')
plt.xlabel('Node')
plt.ylabel('Node')
plt.savefig(os.path.join(SAVE_PATH, SAVE_NAME + '_adjacency_matrix.png'), bbox_inches='tight', dpi=300)

error_history = []
omega_history = []
psi_history = []

omega = X.copy()
new_omega = X.copy()

# Initialize contribution matrix
phi = np.eye(X.shape[0])
new_phi = np.eye(X.shape[0])

# Initialize aggregate matrix
psi = np.zeros((X.shape[0],))
new_psi = np.zeros((X.shape[0],))

phi_converged = False
psi_converged = False

# Initialize data to be aggregated
s = [i for i in range(X.shape[0])]

# Calculate real aggregate
cluster_sum = [sum([s[i] for i in range(X.shape[0]) if y[i] == j]) for j in range(N_CLUSTERS)]
real_psi = np.array([cluster_sum[y[i]] for i in range(X.shape[0])])

# Perform data aggregation
eps = EPS
for iter in tqdm(range(MAX_ITER)):
    for i in range(X.shape[0]):
        if not phi_converged:
            sum_omega = omega[i, :].copy()
            sum_phi = phi[i, :].copy()
        sum_psi = psi[i] + eps * (s[i] - phi[i, i] * psi[i])
        sum_W = 1.
        for j in G.neighbors(i):
            W_ij = gaussian(omega[i, :], omega[j, :], STD)
            if not phi_converged:
                sum_phi += W_ij * phi[j, :]
            sum_omega += W_ij * omega[j, :]
            sum_psi += W_ij * psi[j]
            sum_W += W_ij
        if not phi_converged:
            new_phi[i, :] = sum_phi / sum_W
        new_omega[i, :] = sum_omega / sum_W
        new_psi[i] = sum_psi / sum_W

    # Check if stationary distribution has converged

    if not phi_converged:
        if np.linalg.norm(new_phi - phi) < 1e-6:
            print('Phi converged after {} iterations'.format(iter))
            phi_converged = True
        else:
            # print('%.6f' % np.linalg.norm(new_phi - phi))
            omega = new_omega.copy()
            phi = new_phi.copy()
    if ANNEALING and phi_converged and np.linalg.norm(new_psi - psi) < 0.1 * eps:
        eps = eps * 0.1
    if iter > 0 and np.linalg.norm(new_psi - psi) < 1e-6:
        print('Psi converged after {} iterations'.format(iter))
        MAX_ITER = iter
        break
    else:
        psi = new_psi.copy()

    error_history.append(np.linalg.norm(psi - real_psi))
    omega_history.append(omega.copy())
    psi_history.append(psi.copy())


# Extend history variables to full length
# error_history = error_history + [error_history[-1]] * (MAX_ITER - len(error_history))
# omega_history = omega_history + [omega_history[-1]] * (MAX_ITER - len(omega_history))
# psi_history = psi_history + [psi_history[-1]] * (MAX_ITER - len(psi_history))


# Save history variables to file
np.save(os.path.join(SAVE_PATH, SAVE_NAME + '_error_history.npy'), np.array(error_history))
np.save(os.path.join(SAVE_PATH, SAVE_NAME + '_omega_history.npy'), np.array(omega_history))
np.save(os.path.join(SAVE_PATH, SAVE_NAME + '_psi_history.npy'), np.array(psi_history))

# Plot error history
plt.figure(figsize=FIG_SIZE)
plt.plot(error_history)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.savefig(os.path.join(SAVE_PATH, SAVE_NAME + '_error_history.png'), bbox_inches='tight', dpi=300)

# Calculate trust matrix
V = np.zeros((X.shape[0], X.shape[0]))
for i in range(X.shape[0]):
    for j in range(X.shape[0]):
        V[i, j] = gaussian(omega[i, :], omega[j, :], STD)

# Plot trust matrix
plt.figure(figsize=FIG_SIZE)
plt.imshow(V, cmap='viridis')
plt.xlabel('Node')
plt.ylabel('Node')
plt.savefig(os.path.join(SAVE_PATH, SAVE_NAME + '_trust_matrix.png'), bbox_inches='tight', dpi=300)

# Plot omega history from samples
plt.figure(figsize=FIG_SIZE)
for i in range(omega_history[0].shape[0]):
    plt.plot([omega_history[iter][i, 0] for iter in range(MAX_ITER)], c=colors[y[i]], alpha=0.5)
plt.legend(handles=handles)
plt.xlabel('Iteration')
plt.ylabel('x')
plt.savefig(os.path.join(SAVE_PATH, SAVE_NAME + '_omega_history_x.png'), bbox_inches='tight', dpi=300)

plt.figure(figsize=FIG_SIZE)
for i in range(omega_history[0].shape[0]):
    plt.plot([omega_history[iter][i, 1] for iter in range(MAX_ITER)], c=colors[y[i]], alpha=0.5)
plt.legend(handles=handles)
plt.xlabel('Iteration')
plt.ylabel('y')
plt.savefig(os.path.join(SAVE_PATH, SAVE_NAME + '_omega_history_y.png'), bbox_inches='tight', dpi=300)

# Plot psi history
plt.figure(figsize=FIG_SIZE)
for i in range(psi_history[0].shape[0]):
    plt.plot([real_psi[i] for iter in range(MAX_ITER)], c='k', alpha=0.5, linestyle='-', linewidth=0.5)
for i in range(psi_history[0].shape[0]):
    plt.plot([psi_history[iter][i] for iter in range(MAX_ITER)], c=colors[y[i]], alpha=0.5)

plt.legend(handles=handles)
plt.xlabel('Iteration')
plt.ylabel('Aggregate')
plt.savefig(os.path.join(SAVE_PATH, SAVE_NAME + '_psi_history.png'), bbox_inches='tight', dpi=300)

