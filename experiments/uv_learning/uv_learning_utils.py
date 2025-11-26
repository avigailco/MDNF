import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def visualize_uv(points_gt, points_pred, triangles, colors_gt=None, colors_pred=None):
    """ plot gt and predicted uv coordinates """
    # Define colors based on 0/1 values: 1 is red, 0 is blue
    if colors_gt is not None and colors_pred is not None:
        vertex_colors_gt = np.where(colors_gt == 1, 'r', 'b')
        vertex_colors_pred = np.where(colors_pred == 1, 'r', 'b')

    figsize = (12, 5)  # You can adjust the width and height as needed
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Create triangulation objects for each mesh
    triangulation1 = tri.Triangulation(points_gt[:, 0], points_gt[:, 1], triangles)
    triangulation2 = tri.Triangulation(points_pred[:, 0], points_pred[:, 1], triangles)

    marker_size = 0.01  # Adjusted for visibility
    edge_linewidth = 0.05  # Larger value for thicker edges, 0.5

    # Plot the first mesh in the first subplot
    axs[0].triplot(triangulation1, 'ko-', markersize=marker_size, linewidth=edge_linewidth)
    if colors_gt is not None:
        axs[0].scatter(points_gt[:, 0], points_gt[:, 1], c=vertex_colors_gt, s=marker_size, zorder=3)
    axs[0].set_title('UV gt')
    axs[0].set_aspect('equal')

    # Plot the second mesh in the second subplot
    axs[1].triplot(triangulation2, 'ko-', markersize=marker_size, linewidth=edge_linewidth)
    if colors_pred is not None:
        axs[1].scatter(points_pred[:, 0], points_pred[:, 1], c=vertex_colors_pred, s=marker_size, zorder=3)
    axs[1].set_title('UV pred')
    axs[1].set_aspect('equal')

    plt.show()
