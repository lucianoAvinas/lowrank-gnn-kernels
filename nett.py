import numpy as np
import torch
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
import pandas as pd

# Activate the automatic conversion for pandas and numpy objects
pandas2ri.activate()
numpy2ri.activate()

# Import the necessary R packages
Matrix = importr('Matrix')
nett = importr('nett')

def sample_dcsbm(z, B):
    """
    Generate an edge list for a graph using the sample_dcsbm function from the nett package in R.

    Parameters:
    z (numpy array): Class labels for the nodes.
    B (numpy array): Probability matrix for edge creation.

    Returns:
    torch.Tensor: A 2 x n tensor representing the edge list of the graph.
    """
    # Ensure z and B are numpy arrays
    z = np.asarray(z)
    B = np.asarray(B)

    # Call the sample_dcsbm function from the nett package
    result = nett.sample_dcsbm(z + 1, B)  # z+1 to adjust for R's 1-based indexing

    # Use the summary function to get the (i, j, values) representation
    summary_result = Matrix.summary(result)

    # Convert the summary result to a pandas DataFrame
    summary_df = pandas2ri.rpy2py(summary_result)

    # Extract the (i, j) positions
    rows = summary_df['i'].values - 1  # Convert from 1-based to 0-based indexing
    cols = summary_df['j'].values - 1  # Convert from 1-based to 0-based indexing

    # Create the edge list
    edge_list = np.vstack((rows, cols))

    # Convert the edge list to a PyTorch tensor
    edge_tensor = torch.tensor(edge_list, dtype=torch.long)

    return edge_tensor

# Example usage
if __name__ == "__main__":
    # Create random samples of n with K classes
    np.random.seed(42)
    K = 3  # Number of classes
    n = 10  # Number of samples
    z = np.random.choice(K, n)

    # Create an example B matrix as a numpy array
    B = np.array([[0.5, 0.2, 0.3],
                  [0.2, 0.5, 0.3],
                  [0.3, 0.3, 0.5]])

    # Generate the edge list tensor
    edge_tensor = sample_dcsbm_to_edge_list(z, B)
    print(edge_tensor)
