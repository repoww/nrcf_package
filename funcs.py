import numpy as np
from scipy.io import loadmat # load matlab .mat file, we import the kWh data from our CALLI-VALLI circuit
import matplotlib.pyplot as plt

def obtain_network_data(Feeder_size):
    # input: Feeder_size: an integer representing the number of node in the feeder
    # the above data (bus ID, line ID, etc.) are consistent with [1,Fig.4,Table.I]
    # [1]. Network Reconfiguration of Distribution Systems Using Improved Mixed-Integer Hybrid Differential Evolution
    if Feeder_size == 16:
        bus = np.array([[1, 3, 0, 0],
                        [2, 3, 0, 0],
                        [3, 3, 0, 0],
                        [4, 1, 0.0200 + 0.0160j, 0],
                        [5, 1, 0.0300 + 0.0040j, 0],
                        [6, 1, 0.0200 - 0.0040j, 0],
                        [7, 1, 0.0150 + 0.0120j, 0],
                        [8, 1, 0.0400 + 0.0270j, 0],
                        [9, 1, 0.0500 + 0.0180j, 0],
                        [10, 1, 0.0100 + 0.0090j, 0],
                        [11, 1, 0.0060 - 0.0050j, 0],
                        [12, 1, 0.0450 - 0.0170j, 0],
                        [13, 1, 0.0100 + 0.0090j, 0],
                        [14, 1, 0.0100 - 0.0110j, 0],
                        [15, 1, 0.0100 + 0.0090j, 0],
                        [16, 1, 0.0210 - 0.0080j, 0]])
        # line format: from bus, to bus, line impedance z = r+jx (in p.u.)
        line = np.array([[1,  4,  0.0750, 0.1000],
                         [4,  5,  0.0800, 0.1100],
                         [4,  6,  0.0900, 0.1800],
                         [6,  7,  0.0400, 0.0400],
                         [2,  8,  0.1100, 0.1100],
                         [8,  9,  0.0800, 0.1100],
                         [8,  10, 0.1100, 0.1100],
                         [9,  11, 0.1100, 0.1100],
                         [9,  12, 0.0800, 0.1100],
                         [3,  13, 0.1100, 0.1100],
                         [13, 14, 0.0900, 0.1200],
                         [13, 15, 0.0800, 0.1100],
                         [15, 16, 0.0400, 0.0400],
                         [5,  11, 0.0400, 0.0400],
                         [10, 14, 0.0400, 0.0400],
                         [7,  16, 0.1200, 0.1200]])
    elif Feeder_size == 6:
        bus = np.array([[1, 3, 0.0000 + 0.0000, 0],
                        [2, 3, 0.0000 + 0.0000, 0],
                        [3, 1, 1.0000 + 0.9000, 0],
                        [4, 1, 1.0000 - 1.1000, 0],
                        [5, 1, 1.0000 + 0.9000, 0],
                        [6, 1, 2.0000 - 0.4000, 0]])
        # line format: from bus, to bus, line impedance z = r+jx (in p.u.)
        line = np.array([[1, 3, 0.1100, 0.1100],
                         [3, 4, 0.0900, 0.1200],
                         [5, 4, 0.0400, 0.0400],
                         [6, 5, 0.1100, 0.1100],
                         [2, 6, 0.1100, 0.1100]])
    elif Feeder_size == 9:
        bus = np.array([[1, 3, 0, 0],
                        [2, 3, 0, 0],
                        [3, 1, 0, 0],
                        [4, 1, 0, 0],
                        [5, 1, 0, 0],
                        [6, 1, 0, 0],
                        [7, 1, 0, 0],
                        [8, 1, 0, 0],
                        [9, 1, 0, 0]])
        line = np.array([[1, 3, 0.0750, 0.1000],
                         [3, 4, 0.0800, 0.1100],
                         [4, 5, 0.0400, 0.0400],
                         [6, 5, 0.1100, 0.1100],
                         [7, 6, 0.0800, 0.1100],
                         [2, 7, 0.1100, 0.1100],
                         [7, 8, 0.1100, 0.1100],
                         [8, 9, 0.0400, 0.0400]])
    elif Feeder_size == 11:
        bus = np.array([[1, 3, 0, 0],
                        [2, 3, 0, 0],
                        [3, 3, 0, 0],
                        [4, 1, 0, 0],
                        [5, 1, 0, 0],
                        [6, 1, 0, 0],
                        [7, 1, 0, 0],
                        [8, 1, 0, 0],
                        [9, 1, 0, 0],
                        [10, 1, 0, 0],
                        [11, 1, 0, 0]])
        line = np.array([[1, 4, 0.0750, 0.1000],
                         [4, 5, 0.0800, 0.1100],
                         [5, 6, 0.0400, 0.0400],
                         [7, 6, 0.1100, 0.1100],
                         [8, 7, 0.0800, 0.1100],
                         [2, 8, 0.1100, 0.1100],
                         [8, 9, 0.1100, 0.1100],
                         [9, 10, 0.0400, 0.0400],
                         [11, 10, 0.0900, 0.1200],
                         [3, 11, 0.1100, 0.1100]])
    elif Feeder_size == 2:
        bus = np.array([[1, 3, 0, 0],
                        [2, 1, 0, 0]])
        line = np.array([[1, 2, 0.08, 0.08]])
    v2ref = 1  # reference node voltage
    return bus, line, v2ref

def obtain_feasible_configuration(Feeder_size):
    # function: obtain_feasible_configuration: given a (meshed) network, return all its feasible configuration
    # a configuration is said to be feasible if:
    # (1). it is a (rooted) spanning forest of the network (rooted at reference nodes)
    # (2). it does not contain a singleton node
    # input: Feeder_size: an integer representing the number of node in the feeder
    # TODO: implement D.E.Knuth's generating all trees algorithm at PP 464 of TAOCP 4a
    if Feeder_size == 16:
        config = np.array([[9,8,6,15,4,16,13,12,11,10,5,1,2],
    [9,8,6,15,4,16,13,12,11,10,5,1,14],
    [9,8,6,15,4,16,13,12,11,10,5,14,2],
    [9,8,6,15,4,16,13,12,11,10,5,14,3],
    [9,8,6,15,4,16,13,12,11,10,5,3,2],
    [9,8,6,15,4,16,13,12,11,10,3,2,14],
    [9,8,6,15,4,16,13,12,11,10,3,2,7],
    [9,8,6,15,4,16,13,12,11,10,3,7,14],
    [9,8,6,15,4,16,13,12,11,10,7,14,2],
    [9,8,6,15,4,16,13,12,11,10,7,14,1],
    [9,8,6,15,4,16,13,12,11,10,7,1,2],
    [9,8,6,15,4,16,13,12,11,10,1,2,14],
    [9,8,6,15,4,16,13,12,11,1,2,14,7],
    [9,8,6,15,4,16,13,12,11,1,2,14,3],
    [9,8,6,15,4,16,13,12,11,1,2,3,7],
    [9,8,6,15,4,16,13,12,11,1,2,3,5],
    [9,8,6,15,4,16,13,12,11,1,2,5,7],
    [9,8,6,15,4,16,13,12,11,1,5,7,14],
    [9,8,6,15,4,16,13,12,11,1,5,14,3],
    [9,8,6,15,4,16,13,12,11,1,14,3,7],
    [9,8,6,15,4,16,13,12,11,14,3,7,5],
    [9,8,6,15,4,16,13,12,11,14,3,5,2],
    [9,8,6,15,4,16,13,12,11,14,5,2,7],
    [9,8,6,15,4,16,13,12,11,5,2,7,3],
    [9,8,6,15,4,16,13,12,5,2,7,3,10],
    [9,8,6,15,4,16,13,12,5,2,7,3,14],
    [9,8,6,15,4,16,13,12,5,2,7,3,1],
    [9,8,6,15,4,16,13,12,5,2,7,1,10],
    [9,8,6,15,4,16,13,12,5,2,7,10,14],
    [9,8,6,15,4,16,13,12,5,7,10,14,3],
    [9,8,6,15,4,16,13,12,5,7,10,14,1],
    [9,8,6,15,4,16,13,12,5,7,14,1,3],
    [9,8,6,15,4,16,13,12,7,14,1,3,2],
    [9,8,6,15,4,16,13,12,7,14,1,2,10],
    [9,8,6,15,4,16,13,12,7,14,2,10,3],
    [9,8,6,15,4,16,13,7,14,2,10,3,11],
    [9,8,6,15,4,16,13,7,14,2,10,3,5],
    [9,8,6,15,4,16,13,7,14,2,10,3,1],
    [9,8,6,15,4,16,13,7,14,2,3,1,11],
    [9,8,6,15,4,16,13,7,14,2,3,11,5],
    [9,8,6,15,4,16,13,7,14,3,11,5,1],
    [9,8,6,15,4,16,13,7,14,3,11,1,10],
    [9,8,6,15,4,16,13,7,14,3,1,10,5],
    [9,8,6,15,4,16,13,7,3,1,10,5,2],
    [9,8,6,15,4,16,13,7,3,1,10,2,11],
    [9,8,6,15,4,16,13,7,3,1,2,11,5],
    [9,8,6,15,4,16,13,3,1,2,11,5,10],
    [9,8,6,15,4,16,13,3,1,2,11,10,14],
    [9,8,6,15,4,16,13,3,1,11,10,14,5],
    [9,8,6,15,4,16,13,3,11,10,14,5,2],
    [9,8,6,15,4,16,3,11,10,14,5,2,12],
    [9,8,6,15,4,16,3,11,10,14,5,12,1],
    [9,8,6,15,4,16,3,11,10,14,12,1,2],
    [9,8,6,15,4,16,3,11,10,14,12,1,7],
    [9,8,6,15,4,16,3,11,10,14,12,7,2],
    [9,8,6,15,4,16,3,11,10,12,7,2,1],
    [9,8,6,15,4,16,3,11,10,12,2,1,5],
    [9,8,6,15,4,16,3,11,12,2,1,5,7],
    [9,8,6,15,4,16,3,11,12,2,1,7,14],
    [9,8,6,15,4,16,3,11,12,2,7,14,5],
    [9,8,6,15,4,16,3,11,12,7,14,5,1],
    [9,8,6,15,4,16,3,12,7,14,5,1,10],
    [9,8,6,15,4,16,3,12,7,14,5,10,2],
    [9,8,6,15,4,16,3,12,7,14,10,2,1],
    [9,8,6,15,4,16,3,12,7,10,2,1,5],
    [9,8,6,15,4,3,12,7,10,2,1,5,13],
    [9,8,6,15,4,3,12,7,10,2,1,13,14],
    [9,8,6,15,4,3,12,7,10,2,1,13,11],
    [9,8,6,15,4,3,12,7,10,2,13,11,14],
    [9,8,6,15,4,3,12,7,10,2,13,14,5],
    [9,8,6,15,4,3,12,7,10,13,14,5,1],
    [9,8,6,15,4,3,12,7,10,13,14,1,11],
    [9,8,6,15,4,3,12,7,13,14,1,11,2],
    [9,8,6,15,4,3,12,7,13,14,1,11,5],
    [9,8,6,15,4,3,12,7,13,14,11,5,2],
    [9,8,6,15,4,3,12,7,13,11,5,2,1],
    [9,8,6,15,4,3,12,13,11,5,2,1,10],
    [9,8,6,15,4,3,12,13,11,5,2,10,14],
    [9,8,6,15,4,3,12,13,11,5,10,14,1],
    [9,8,6,15,4,3,12,13,11,10,14,1,2],
    [9,8,6,15,3,12,13,11,10,14,1,2,16],
    [9,8,6,15,3,12,13,11,10,14,1,16,7],
    [9,8,6,15,3,12,13,11,10,14,1,16,5],
    [9,8,6,15,3,12,13,11,10,14,16,5,2],
    [9,8,6,15,3,12,13,11,10,14,16,2,7],
    [9,8,6,15,3,12,13,11,10,16,2,7,1],
    [9,8,6,15,3,12,13,11,10,16,2,1,5],
    [9,8,6,15,3,12,13,11,16,2,1,5,7],
    [9,8,6,15,3,12,13,11,16,2,1,7,14],
    [9,8,6,15,3,12,13,11,16,2,7,14,5],
    [9,8,6,15,3,12,13,11,16,7,14,5,1],
    [9,8,6,15,3,12,13,16,7,14,5,1,10],
    [9,8,6,15,3,12,13,16,7,14,5,10,2],
    [9,8,6,15,3,12,13,16,7,14,10,2,1],
    [9,8,6,15,3,12,13,16,7,10,2,1,5],
    [9,8,6,3,12,13,16,7,10,2,1,5,11],
    [9,8,6,3,12,13,16,7,10,2,1,11,14],
    [9,8,6,3,12,13,16,7,10,2,11,14,5],
    [9,8,6,3,12,13,16,7,10,2,11,14,4],
    [9,8,6,3,12,13,16,7,10,2,11,4,5],
    [9,8,6,3,12,13,16,7,10,11,4,5,14],
    [9,8,6,3,12,13,16,7,10,11,5,14,1],
    [9,8,6,3,12,13,16,7,11,5,14,1,4],
    [9,8,6,3,12,13,16,7,11,5,14,4,2],
    [9,8,6,3,12,13,16,7,11,5,4,2,1],
    [9,8,6,3,12,13,16,7,11,4,2,1,14],
    [9,8,6,3,12,13,7,11,4,2,1,14,10],
    [9,8,6,3,12,13,7,11,4,2,1,10,5],
    [9,8,6,3,12,13,7,11,4,2,10,5,14],
    [9,8,6,3,12,13,7,11,4,10,5,14,1],
    [9,8,6,3,12,7,11,4,10,5,14,1,16],
    [9,8,6,3,12,7,11,4,10,5,14,16,2],
    [9,8,6,3,12,7,11,4,10,5,16,2,1],
    [9,8,6,3,12,7,11,4,10,16,2,1,14],
    [9,8,6,3,7,11,4,10,16,2,1,14,13],
    [9,8,6,3,7,11,4,10,16,2,1,13,5],
    [9,8,6,3,7,11,4,10,16,2,13,5,14],
    [9,8,6,3,7,11,4,10,16,13,5,14,1],
    [9,8,6,7,11,4,10,16,13,5,14,1,12],
    [9,8,6,7,11,4,10,16,13,5,14,12,2],
    [9,8,6,7,11,4,10,16,13,5,12,2,1],
    [9,8,6,7,11,4,10,16,13,12,2,1,14],
    [9,8,7,11,4,10,16,13,12,2,1,14,15],
    [9,8,7,11,4,10,16,13,12,2,1,14,5],
    [9,8,7,11,4,10,16,13,12,2,14,5,3],
    [9,8,7,11,4,10,16,13,12,2,14,3,15],
    [9,8,7,11,4,10,16,13,2,14,3,15,1],
    [9,8,7,11,4,10,16,13,2,14,3,1,5],
    [9,8,7,11,4,10,16,2,14,3,1,5,12],
    [9,8,7,11,4,10,16,2,14,3,1,12,15],
    [9,8,7,11,4,10,2,14,3,1,12,15,13],
    [9,8,7,11,4,10,2,14,3,1,12,13,5],
    [9,8,7,11,4,2,14,3,1,12,13,5,15],
    [9,8,7,11,4,2,14,3,1,12,13,5,16],
    [9,8,7,11,4,2,14,3,1,12,13,16,15],
    [9,8,7,11,4,2,14,3,1,12,16,15,5],
    [9,8,7,11,4,2,14,3,1,16,15,5,13],
    [9,8,7,11,4,2,14,3,16,15,5,13,12],
    [9,8,7,11,4,2,14,16,15,5,13,12,1],
    [9,8,7,11,2,14,16,15,5,13,12,1,3],
    [9,8,7,11,2,14,16,15,13,12,1,3,10],
    [9,8,7,11,2,14,16,13,12,1,3,10,5],
    [9,8,7,2,14,16,13,12,1,3,10,5,15],
    [9,8,7,2,14,16,13,12,1,3,5,15,4],
    [9,8,7,2,14,16,13,12,1,5,15,4,10],
    [9,8,7,2,14,16,13,12,5,15,4,10,3],
    [9,8,7,2,14,16,13,5,15,4,10,3,1],
    [9,8,7,2,14,16,5,15,4,10,3,1,12],
    [9,8,7,2,14,5,15,4,10,3,1,12,13],
    [9,8,2,14,5,15,4,10,3,1,12,13,11],
    [9,8,2,14,5,15,4,10,3,1,12,11,16],
    [9,8,2,14,5,15,4,10,3,1,11,16,13],
    [9,8,2,14,5,15,4,10,3,11,16,13,12],
    [9,8,2,14,5,15,4,10,11,16,13,12,1],
    [9,8,2,14,5,15,4,11,16,13,12,1,3],
    [9,8,2,14,5,15,11,16,13,12,1,3,10],
    [9,2,14,5,15,11,16,13,12,1,3,10,6],
    [9,2,14,5,15,11,16,13,12,1,3,6,4],
    [9,2,14,5,15,11,16,13,12,1,3,6,7],
    [9,2,14,5,15,11,16,13,12,1,6,7,4],
    [9,2,14,5,15,11,16,13,12,1,6,4,10],
    [9,2,14,5,15,11,16,13,12,6,4,10,3],
    [9,2,14,5,15,11,16,13,12,6,4,3,7],
    [9,2,14,5,15,11,16,13,6,4,3,7,1],
    [9,2,14,5,15,11,16,13,6,4,3,1,10],
    [9,2,14,5,15,11,16,6,4,3,1,10,12],
    [9,2,14,5,15,11,16,6,4,3,1,12,7],
    [9,2,14,5,15,11,6,4,3,1,12,7,13],
    [9,2,14,5,15,11,6,4,3,1,12,13,10],
    [9,2,14,5,15,6,4,3,1,12,13,10,7],
    [9,2,14,5,15,6,4,3,1,12,13,7,16],
    [9,2,14,5,15,6,4,3,1,12,7,16,10],
    [9,2,14,5,15,6,4,3,1,7,16,10,13],
    [9,2,14,5,15,6,4,3,7,16,10,13,12],
    [9,2,14,5,15,6,4,7,16,10,13,12,1],
    [9,2,14,5,15,6,7,16,10,13,12,1,3],
    [9,2,14,5,6,7,16,10,13,12,1,3,11],
    [9,2,14,5,6,7,16,10,13,12,1,11,4],
    [9,2,14,5,6,7,16,10,13,12,11,4,3],
    [9,2,14,5,6,7,16,10,13,11,4,3,1],
    [9,2,14,5,6,7,16,10,11,4,3,1,12],
    [9,2,14,5,6,7,16,11,4,3,1,12,13],
    [9,2,14,5,6,7,11,4,3,1,12,13,10],
    [9,2,14,6,7,11,4,3,1,12,13,10,15],
    [9,2,14,6,7,11,4,3,1,12,13,15,16],
    [9,2,14,6,7,11,4,3,1,12,15,16,10],
    [9,2,14,6,7,11,4,3,1,15,16,10,13],
    [9,2,14,6,7,11,4,3,15,16,10,13,12],
    [9,2,14,6,7,11,4,15,16,10,13,12,1],
    [9,2,14,6,7,11,15,16,10,13,12,1,3],]) - 1 # the -1 is for matching python indexing convention
    elif Feeder_size == 6:
        config = np.array([[4, 5, 1, 2],
                           [4, 5, 1, 3],
                           [4, 5, 3, 2],
                           [4, 3, 2, 1],
                           [3, 2, 1, 5]]) - 1  # the -1 is for matching python indexing convention
    elif Feeder_size == 9:
        config = np.array([[4, 8, 7, 5, 6, 1, 2, ],
                           [4, 8, 7, 5, 6, 1, 3, ],
                           [4, 8, 7, 5, 6, 3, 2, ],
                           [4, 8, 7, 5, 3, 2, 1, ],
                           [4, 8, 7, 3, 2, 1, 6, ],
                           [8, 7, 3, 2, 1, 6, 5, ]]) - 1  # the -1 is for matching python indexing convention
    elif Feeder_size == 11:
        config = np.array([[4,5,8,9,10,6,1,2],
                            [4,5,8,9,10,6,1,3],
                            #[4,5,8,9,10,6,3,2],
                            [4,5,8,9,10,3,2,1],
                            #[4,5,8,9,10,3,2,7],
                            [4,5,8,9,10,3,7,1],
                            [4,5,8,9,10,7,1,2],
                            [4,5,8,9,7,1,2,3],
                            [4,5,8,9,7,1,2,6],
                            [4,5,8,9,7,1,6,3],
                            #[4,5,8,9,7,6,3,2],
                            #[4,5,8,7,6,3,2,10],
                            [4,5,8,7,6,3,10,1],
                            [4,5,8,7,6,10,1,2],
                            [4,5,8,7,10,1,2,3],
                            [4,5,7,10,1,2,3,9],
                            [4,5,7,10,1,2,9,6],
                            [4,5,7,10,1,9,6,3],
                            #[4,5,7,10,9,6,3,2],
                            [4,7,10,9,6,3,2,1],
                            [4,7,10,9,3,2,1,8],
                            [4,7,10,3,2,1,8,6],
                            [4,7,3,2,1,8,6,9],
                            [4,3,2,1,8,6,9,10],
                            [3,2,1,8,6,9,10,5],
                            [3,2,1,8,6,9,5,7],
                            [3,2,1,8,6,5,7,10],
                            [3,2,1,8,5,7,10,9],
                            [3,2,1,5,7,10,9,6]]) - 1  # the -1 is for matching python indexing convention
    elif Feeder_size == 2:
        config = np.array([[1]]) - 1
    return config

def obtain_load_time_series(Feeder_size, num_time_series):
    cust = loadmat("./yuanqi_data/CALLI_VALLI/cust.mat")
    tran = loadmat("./yuanqi_data/CALLI_VALLI/tran.mat")
    Goodtran = loadmat("./yuanqi_data/CALLI_VALLI/Goodtran.mat")
    cust = cust["cust"]
    tran = tran["tran"]
    Goodtran = Goodtran["Goodtran"]
    cust = cust[0]
    tran = tran[0]
    Goodtran = Goodtran[0]
    if Feeder_size == 16:
        num_tran_needed = 13
    elif Feeder_size == 6:
        num_tran_needed = 4
    elif Feeder_size == 9:
        num_tran_needed = 7
    elif Feeder_size == 11:
        num_tran_needed = 8
    elif Feeder_size == 2:
        num_tran_needed = 1

    P_out = np.zeros((168, 0, num_tran_needed))  # 1st dim: number of hours in a week; 2nd dim: to be h-stacked; 3rd dim: num_tran_needed
    Q_out = np.zeros((168, 0, num_tran_needed))
    tran_output = np.array([-1])


    for time_series in range(num_time_series):

        y = {}
        cust_cnt = 0
        for time_series_inner in range(num_tran_needed):
            y[time_series_inner] = np.zeros((4440, 1))
            for ii in range(30):
                if len(cust[ii+cust_cnt]["t_kWh"][0][0]) == 4440:
                    y[time_series_inner] = y[time_series_inner] + cust[ii+cust_cnt]["kWh"][0][0]
                    cust_cnt += 1
        p = y
        q = {}
        for ii in range(len(p)):
            q[ii] = p[ii] * 0.2

        # reshape into episode-time_stamp-load node form
        P = np.empty(shape=(7 * 24, len(p[0]) // (7 * 24), num_tran_needed))  # numpy.empty: Return a new array of given shape and type, without initializing entries.
        Q = np.empty(shape=(7 * 24, len(q[0]) // (7 * 24), num_tran_needed))
        for ii in range(len(p)):
            for jj in range(len(p[0]) // (7 * 24)):
                P[:, jj:jj + 1, ii] = p[ii][jj * (7 * 24):(jj + 1) * (7 * 24)]
                Q[:, jj:jj + 1, ii] = q[ii][jj * (7 * 24):(jj + 1) * (7 * 24)]

        P_out = np.hstack((P_out, P))
        Q_out = np.hstack((Q_out, Q))
    Sbase = 100000
    if Feeder_size == 16:
        load_normalize = Feeder_size*np.max(P_out)
    elif Feeder_size == 6:
        load_normalize = Feeder_size*np.max(P_out)  # TODO
    elif Feeder_size == 9:
        load_normalize = Feeder_size*np.max(P_out)  # TODO
    elif Feeder_size == 11:
        load_normalize = Feeder_size*np.max(P_out)*2  # TODO
    elif Feeder_size == 2:
        load_normalize = Feeder_size*np.max(P_out)/6  # TODO
    P_out = P_out / load_normalize
    Q_out = Q_out / load_normalize
    # convert from load to net injection
    P_out = -P_out
    Q_out = -Q_out
    return P_out, Q_out, Sbase, tran_output

def load_flow(state, action, Feeder_size):
    bus, line_ori, v2ref = obtain_network_data(Feeder_size)
    config = obtain_feasible_configuration(Feeder_size)
    line = line_ori[config[action, :], :]

    num_edge = line.shape[0]
    num_node = bus.shape[0]
    num_ref_node = np.size(np.where(bus[:, 1] == 3))
    num_nonref_node = num_node - num_ref_node

    P_spec = state[:len(state) // 2]  # the operator // returns integer
    Q_spec = state[len(state) // 2:]  # P and Q in p.u.

    App = np.zeros((num_nonref_node, num_edge))
    Apq = np.zeros((num_nonref_node, num_edge))
    Apv = np.zeros((num_nonref_node, num_node))
    Apl = np.zeros((num_nonref_node, num_edge))
    Aqp = np.zeros((num_nonref_node, num_edge))
    Aqq = np.zeros((num_nonref_node, num_edge))
    Aqv = np.zeros((num_nonref_node, num_node))
    Aql = np.zeros((num_nonref_node, num_edge))
    Avp = np.zeros((num_ref_node, num_edge))
    Avq = np.zeros((num_ref_node, num_edge))
    Avv = np.zeros((num_ref_node, num_node))
    Avl = np.zeros((num_ref_node, num_edge))
    Alp = np.zeros((num_nonref_node, num_edge))
    Alq = np.zeros((num_nonref_node, num_edge))
    Alv = np.zeros((num_nonref_node, num_node))
    All = np.zeros((num_nonref_node, num_edge))

    for ii in range(num_ref_node, num_node):  # looping over non-reference nodes
        i_set = np.where(line[:, 0] == bus[ii, 0])
        j_set = np.where(line[:, 1] == bus[ii, 0])

        # first linear eqn
        if i_set[0].size != 0:
            App[ii - num_ref_node, i_set] = 1
        if j_set[0].size != 0:
            App[ii - num_ref_node, j_set] = -1
            Apl[ii - num_ref_node : ii-num_ref_node+1, j_set] = line[j_set, 2:3][0].T

        # second linear eqn
        if i_set[0].size != 0:
            Aqq[ii - num_ref_node, i_set] = 1
        if j_set[0].size != 0:
            Aqq[ii - num_ref_node, j_set] = -1
            Aql[ii - num_ref_node : ii-num_ref_node+1, j_set] = line[j_set, 3:4][0].T

    # third linear eqn
    Avv[:num_ref_node, :num_ref_node] = np.identity(num_ref_node)

    # forth linear eqn
    for ij in range(num_edge):
        Alp[ij, ij] = -2 * line[ij, 2]
        Alq[ij, ij] = -2 * line[ij, 3]
        Alv[ij, int(line[ij, 0] - 1)] = 1  # from node， the minus 1 is for matching python indexing convention
        Alv[ij, int(line[ij, 1] - 1)] = -1  # to node
        All[ij, ij] = line[ij, 2] ** 2 + line[ij, 3] ** 2

    A = np.vstack((np.hstack((App, Apq, Apv, Apl)),
                   np.hstack((Aqp, Aqq, Aqv, Aql)),
                   np.hstack((Avp, Avq, Avv, Avl)),
                   np.hstack((Alp, Alq, Alv, All))))
    b = np.vstack((P_spec.reshape((len(P_spec), 1)),
                   Q_spec.reshape((len(Q_spec), 1)),
                   np.ones((num_ref_node, 1)) * v2ref,
                   np.zeros((num_edge, 1))))
    # initial guess
    Pij_ini = np.zeros((num_edge, 1))
    Qij_ini = np.zeros((num_edge, 1))
    Vi2_ini = np.ones((num_node, 1)) * v2ref
    Lij_ini = np.zeros((num_edge, 1))
    x = np.vstack((Pij_ini, Qij_ini, Vi2_ini, Lij_ini))
    # being N-R procedure
    dx = [1000, 1]
    iter = 0
    convergence_flag = 1 # if convergence_flag == 1, then it is converged
    while max(np.absolute(dx)) > 0.00000001:
        # mismatch vector of the quadratic equation
        Dg = x[:num_edge] ** 2 + x[num_edge:2 * num_edge] ** 2 - (
            np.multiply(x[2 * num_edge + num_node:], x[2 * num_edge + (line[:, 0] - 1).astype(
                int)]))  # numpyArray.astype(int): convert to integer element-wise
        # mismatch jacobian of the quadratic equation
        Gp = np.zeros((num_edge, num_edge))
        Gq = np.zeros((num_edge, num_edge))
        Gv = np.zeros((num_edge, num_node))
        Gl = np.zeros((num_edge, num_edge))
        for ij in range(num_edge):
            Gp[ij, ij] = 2 * x[ij]  # 2*pij
            Gq[ij, ij] = 2 * x[ij + num_edge]
            Gv[ij, int(line[ij, 0] - 1)] = -x[2 * num_edge + num_node + ij]  # the minus 1 is for matching python indexing convention
            Gl[ij, ij] = -x[2 * num_edge + int(line[ij, 0] - 1)]  # the minus 1 is for matching python indexing convention
        DG = np.hstack((Gp, Gq, Gv, Gl))
        # overall mismatch vector
        Df = np.vstack((A.dot(x) - b, Dg))  # where A.dot(x) is matrix-vector multi for numpy array A and x
        # overall mismatch Jacobian
        J = np.vstack((A, DG))
        # solve
        dx = np.linalg.solve(-J, Df)
        # update
        x = x + dx
        iter += 1
        if iter > 20:
            #print("power flow does not converge")
            convergence_flag = 0
            break
    Pij = x[:num_edge]
    Qij = x[num_edge:2 * num_edge]
    Vi2 = x[2 * num_edge:2 * num_edge + num_node]
    Lij = x[2 * num_edge + num_node:]
    if convergence_flag == 1:  # if the load flow converges
        total_loss = line[:, 2].dot(Lij)
    else:
        total_loss = 1
        Vi2 = np.zeros(shape=Vi2.shape)
    if np.isnan(Vi2).any():
        print('NaN in funcs.py: Voltage')
    if np.isnan(total_loss):
        print('NaN in funcs.py: loss')
    return Pij, Qij, Vi2, Lij, total_loss, iter, convergence_flag

def continuation_load_flow(state, action, Feeder_size, node_inquiry):
    # func: continuation_load_flow: for the given network and loading condition, find the load flow solution
    # using the ideal proposed in:
    # the continuation power flow: a tool for steady state voltage stability analysis by Dr.Ajjarapu
    # inputs: state, action, Feeder_size: as ditto func: load_flow
    #         node_inquiry:
    # outputs: Pij, Qij, Vi2, Lij, total_loss, iter, convergence_flag: as ditto func: load_flow
    #         p_node_inquiry, v_node_inquiry: PV curve at the node of inquiry
    # step 1: run conventional power flow on base case, the following while loop continues until a
    #         converging load flow is found
    p_node_inquiry = np.array([])
    v_node_inquiry = np.array([])
    convergence_flag = 0
    dividing = 2
    while convergence_flag == 0:
        Pij, Qij, Vi2, Lij, total_loss, iter, convergence_flag = load_flow(state/dividing, action, Feeder_size)
        if convergence_flag == 1:
            break
        else:
            dividing += 1

    # step 2: obtain problem data:
    bus, line_ori, v2ref = obtain_network_data(Feeder_size)
    config = obtain_feasible_configuration(Feeder_size)
    line = line_ori[config[action, :], :]
    num_edge = line.shape[0]
    num_node = bus.shape[0]
    num_ref_node = np.size(np.where(bus[:, 1] == 3))
    num_nonref_node = num_node - num_ref_node
    P_spec = state[:len(state) // 2]  # the operator // returns integer
    Q_spec = state[len(state) // 2:]  # P and Q in p.u.
    P_spec_0 = P_spec / dividing  # base case net injection
    Q_spec_0 = Q_spec / dividing

    K = 10  # please refer to the original paper for the meaning of K
    # first linear eqn
    App = np.zeros((num_nonref_node, num_edge))
    Apq = np.zeros((num_nonref_node, num_edge))
    Apv = np.zeros((num_nonref_node, num_node))
    Apl = np.zeros((num_nonref_node, num_edge))
    Aplambda = K * (-P_spec + P_spec_0).reshape(
        (-1, 1))  # note that, in our case we assume non-ref bus only have loads but no generation
    # second linear eqn
    Aqp = np.zeros((num_nonref_node, num_edge))
    Aqq = np.zeros((num_nonref_node, num_edge))
    Aqv = np.zeros((num_nonref_node, num_node))
    Aql = np.zeros((num_nonref_node, num_edge))
    Aqlambda = K * (-Q_spec + Q_spec_0).reshape((-1, 1))
    # third linear eqn
    Avp = np.zeros((num_ref_node, num_edge))
    Avq = np.zeros((num_ref_node, num_edge))
    Avv = np.zeros((num_ref_node, num_node))
    Avl = np.zeros((num_ref_node, num_edge))
    Avlambda = np.zeros((num_ref_node, 1))
    # forth linear eqn
    Alp = np.zeros((num_nonref_node, num_edge))
    Alq = np.zeros((num_nonref_node, num_edge))
    Alv = np.zeros((num_nonref_node, num_node))
    All = np.zeros((num_nonref_node, num_edge))
    Allambda = np.zeros((num_nonref_node, 1))

    for ii in range(num_ref_node, num_node):  # looping over non-reference nodes
        i_set = np.where(line[:, 0] == bus[ii, 0])
        j_set = np.where(line[:, 1] == bus[ii, 0])

        # first linear eqn
        if i_set[0].size != 0:
            App[ii - num_ref_node, i_set] = 1
        if j_set[0].size != 0:
            App[ii - num_ref_node, j_set] = -1
            Apl[ii - num_ref_node: ii - num_ref_node + 1, j_set] = line[j_set, 2:3][0].T

        # second linear eqn
        if i_set[0].size != 0:
            Aqq[ii - num_ref_node, i_set] = 1
        if j_set[0].size != 0:
            Aqq[ii - num_ref_node, j_set] = -1
            Aql[ii - num_ref_node: ii - num_ref_node + 1, j_set] = line[j_set, 3:4][0].T

    # third linear eqn
    Avv[:num_ref_node, :num_ref_node] = np.identity(num_ref_node)

    # forth linear eqn
    for ij in range(num_edge):
        Alp[ij, ij] = -2 * line[ij, 2]
        Alq[ij, ij] = -2 * line[ij, 3]
        Alv[ij, int(line[ij, 0] - 1)] = 1  # from node， the minus 1 is for matching python indexing convention
        Alv[ij, int(line[ij, 1] - 1)] = -1  # to node
        All[ij, ij] = line[ij, 2] ** 2 + line[ij, 3] ** 2

    A = np.vstack((np.hstack((App, Apq, Apv, Apl, Aplambda)),
                   np.hstack((Aqp, Aqq, Aqv, Aql, Aqlambda)),
                   np.hstack((Avp, Avq, Avv, Avl, Avlambda)),
                   np.hstack((Alp, Alq, Alv, All, Allambda))))
    b = np.concatenate((P_spec_0.reshape((len(P_spec_0), )),
                   Q_spec_0.reshape((len(Q_spec_0), )),
                   np.ones((num_ref_node, )) * v2ref,
                   np.zeros((num_edge, ))))
    # initial point for the predictor, which is the solution of the power flow in the base case P_spec_0 and Q_spec_0
    Pij_ini = Pij
    Qij_ini = Qij
    Vi2_ini = Vi2
    Lij_ini = Lij
    lambda_ini = 0
    x = np.vstack((Pij_ini, Qij_ini, Vi2_ini, Lij_ini, lambda_ini))
    x = np.squeeze(x)  # remove single-dimensional entries from the shape of an array

    x_bar_lambda = 100
    iter = -1
    while x_bar_lambda > 0.0:
        iter += 1
        # the nonlinear eqn, linearized around the initial point
        Gp = np.zeros((num_edge, num_edge))
        Gq = np.zeros((num_edge, num_edge))
        Gv = np.zeros((num_edge, num_node))
        Gl = np.zeros((num_edge, num_edge))
        Glambda = np.zeros((num_edge, 1))
        for ij in range(num_edge):
            Gp[ij, ij] = 2 * x[ij]  # 2*pij
            Gq[ij, ij] = 2 * x[ij + num_edge]
            Gv[ij, int(line[ij, 0] - 1)] = -x[
                2 * num_edge + num_node + ij]  # the minus 1 is for matching python indexing convention
            Gl[ij, ij] = -x[2 * num_edge + int(line[ij, 0] - 1)]  # the minus 1 is for matching python indexing convention
        DG = np.hstack((Gp, Gq, Gv, Gl, Glambda))

        # jacobian of the lambda-augmented system
        if iter == 0: # if this is the first iteration
            idx_k = x.size - 1
        else:
            idx_k = np.argmax(np.abs(x_bar))
        ek = np.zeros(shape=(A.shape[1],))
        ek[idx_k] = 1
        F = np.vstack((A, DG, ek))

        # RHS of the (tangent form) lambda-augmented system
        z = np.zeros(shape=(F.shape[0],))
        if iter == 0: # if this is the first iteration
            z[-1] = 1
        else:
            if x_bar[idx_k] > 0:
                z[-1] = 1
            else:
                z[-1] = -1
        x_bar = np.linalg.solve(F, z)
        x_bar_lambda = x_bar[-1]

        # one-step prediction
        sigma = 0.01  # prediction step size
        x_star = x + sigma*x_bar

        # corrector step
        dx_inner = [1000, 1]
        x = x_star  # initial guess (the predicted step)
        iter_inner = -1
        while max(np.absolute(dx_inner)) > 0.00001:
            iter_inner += 1
            # re-evaluate the system of equations
            # mismatch vector of the quadratic equation
            Dg = x[:num_edge]**2 + x[num_edge:2*num_edge]**2 - (
                np.multiply(x[2*num_edge+num_node:3*num_edge+num_node], x[2*num_edge+(line[:, 0]-1).astype(int)]))
            # mismatch jacobian of the quadratic equation
            Gp = np.zeros((num_edge, num_edge))
            Gq = np.zeros((num_edge, num_edge))
            Gv = np.zeros((num_edge, num_node))
            Gl = np.zeros((num_edge, num_edge))
            Glambda = np.zeros((num_edge, 1))
            for ij in range(num_edge):
                Gp[ij, ij] = 2 * x[ij]  # 2*pij
                Gq[ij, ij] = 2 * x[ij + num_edge]
                Gv[ij, int(line[ij, 0] - 1)] = -x[2 * num_edge + num_node + ij]  # the minus 1 is for matching python indexing convention
                Gl[ij, ij] = -x[2 * num_edge + int(line[ij, 0] - 1)]  # the minus 1 is for matching python indexing convention
            DG = np.hstack((Gp, Gq, Gv, Gl, Glambda))

            # mismatch vector of the lambda-augmented system (one more equation is needed before solving the system)
            Df = np.concatenate((A.dot(x) - b, Dg, [x[idx_k]-x_star[idx_k]]))

            # mismatch jacobian of the lambda-augmented system
            J = np.vstack((A, DG))
            vec = np.zeros(shape=(J.shape[1],))
            vec[idx_k] = 1
            J = np.vstack((J, vec))

            # solve for the predictor increment
            dx_inner = np.linalg.solve(-J, Df)
            # update
            x = x + dx_inner
            if iter_inner > 20:
                print("the corrector cannot converge, please select a smaller sigma to ensure the predictor enters")
                print("the radius of convergence of corrector")

        # demand at the current lambda value
        p_load = P_spec_0[node_inquiry - num_ref_node - 1] + x[-1]*K*(P_spec[node_inquiry - num_ref_node - 1] -
                                                                      P_spec_0[node_inquiry - num_ref_node - 1])
        p_load = -p_load
        p_node_inquiry = np.append(p_node_inquiry, p_load)  # the minus 1 is for matching python convention
        v_node_inquiry = np.append(v_node_inquiry, np.sqrt(x[2*num_edge + node_inquiry - 1]))
        # the minus 1 is for matching python convention

        if iter > 1000:
            print("the continuation power flow does not converge")
            convergence_flag = 0
            break
    Pij = x[:num_edge]
    Qij = x[num_edge:2 * num_edge]
    Vi2 = x[2 * num_edge:2 * num_edge + num_node]
    Lij = x[2 * num_edge + num_node:3 * num_edge + num_node]
    lambda_param = x[-1]
    total_loss = line[:, 2].dot(Lij)
    return Pij, Qij, Vi2, Lij, lambda_param, total_loss, iter, convergence_flag, p_node_inquiry, v_node_inquiry

def network_loss(state, action, Feeder_size):
    # function network_loss: given state and action, compute network loss
    # input: state: numpy array of the form: np.vstack((real power injection, reactive power injection)), all in p.u.
    #        action: a positive integer indicate the network configuration, for example, 1 represents config 1
    # output: loss: total loss of the network
    #         convergence: a binary variable. convergence = 1 if the load flow converges; = 0 if the load flow
    #                      does not converge
    _, _, _, _, loss, _, convergence_flag = load_flow(state, action, Feeder_size)
    return loss, convergence_flag

def optimal_network_loss(state, Feeder_size):
    # function optimal_network_loss: find the theoretical optimal loss value
    # input: state: numpy array of the form: np.vstack((real power injection, reactive power injection)), all in p.u.
    # output: opt_config: optimal network configuration
    #         opt_loss: loss of the optimal network configuration
    #         convergence: a binary variable. convergence = 1 if load flow for all configurations for the given state
    #               converge; otherwise
    if Feeder_size == 16:
        num_output = 190  # number of feasible configurations
    elif Feeder_size == 6:
        num_output = 5
    elif Feeder_size == 9:
        num_output = 6
    elif Feeder_size == 11:
        num_output = 29
    loss_all = np.zeros((num_output, 1))
    convergence = 1
    for ii in range(num_output):
        _, _, _, _, loss, _, convergence_flag = load_flow(state, ii, Feeder_size)
        loss_all[ii] = loss
        convergence = convergence * convergence_flag
    opt_loss = np.asscalar(min(loss_all))  # np.asscalar: convert np.array of size 1 to scalar
    opt_config = loss_all.argmin()  # return the index of the minimum element
    return opt_config, opt_loss, convergence

def reward(state, action, action_prev, w_violation, w_switch, volt2_dev, curr2_dev, v2ref, Feeder_size):
    _, _, Vi2, Lij, loss, _, convergence_flag = load_flow(state, action, Feeder_size)

    config = obtain_feasible_configuration(Feeder_size)

    volt2_vio = np.sum((np.absolute(Vi2-v2ref) - volt2_dev).clip(min=0))
    curr2_vio = np.sum((np.absolute(Lij-0) - curr2_dev).clip(min=0))
    if convergence_flag == 1:  # if the load flow converges properly
        r = -(loss +  # power loss term
              w_violation*(volt2_vio + curr2_vio) +  # voltage and current violation term
              w_switch*np.setdiff1d(config[action, :], config[action_prev, :]).size*2) * 100
    else:  # if the load flow does not converge
        r = -loss * 100
    return r


def reward_quick(Loss_allstate, config, state_ID, action, action_prev, w_switch):
    # function: reward_quick: calculate reward faster than reward() by utilizing pre-solved power flow results
    loss = Loss_allstate[state_ID, action]
    r = -(loss + w_switch*np.setdiff1d(config[action, :], config[action_prev, :]).size*2)
    r = r * 50
    return r

def time_series_forecasting_initial(P, Q):
    num_time_series = P.shape[2]
    P_D1_24  = np.zeros(shape=(24,  num_time_series))
    P_W1_168 = np.zeros(shape=(168, num_time_series))
    Q_D1_24  = np.zeros(shape=(24,  num_time_series))
    Q_W1_168 = np.zeros(shape=(168, num_time_series))

    week_0 = 0
    week_1 = 1  # according to [1], two weeks of initial time series data is required for the initialization of L D W
    P_ini = np.vstack((P[:, week_0, :], P[:, week_1, :]))
    Q_ini = np.vstack((Q[:, week_0, :], Q[:, week_1, :]))

    # obtain L0:
    P_L0 = np.average(P_ini, axis=0)
    Q_L0 = np.average(Q_ini, axis=0)

    # obtain D1_24
    for step in range(24):  # iterate over 24 hours
        for jj in range(14):  # iterate over 14 days of the two weeks
            P_D1_24[step, :] = P_D1_24[step, :] + P_ini[step+jj*24, :]/np.average(P_ini[max(step+jj*24-12,0):min(step+jj*24+12,24*14),:],axis=0)
            Q_D1_24[step, :] = Q_D1_24[step, :] + Q_ini[step+jj*24, :]/np.average(Q_ini[max(step+jj*24-12,0):min(step+jj*24+12,24*14),:],axis=0)
        P_D1_24[step, :] = P_D1_24[step, :]/14
        Q_D1_24[step, :] = Q_D1_24[step, :]/14

    # obtain W1_168
    for step in range(168):  # iterate over 168 hours
        for jj in range(2):  # iterate over 2 weeks of the two weeks
            P_W1_168[step, :] = P_W1_168[step, :] + P_ini[step+jj*168, :]/np.average(P_ini[max(step+jj*168-84,0):min(step+jj*168+84,168*2),:],axis=0)
            Q_W1_168[step, :] = Q_W1_168[step, :] + P_ini[step+jj*168, :]/np.average(P_ini[max(step+jj*168-84,0):min(step+jj*168+84,168*2),:],axis=0)
        P_W1_168[step, :] = P_W1_168[step, :]/2
        Q_W1_168[step, :] = Q_W1_168[step, :]/2
    return P_L0, P_D1_24, P_W1_168, Q_L0, Q_D1_24, Q_W1_168

def time_series_forecast_24_steps(P_L0, P_D1_24, P_W1_168, Q_L0, Q_D1_24, Q_W1_168, P_nextday, Q_nextday, day_num):
    num_time_series = P_L0.size
    h = 24  # number of steps to be forecasted ahead
    P_h = np.zeros(shape=(h, num_time_series))
    Q_h = np.zeros(shape=(h, num_time_series))

    # obtain the forecast P_h and Q_h
    for step in range(h):
        P_h[step, :] = P_L0 * P_D1_24[step, :] * P_W1_168[step + day_num*24, :]
        Q_h[step, :] = Q_L0 * Q_D1_24[step, :] * Q_W1_168[step + day_num*24, :]

    # update the state variables, i.e. L, D, W
    alpha = 0.2
    delta = 0.2
    omega = 0.2
    for step in range(h):
        # update L:
        P_L0 = alpha * (P_nextday[step, :]/(P_D1_24[step, :]*P_W1_168[step + day_num*24, :])) + (1-alpha) * P_L0
        Q_L0 = alpha * (Q_nextday[step, :]/(Q_D1_24[step, :]*Q_W1_168[step + day_num*24, :])) + (1-alpha) * Q_L0
        # update D:
        P_D_temp = delta * (P_nextday[step, :]/(P_L0*P_W1_168[step + day_num*24, :])) + (1-delta) * P_D1_24[step, :]
        Q_D_temp = delta * (Q_nextday[step, :]/(Q_L0*Q_W1_168[step + day_num*24, :])) + (1-delta) * Q_D1_24[step, :]
        # update W:
        P_W1_168[step + day_num*24, :] = omega * (P_nextday[step, :]/(P_L0*P_D1_24[step, :])) + (1-omega) * P_W1_168[step + day_num*24, :]
        Q_W1_168[step + day_num*24, :] = omega * (Q_nextday[step, :]/(Q_L0*Q_D1_24[step, :])) + (1-omega) * Q_W1_168[step + day_num*24, :]

        P_D1_24[step, :] = P_D_temp
        Q_D1_24[step, :] = Q_D_temp
    return P_L0, P_D1_24, P_W1_168, Q_L0, Q_D1_24, Q_W1_168, P_h, Q_h

