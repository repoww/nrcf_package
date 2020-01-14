import numpy as np
import funcs as fn

Feeder_size = 16  # 16 bus system
#Feeder_size = 6  # 6 bus system
#Feeder_size = 9  # 9 bus system
#Feeder_size = 11  # 11 bus system
if Feeder_size == 16:
    num_edge = 13
    num_node = 16
    num_nonref_node = 13
elif Feeder_size == 6:  # 6 bus system
    num_edge = 4  # note that, num_edge is the number of edges in the reference nodes-merged graph
    num_node = 6
    num_nonref_node = 4
elif Feeder_size == 9:  # 9 bus system
    num_edge = 7
    num_node = 9
    num_nonref_node = 7
elif Feeder_size == 11:  # 11 bus system
    num_edge = 8  # number of edges in any of the spanning forest, (reference nodes are not merged)
    num_node = 11  # total number of buses
    num_nonref_node = 8  # number of load buses
num_outputs = fn.obtain_feasible_configuration(Feeder_size).shape[0]
num_ref_node = num_node - num_nonref_node

iteration = 0
Loss_allstate = np.zeros((0, num_outputs))  # Loss_allstate[i,j] is the loss of configuration j at state i, if the power flow does not converge, then Loss_allstate[i,j] is a big number
Rank_allstate = np.zeros((0, num_outputs))  # Rank_allstate[i,j] is the ranking of loss of configuration j at state i
Volt_allstate = np.zeros((0, num_outputs, num_node))  # Volt_allstate[i,j,k] is the voltage magnitude (not squared) of node k, configuration j at state i, if the power flow does not converge, then Volt_allstate[i,j] is 0
P, Q, Sbase, tran_selected = fn.obtain_load_time_series(Feeder_size, 1)
M = P.shape[1]  # the number along the 2nd dimension is the number of episodes we could have
T = P.shape[0]  # T is the horizon
converge_all_state = np.array([])  # an array of binary variables. converge_all_state[ii] = 1 if at the state ii, all
# spanning trees have converged power flow; = 0 if at least one spanning tree does not converge
for episode in range(0, M):
    # start an episode
    for t in range(0, T):
        # start a time stamp
        iteration += 1
        print(iteration)
        state = np.array([[P[t, episode, 0]]])
        for ii in range(1, P.shape[2]):
            state = np.append(state, [[P[t, episode, ii]]], axis=0)
        for ii in range(Q.shape[2]):
            state = np.append(state, [[Q[t, episode, ii]]], axis=0)
        converge = 1
        loss_allconfig = []
        volt_allconfig = np.zeros((1, num_outputs, num_node))

        for ii in range(num_outputs):
            Pij, Qij, Vi2, Lij, loss, iter, convergence_flag = fn.load_flow(state, ii, Feeder_size)
            converge = converge * convergence_flag
            loss_allconfig.append(loss)
            Vi = np.sqrt(Vi2)
            for jj in range(num_node):
                if np.isnan(Vi[jj]):
                    print('NaN in main: single time single configuration')
                else:
                    volt_allconfig[0, ii, jj] = Vi[jj]
        #if converge == 0:
            #print("the load flow are non-convergent for at least one configuration. Program terminates")
            #sys.exit()
        Loss_allstate = np.vstack((Loss_allstate, np.asarray(loss_allconfig).T))
        Volt_allstate = np.concatenate((Volt_allstate, volt_allconfig), axis=0)
        if np.isnan(Volt_allstate).any():
            print('NaN in main: all time all configuration')
        # convert a list to a new list of same size but with ranking of each element
        # ref: https://codereview.stackexchange.com/questions/65031/creating-a-list-containing-the-rank-of-the-elements-in-the-original-list
        indices = list(range(len(loss_allconfig)))
        indices.sort(key=lambda x: loss_allconfig[x])
        output = [0] * len(indices)
        for i, x in enumerate(indices):
            output[x] = i
        Rank_allstate = np.vstack((Rank_allstate, np.asarray(output).T))
        converge_all_state = np.append(converge_all_state, converge)

print("do all the power flows converge? 1-yes 0-no")
print(np.prod(converge_all_state))

if Feeder_size == 16:
    np.savetxt("loss.txt", Loss_allstate)
    np.savetxt("rank.txt", Rank_allstate)
    np.save("volt",  Volt_allstate)
elif Feeder_size == 6:
    np.savetxt("loss_6bus.txt", Loss_allstate)
    np.savetxt("rank_6bus.txt", Rank_allstate)
elif Feeder_size == 9:
    np.savetxt("loss_9bus.txt", Loss_allstate)
    np.savetxt("rank_9bus.txt", Rank_allstate)
elif Feeder_size == 11:
    np.savetxt("loss_11bus.txt", Loss_allstate)
    np.savetxt("rank_11bus.txt", Rank_allstate)