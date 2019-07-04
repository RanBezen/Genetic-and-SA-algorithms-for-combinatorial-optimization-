from itertools import combinations
import copy
import random
import time
import math
import numpy as np

def collect_edges_A(edges,verticesA):
    edges_A = []
    for w in range(len(edges)):  # collect A edges
        edge = edges[w]
        GroupFlag = all(elem in set(verticesA) for elem in set(edge))  # check if the edge belong to group A
        if GroupFlag:
            edges_A.append(edge)
    return edges_A

def get_phi(verticesA, verticesB, edges):
    """
    this function find Phi by verticesA, verticesB, edges
    :param verticesA:
    :param verticesB:
    :param edges:
    :return:  phi, edges_to_cut
    """
    edges_to_cut = []
    volA = 0
    volB = 0
    cut = 0
    for w in range(len(edges)):
        edge = edges[w]
        CutFlag = any(elem in set(verticesA) for elem in set(edge))
        GroupAflag = all(elem in set(verticesA) for elem in set(edge))
        GroupBflag = all(elem in set(verticesB) for elem in set(edge))

        if CutFlag and not GroupAflag and not GroupBflag:   # check if the edge is in cut
            cut += 1
            edges_to_cut.append(edge)
        elif GroupAflag:    # check if the edge belong to A
            volA += 2
        elif GroupBflag:    # check if the edge belong to B
            volB += 2
    volA += cut
    volB += cut
    if min(volA, volB) > 0:
        phi = cut / min(volA, volB)
    else:
        phi = 0
    return phi, edges_to_cut

def touched_ver(edges,vertices_A_original):
    vertices = [val for sublist in edges for val in sublist]  # get all vertices
    vertices = set(vertices)
    verticesB = [item for item in vertices if item not in vertices_A_original]  # get vertices of A_ B
    touch_ver = []
    for w in range(len(edges)):
        edge = edges[w]
        CutFlag = any(elem in set(vertices_A_original) for elem in set(edge))
        GroupAflag = all(elem in set(vertices_A_original) for elem in set(edge))
        GroupBflag = all(elem in set(verticesB) for elem in set(edge))
        if CutFlag and not GroupAflag and not GroupBflag:  # check if the edge is in cut
            touch_ver.append(edge)
    touch_ver = set([val for sublist in touch_ver for val in sublist])
    touch_ver = [item for item in touch_ver if item not in vertices_A_original] #get vertices of A_ B
    return touch_ver

def lower_bound_cheeger(edges,vertices=[]):

    if not vertices:
        vertices = [val for sublist in edges for val in sublist]  # get all vertices
    vertices = list(set(vertices))

    dist_mat = np.zeros((len(vertices),len(vertices)))
    norm_dist_mat = np.zeros((len(vertices),len(vertices)))
    encode_ver = []
    for k in range(len(vertices)):
        encode_ver.append(k)
    for edge in edges:
        ver1 = edge[0]
        ver2 = edge[1]
        ver1 = encode_ver[vertices.index(ver1)]
        ver2 = encode_ver[vertices.index(ver2)]
        dist_mat[ver1, ver2] = 1
        dist_mat[ver2, ver1] = 1
    for i in range(0,len(vertices)):
        row = dist_mat[i][:]
        sum_row = np.sum(row)
        norm_dist_mat[i][:] = row/sum_row
    eigen_values,eigen_vectors = np.linalg.eig(norm_dist_mat)
    eigen_values = np.sort(eigen_values)
    eigen_values = eigen_values.tolist()

    if eigen_values:
        eigen_values.pop()
        eigen_values = np.asarray(eigen_values)
        lambda_2 = eigen_values.max()
        #lambda_2 = eigen_values[len(eigen_values)-2]
        lower_bound = (1-lambda_2)/2
        if lambda_2 < 1:
            upper_bound = (2 - 2 * lambda_2) ** 0.5
        else:
            upper_bound = 1
    else:
        lower_bound = 0
        upper_bound = 1



    return lower_bound, upper_bound

def bruteForce_graphCond(edges, vertices=[]):
    """
    This function is brute force algorithm for calculation graph conductance
    :param edges:
    :param vertices: optional
    :return:
    """
    cutted_edges = []
    first = True
    if not vertices:
        vertices = [val for sublist in edges for val in sublist]
        vertices = set(vertices)
    run_to = round(len(vertices) / 2) + 1   #run on the half of the vertics number
    for num in range(1, run_to):
        combs = list(combinations(vertices, num))
        for k in range(len(combs)): #check all combinations
            verticesA = list(combs[k])
            verticesB = [item for item in vertices if item not in verticesA]
            phi_temp, edges_to_cut = get_phi(verticesA,verticesB,edges)
            if first:   #init phi
                phi = phi_temp
                first = False
            if phi_temp<phi:    #update phi
                phi = phi_temp
                cutted_edges = edges_to_cut

    return phi, cutted_edges



def bruteForce_CommExpand(edges, vertices_A_original, phi):
    """
    this function make brute force algorithm for community expantion
    :param edges: egdes list input
    :param vertices_A_original: A ver sublist
    :param phi: lower bound phi
    :return: maxA_vertices, maxA_phi, maxA_edges_to_cut
    """
    phi_temp = 0
    phi_out = phi
    edges_to_cut = []
    maxA_vertices = vertices_A_original
    flagMax=False
    vertices = [val for sublist in edges for val in sublist] #get all vertices
    vertices = set(vertices)
    verticesB = [item for item in vertices if item not in vertices_A_original] #get vertices of A_ B
    run_to = len(verticesB)+1
    for num in range(1, run_to): # run on all posible combinations
        comb_vars_number = run_to-num # begin from the largest group
        combs = list(combinations(verticesB,comb_vars_number))  #create combinations from B
        for k in range(len(combs)): #run on all comb_vars_number combinations
            addon=list(combs[k])
            verticesA = vertices_A_original + addon # add to A group
            edges_A = []
            # get the edges of group A
            edges_A = collect_edges_A(edges, verticesA)

            # calc the Conduntace of group A

            phi_temp, edges_to_cut = bruteForce_graphCond(edges_A,verticesA) #check Phi with brute force graphCond

            if phi_temp != 888 and phi_temp>=phi:
                phi_out = phi_temp
                maxA_vertices = verticesA
                flagMax = True
                break    #   if the phi over the trashold break the loop and continue
        if flagMax:
            #phi_temp = phi
            break    #   if the phi over the trashold with comb_vars_number, break.

    return maxA_vertices, phi_out, edges_to_cut

def greedy_graphCond(edges, vertices=[], random_ver=False):
    """
    This function is brute force algorithm for calculation graph conductance
    :param edges:
    :param vertices: optional
    :return:
    """
    cutted_edges = []
    A_final = []
    first = True
    verticesA = []
    #verticesA_min = []
    if not vertices:
        vertices = [val for sublist in edges for val in sublist]
        vertices = set(vertices)
    run_to = round(len(vertices) / 2) + 1   #run on the half of the vertics number
        ###########
    for num in range(1, run_to):
        phi_interval=1
        vertices = [item for item in vertices if item not in verticesA]
        verticesA_test = copy.deepcopy(verticesA)
        if random_ver:
            random.shuffle(vertices)
        for ver in vertices:  # check all combinations
            verticesA_test.append(ver)
            verticesB = [item for item in vertices if item not in verticesA_test]
            phi_temp, edges_to_cut = get_phi(verticesA_test, verticesB, edges)
            if first:  # init phi
                verticesA = copy.deepcopy(verticesA_test)
                #verticesA_min = copy.deepcopy(verticesA_test)
                phi_min = phi_temp
                phi_interval = phi_temp
                first = False

            if phi_temp < phi_interval:
                verticesA = copy.deepcopy(verticesA_test)
                phi_interval = phi_temp

            if phi_temp < phi_min:  # update phi
                #verticesA_min = copy.deepcopy(verticesA_test)
                phi_min = phi_temp
                cutted_edges = edges_to_cut
                A_final = copy.deepcopy(verticesA)
            verticesA_test.remove(ver)
    return phi_min,cutted_edges,A_final

def greedy_backward_CommExpand(edges, vertices_A_original, phi, conductance_method='greedy'):
    """
    this function make brute force algorithm for community expantion
    :param edges: egdes list input
    :param vertices_A_original: A ver sublist
    :param phi: lower bound phi
    :return: maxA_vertices, maxA_phi, maxA_edges_to_cut
    """
    phi_temp = 0
    edges_to_cut = []
    maxA_vertices = vertices_A_original
    flagMax=False
    vertices = [val for sublist in edges for val in sublist] #get all vertices
    vertices = set(vertices)
    verticesB_org = [item for item in vertices if item not in vertices_A_original] #get vertices of A_ B
    run_to = len(verticesB_org)+1
    verticesA = copy.deepcopy(vertices_A_original)
    for num in range(1, run_to): # run on all posible combinations
        phi_interval = 0
        verticesB = [item for item in vertices if item not in verticesA]
        verticesA_test = copy.deepcopy(verticesA)
        for ver in verticesB: #run on all comb_vars_number combinations
            ver = [ver]
            addon=[item for item in verticesB if item not in ver]
            verticesA_test = verticesA_test + addon # add to A group
            edges_A = []
            # get the edges of group A
            edges_A = collect_edges_A(edges, verticesA_test)

            # calc the Conduntace of group A
            if conductance_method == 'greedy':
                phi_temp, edges_to_cut,_ = greedy_graphCond(edges_A,verticesA_test) #check Phi with greedy graphCond
            if conductance_method == 'jocker':
                phi_temp, edges_to_cut = jocker_graphCond(edges_A,verticesA_test) #check Phi with jocker graphCond
            if phi_temp >= phi_interval:
                phi_interval = phi_temp
                verticesA = copy.deepcopy(verticesA_test)
            if phi_temp>=phi:
                verticesA = copy.deepcopy(verticesA_test)
                maxA_vertices = copy.deepcopy(verticesA_test)
                flagMax = True
                break    #   if the phi over the trashold break the loop and continue
            verticesA_test = [item for item in verticesA_test if item not in addon]
        if flagMax:
            break    #   if the phi over the trashold with comb_vars_number, break.

    return maxA_vertices, phi_temp, edges_to_cut



def greedy_forward_CommExpand(edges, vertices_A_original, phi,conductance_method='greedy'):
    """
    this function make brute force algorithm for community expantion
    :param edges: egdes list input
    :param vertices_A_original: A ver sublist
    :param phi: lower bound phi
    :return: maxA_vertices, maxA_phi, maxA_edges_to_cut
    """
    phi_temp = 0
    edges_to_cut = []
    maxA_vertices = vertices_A_original
    flagMax=False
    vertices = [val for sublist in edges for val in sublist] #get all vertices
    vertices = set(vertices)
    verticesB_org = [item for item in vertices if item not in vertices_A_original] #get vertices of A_ B
    run_to = len(verticesB_org)+1
    verticesA = copy.deepcopy(vertices_A_original)
    for num in range(1, run_to): # run on all posible combinations
        phi_interval = 0
        verticesB = [item for item in vertices if item not in verticesA]
        verticesA_test = copy.deepcopy(verticesA)
        for ver in verticesB: #run on all comb_vars_number combinations
            ver = [ver]
            #addon=[item for item in verticesB if item not in ver]
            verticesA_test = verticesA_test + ver # add to A group
            edges_A = []
            # get the edges of group A
            edges_A = collect_edges_A(edges, verticesA_test)

            if conductance_method == 'greedy':
                phi_temp, edges_to_cut,_ = greedy_graphCond(edges_A,verticesA_test) #check Phi with greedy graphCond
            if conductance_method == 'jocker':
                phi_temp, edges_to_cut = jocker_graphCond(edges_A,verticesA_test) #check Phi with jocker graphCond            if phi_temp >= phi_interval:
                phi_interval = phi_temp
                verticesA = copy.deepcopy(verticesA_test)
            if phi_temp>=phi:
                verticesA = copy.deepcopy(verticesA_test)
                maxA_vertices = copy.deepcopy(verticesA_test)
                flagMax = True
                break    #   if the phi over the trashold break the loop and continue
            verticesA_test = [item for item in verticesA_test if item not in ver]
        if flagMax:
            break    #   if the phi over the trashold with comb_vars_number, break.

    return maxA_vertices, phi_temp, edges_to_cut


def jocker_CommExpand(edges, vertices_A_original, phi, BF_conductance_method='jocker'):
    phi = min(1,phi*(1.05))
    forward_A_vertices, forward_phi_temp, forward_edges_to_cut = greedy_forward_CommExpand(edges, vertices_A_original, phi)
    backward_A_vertices, backward_phi_temp, backward_edges_to_cut = greedy_backward_CommExpand(edges, vertices_A_original, phi)
    togheder = forward_A_vertices + backward_A_vertices
    togheder = set(togheder)
    togheder = list(togheder)

    max_A = backward_A_vertices
    phi_max = backward_phi_temp
    edges_to_cut = backward_edges_to_cut

    edges_A = collect_edges_A(edges,togheder)
    togheder_phi, togheder_edges_to_cut,_ = jocker_graphCond(edges_A, togheder)

    if togheder_phi >= phi:
        max_A = togheder
        phi_max = togheder_phi
        edges_to_cut = togheder_edges_to_cut
        max_A, phi_max, edges_to_cut = greedy_forward_CommExpand(edges, max_A, phi)

    elif len(forward_A_vertices) > len(backward_A_vertices):
        max_A = forward_A_vertices
        phi_max = forward_phi_temp
        edges_to_cut = forward_edges_to_cut
    #max_A, phi_max, edges_to_cut = bruteForce_CommExpand(edges, max_A, phi, conductance_method=BF_conductance_method) #choose 'jocker' or 'greedy'
    return max_A, phi_max, edges_to_cut

def jocker_graphCond(edges, vertices=[]):
    phi=1.00001
    cutted_edges_min = []
    for i in range(len(edges)):
        phi_temp, cutted_edges,A_final = greedy_graphCond(edges,vertices,random_ver=True)
        if phi_temp < phi:
            phi=phi_temp
            cutted_edges_min = cutted_edges
    return phi, cutted_edges_min,A_final



def vertices_for_choosing_Gen(edges, vertices_A_original):

    touch_ver = touched_ver(edges, vertices_A_original)
    if touch_ver:
        ver_to_choose = touch_ver
    else:
        vertices = [val for sublist in edges for val in sublist]  # get all vertices
        vertices = set(vertices)
        ver_to_choose = [item for item in vertices if item not in vertices_A_original]  # get vertices of A_ B
    return ver_to_choose

def Gen_smart_init_population(edges,vertices_A_original, number_of_Gens):

    A_vertices_lists = [None] * number_of_Gens
    edges_A_lists = [None] * number_of_Gens
    ver_to_choose = vertices_for_choosing_Gen(edges, vertices_A_original)
    num_of_ver=len(ver_to_choose)
    for i in range(number_of_Gens):
        vertices_numer_to_add = random.sample(range(1, num_of_ver+1), 1)[0]

        add = random.sample(ver_to_choose, vertices_numer_to_add)
        verticesA = copy.deepcopy(vertices_A_original)
        verticesA = verticesA + add  # add to A group
        # get the edges of group A
        edges_A = collect_edges_A(edges, verticesA)

        A_vertices_lists[i]=verticesA
        edges_A_lists[i]=edges_A
    return A_vertices_lists, edges_A_lists

def Gen_eval(A_vertices_lists,edges_A_lists,edges,phi, vertices_A_original, timeout):
    start = time.time()


    vertices = [val for sublist in edges for val in sublist]  # get all vertices
    vertices = set(vertices)
    n=len(vertices)
    A_phi_lists=[]
    A_len_lists=[]
    scores=[]
    bestA_len = len(vertices_A_original)
    best_A = vertices_A_original
    for i in range(len(A_vertices_lists)):
        l=len(A_vertices_lists[i])
        A_len_lists.append(l)
        #phi_temp,_= jocker_graphCond(edges=edges_A_lists[i], vertices=A_vertices_lists[i])
        #print(edges_A_lists[i])
        lower_bound, upper_bound = lower_bound_cheeger(edges=edges_A_lists[i])
        if lower_bound < phi and upper_bound >= phi:
            #phi_temp, _ = bruteForce_graphCond(edges=edges_A_lists[i], vertices=A_vertices_lists[i])
            #print(timeout/len(A_vertices_lists))

            #s=start = time.time()
            t = timeout/len(A_vertices_lists)
            #t=0.1
            phi_temp, _ = simulated_annealing_graphCond(edges=edges_A_lists[i], timeout=t, vertices=A_vertices_lists[i])
            #print('רצוי', t)
            #print('מצוי', time.time() - s)
        elif lower_bound >= phi:
            phi_temp = lower_bound
        elif upper_bound < phi:
            phi_temp = upper_bound*0.9
        #print(phi_temp, phi)
        A_phi_lists.append(phi_temp)

        if phi_temp < phi/2:
            score = 0
        else:
            lambd = phi_temp / phi
            score = math.exp(l / n - 1) * min((lambd), 1)

        if phi_temp >= phi:
            if bestA_len < l:
                best_A = A_vertices_lists[i]
                bestA_len = l

        scores.append(score)
    #print('eval time',time.time() - start)
    return A_phi_lists,scores,A_len_lists, best_A

def Gen_selection(A_vertices_lists,edges_A_lists,scores,selection_num):

    sortA_vertices = [x for _, x in sorted(zip(scores, A_vertices_lists), reverse=True)]
    sortA_edges = [x for _, x in sorted(zip(scores, edges_A_lists), reverse=True)]
    sortA_scores = [x for _, x in sorted(zip(scores, scores), reverse=True)]

    if 0 in scores:
        zero_indx = sortA_scores.index(0) + 1
        if zero_indx < selection_num:

            selection_num = zero_indx
    new_A_selection = sortA_vertices[:selection_num]
    new_A_edges = sortA_edges[:selection_num]

    return new_A_selection, new_A_edges

def Gen_crossover(vertices_A_original,A_vertices_lists, edges,edges_A_lists,number_of_Gens,best_A):
    A_vertices_lists.append(best_A)
    selection_num = len(A_vertices_lists)
    crossovers = number_of_Gens - selection_num
    for i in range(crossovers):
        #intersect
        n = random.sample(range(0, selection_num), 2)
        ver1=A_vertices_lists[n[0]]
        ver2=A_vertices_lists[n[1]]
        ver1_added = [item for item in ver1 if item not in vertices_A_original]  # get vertices of A_ B
        ver2_added = [item for item in ver2 if item not in vertices_A_original]

        new_addon = set(ver1_added + ver2_added)
        new_pop = vertices_A_original + list(new_addon)

        edges_new_pop = collect_edges_A(edges, new_pop)
        A_vertices_lists.append(new_pop)
        edges_A_lists.append(edges_new_pop)

    return A_vertices_lists, edges_A_lists

def Gen_mutation(vertices_A_original,A_vertices_lists, edges):

    edges_A_lists = []
    new_A_vertices_lists = []

    for i in range(len(A_vertices_lists)-1):
        choosen_A = A_vertices_lists[i]

        vertices_added = [item for item in choosen_A if item not in vertices_A_original ]  # get vertices of A_ B

        if vertices_added:
            vertices_to_release = random.sample(vertices_added, 1)[0]

            touch_ver = touched_ver(edges, choosen_A)
            if touch_ver:
                ver_to_choose = touch_ver
                add = random.sample(ver_to_choose, 1)
                choosen_A[choosen_A.index(vertices_to_release)] = add[0]

        new_A_vertices_lists.append(choosen_A)
        edges_A = collect_edges_A(edges, choosen_A)
        edges_A_lists.append(edges_A)
    return new_A_vertices_lists, edges_A_lists

def Gen_ending_selection(A_vertices_population,edges_A_population,edges, phi,best_A):

    A_phi_lists , _, _ ,best_A = Gen_eval(A_vertices_population, edges_A_population, edges, phi,best_A,0.5)
    final_As = []
    final_phis = []
    final_lens = []
    for i in range(len(A_vertices_population)):
        phi_tmp = A_phi_lists[i]
        if phi_tmp >= phi:
           final_As.append(A_vertices_population[i])
           final_phis.append(phi_tmp)
           final_lens.append(len(A_vertices_population[i]))

    if final_lens:
        max_indx = final_lens.index(max(final_lens))
        maxA_phi = final_phis[max_indx]
        maxA_vertices = final_As[max_indx]
    else:
        maxA_phi = phi
        maxA_vertices = best_A

    return maxA_vertices, maxA_phi

def Genetic_Algorithms_CommExpand(edges, vertices_A_original, phi,timeout):
    number_of_Gens = min(len(set([val for sublist in edges for val in sublist])),20)
    selection_num = int(number_of_Gens/2)
    start = time.time()
    time.clock()
    elapsed = 0

    cond_time = (elapsed/10 + 0.1)
    A_vertices_population, edges_A_population = Gen_smart_init_population(edges,vertices_A_original, number_of_Gens = number_of_Gens)
    A_phi_lists,scores,_, best_A = Gen_eval(A_vertices_population,edges_A_population,edges,phi,vertices_A_original,timeout=cond_time)
    timeout -= 1
    while elapsed < timeout:
        elapsed = time.time() - start
        remain_time = timeout - elapsed
        cond_time = min((elapsed / number_of_Gens + 0.1)**0.5,remain_time/10)
        #cond_time =  (elapsed / number_of_Gens + 0.1)**0.5 # if remain_time > elapsed/2 else 0.5
        #print('start',elapsed)
        #print('cond_time',cond_time)
        A_vertices_population, edges_A_population = Gen_selection(A_vertices_population, edges_A_population, scores, selection_num=selection_num)
        #print('selection',time.time() - start)
        #if the selection delete all the populations
        if len(A_vertices_population) == 1:
            A_vertices_population, edges_A_population = Gen_smart_init_population(edges, vertices_A_original, number_of_Gens=selection_num)
            A_phi_lists, scores, _, best_A = Gen_eval(A_vertices_population, edges_A_population, edges, phi,best_A,timeout=cond_time)
            A_vertices_population, edges_A_population = Gen_selection(A_vertices_population, edges_A_population, scores,selection_num=selection_num)
            #print('return to init', len(A_vertices_population))
            if len(A_vertices_population) == 1:
                #print('break')
                break
        A_vertices_population, edges_A_population = Gen_crossover(vertices_A_original, A_vertices_population, edges, edges_A_population, number_of_Gens,best_A)
        #print('Gen_crossover',time.time() - start)

        A_vertices_population, edges_A_population = Gen_mutation(vertices_A_original,A_vertices_population, edges)
        #print('Gen_mutation',time.time() - start)

        A_phi_lists,scores,_,best_A = Gen_eval(A_vertices_population, edges_A_population,edges,phi,best_A,timeout=cond_time)

        elapsed = time.time() - start
        #print('Gen_eval', elapsed)

    #print('end:',elapsed)
    if len(A_vertices_population) > 1:
        A_vertices_population, edges_A_population = Gen_selection(A_vertices_population, edges_A_population, scores, selection_num=selection_num)
        maxA_vertices, maxA_phi = Gen_ending_selection(A_vertices_population, edges_A_population, edges, phi,best_A)
    else:
        maxA_vertices = best_A
        maxA_phi = phi
    print(elapsed)
    return maxA_vertices, maxA_phi

def SA_init(edges):
    #vertice = random.sample(vertices, 1)[0]
    phi,_,A_final = jocker_graphCond(edges)

    return A_final,phi

def SA_neighbour_sol(verticesA,edges):
    verticesA_new = copy.deepcopy(verticesA)

    #only reduce
    #only add
    #rduce and add
    p = random.random()
    if p <= 0.333 and verticesA_new:
        #only reduce
        verticesA_new.remove(random.sample(verticesA_new, 1)[0])

    elif p > 0.333 and p <= 0.666 and verticesA_new:
        #only add
        neighbours = touched_ver(edges, verticesA_new)
        if neighbours:
            neighbour = random.sample(neighbours, 1)[0]
            verticesA_new.append(neighbour)

    elif p > 0.666 and verticesA_new:
        # reduce and add
        verticesA_new.remove(random.sample(verticesA_new, 1)[0])
        neighbours = touched_ver(edges, verticesA_new)
        if neighbours:
            neighbour = random.sample(neighbours, 1)[0]
            verticesA_new.append(neighbour)

    return verticesA_new

def SA_cost(verticesA,verticesA_neighbour,edges):
    vertices = [val for sublist in edges for val in sublist]
    vertices = set(vertices)
    verticesB = [item for item in vertices if item not in verticesA]
    cost_1,_ = get_phi(verticesA,verticesB,edges)


    verticesB = [item for item in vertices if item not in verticesA_neighbour]
    cost_2,_ = get_phi(verticesA_neighbour,verticesB,edges)

    d_cost = cost_2 - cost_1
    return d_cost

def simulated_annealing_graphCond(edges, timeout, vertices=[]):
    start = time.time()
    if not vertices:
        vertices = [val for sublist in edges for val in sublist]
        vertices = set(vertices)
    #initial
    verticesA = []
    init_A,phi_min = SA_init(edges)
    verticesA = init_A
    verticesB = [item for item in vertices if item not in verticesA]
    T = 1 - 0

    TL = round(len(vertices) / 4) + 1  # run on the half of the vertics number
    time.clock()
    elapsed = 0
    k=0
    edges_to_cut_min = []
    zero = False
    elapsed = time.time() - start
    while elapsed < timeout and not zero:
        #elapsed = time.time() - start

        for i in range(1, TL):

            verticesA_neighbour = SA_neighbour_sol(verticesA, edges)

            if len(verticesB)<2 or not verticesA_neighbour:
                init_A,phi_temp = SA_init(edges)
                verticesA = init_A
                verticesB = [item for item in vertices if item not in verticesA]
                T = 1 - 0
                break

            d_cost = SA_cost(verticesA,verticesA_neighbour,edges)

            if d_cost <= 0:
                verticesA = copy.deepcopy(verticesA_neighbour)
            else:
                q = random.random()
                p = math.exp(-d_cost/(T+0.00001))
                if q < p:
                    verticesA = copy.deepcopy(verticesA_neighbour)
            verticesB = [item for item in vertices if item not in verticesA]
            phi_temp, edges_to_cut = get_phi(verticesA,verticesB,edges)
            if phi_min > phi_temp:
                phi_min = phi_temp
                edges_to_cut_min = edges_to_cut
            if phi_min == 0:
                zero = True
                break
            if time.time() - start < timeout:
                break
        #Quadratic multiplicative cooling
        k+=1
        T = T/(1+0.2*k**2)

        elapsed = time.time() - start
    print(elapsed)
    return phi_min, edges_to_cut_min

def stopwatch(seconds):
    start = time.time()
    time.clock()
    elapsed = 0
    print(elapsed)
    while elapsed < seconds:
        elapsed = time.time() - start
        elapsed = int(elapsed)
        #time.sleep(1)
    print(elapsed)

# stopwatch(10)
