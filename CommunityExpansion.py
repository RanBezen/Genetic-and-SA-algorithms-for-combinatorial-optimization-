import argparse
import glob
from utils_comb import bruteForce_CommExpand, greedy_backward_CommExpand, greedy_forward_CommExpand, jocker_CommExpand,Genetic_Algorithms_CommExpand,bruteForce_graphCond
import os
from openpyxl import Workbook, load_workbook

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='',
                        help='What operation? Can choose name or parse')
    parser.add_argument('--infile', type=str, default='inputComm.txt',
                        help='insert path')
    parser.add_argument('--algorithm', type=str, default='',
                        help='algorithm name')
    parser.add_argument('--timeout', type=float, default=60,
                        help='algorithm name')
    args = parser.parse_args()
    mode(args)

def mode(args):
    if args.mode == 'name':
        print('Ran')
    elif args.mode == 'parse':
        parse(args.infile)
    elif args.mode == 'solve':
        if args.algorithm == 'brute_force':
            brute_force(args.infile)
        if args.algorithm == 'greedy':
            greedy(args.infile)
        if args.algorithm == 'jocker':
            jocker(args.infile)
    elif args.mode == 'compete':
        compete(args.infile, args.timeout)


def parse(path):
    edges_list, vertices_A, phi = read_txt(path)
    vertices_list_flat = [val for sublist in edges_list for val in sublist]
    vertices_list = set(vertices_list_flat)
    edges_A = []
    for edge in edges_list:
        # check if list1 contains all elements in list2
        result = all(elem in set(vertices_A) for elem in set(edge))
        # if set(vertices_A) >= set(edge):
        if result:
            edges_A.append(edge)
    print(len(vertices_list))
    print(len(edges_list))
    print(len(vertices_A))
    print(len(edges_A))
    print(phi)

def read_txt(path):
    """
    read txt file to edges_list, vertices_A and phi
    :param path:
    :return: edges_list, vertices_A, phi
    """
    file = open(path, 'r')
    rows = file.read().splitlines()
    edges_list = []
    flag_A = False
    flag_phi = False
    vertices_A = []
    for row in rows:
        edge = row.split(' ')
        if edge[0] == 'A':
            flag_A = True
            edge = edge[1:]
        if edge[0] == 'phi':
            flag_phi = True
        if not flag_A and not flag_phi:
            temp = []
            for ver in edge:
                temp.append(int(ver))
            edges_list.append(temp)
        if flag_A and not flag_phi:
            for ver in edge:
                if ver!='':
                    vertices_A.append(int(ver))
        if flag_phi:
            phi = float(edge[1])
    return edges_list, vertices_A, phi

def brute_force(path):
    edges, vertices_A, phi = read_txt(path)
    maxA_vertices, maxA_phi, edge_to_cut = bruteForce_CommExpand(edges, vertices_A,phi)
    maxA_vertices = set(maxA_vertices)
    str_A = ' '.join(str(e) for e in maxA_vertices)
    print(str_A, '|',len(maxA_vertices), '|',maxA_phi)
    #print('brute force:',str_A, '|',len(maxA_vertices))
    return maxA_vertices


def brute_force_under(path,under):
    edges, vertices_A, phi = read_txt(path)
    vertices = [val for sublist in edges for val in sublist]  # get all vertices
    vertices = set(vertices)
    if len(vertices) < under:
        maxA_vertices, maxA_phi, edge_to_cut = bruteForce_CommExpand(edges, vertices_A,phi)
        maxA_vertices = set(maxA_vertices)
        str_A = ' '.join(str(e) for e in maxA_vertices)
        print(str_A, '|',len(maxA_vertices), '|',maxA_phi)
        #print('brute force:',str_A, '|',len(maxA_vertices))
    else:
        maxA_vertices = [0,0,0,0]
    return maxA_vertices


def greedy(path):
    edges, vertices_A, phi = read_txt(path)
    maxA_vertices, maxA_phi, edge_to_cut = greedy_backward_CommExpand(edges, vertices_A,phi)

    str_A = ' '.join(str(e) for e in maxA_vertices)
    #print('greedy backward:',str_A, '|',len(maxA_vertices))
    #maxA_vertices, maxA_phi, edge_to_cut = greedy_forward_CommExpand(edges, vertices_A, phi)
    #str_A = ' '.join(str(e) for e in maxA_vertices)
    #print('greedy forward:',str_A, '|',len(maxA_vertices))

    print(str_A)

def jocker(path):
    edges, vertices_A, phi = read_txt(path)
    maxA_vertices, maxA_phi, edge_to_cut = jocker_CommExpand(edges, vertices_A, phi)
    str_A = ' '.join(str(e) for e in maxA_vertices)
    print(str_A)

    # print('jocker with BF(jocker):', str_A, '|',len(maxA_vertices))
    #
    # maxA_vertices, maxA_phi, edge_to_cut = jocker_CommExpand(edges, vertices_A, phi, BF_conductance_method='greedy')
    # str_A = ' '.join(str(e) for e in maxA_vertices)
    # print('jocker with BF(greedy):', str_A, '|', len(maxA_vertices))

def compete(path,timeout):
    edges, vertices_A, phi = read_txt(path)
    maxA_vertices, maxA_phi = Genetic_Algorithms_CommExpand(edges, vertices_A, phi,timeout)
    maxA_vertices = set(maxA_vertices)
    str_A = ' '.join(str(e) for e in maxA_vertices)
    print(str_A, '|',len(maxA_vertices), '|',maxA_phi)
    return maxA_vertices, edges, phi


def run_all(path):
    #greedy(path); print('greedy')
    #jocker(path); print('joker')
    brute_force(path); print('brute_force')
    compete(path, timeout=10); print('Genetic_Algorithms')


if __name__ == '__main__':
    main()

    #run_all('nimrod.txt')

    timeouts=[5,10,20,30,50]
    for time in timeouts:
        for infile in glob.glob1('D:/PycharmProjects/optComb/test', 'inputComm*.txt'):
        #for infile in glob.glob1('C:/Users/ran/PycharmProjects/optComb', 'inputComm_6*.txt'):
            path = 'test/' + infile
            print(infile, '#####################')

            #run_all(infile)
            #A = brute_force(infile)
            #A = brute_force_under(infile,20)

            A, edges,phi_original = compete(path, timeout=time)
            phi_pred_A, _ = bruteForce_graphCond(edges,A)

            str_A = ' '.join(str(e) for e in A)
            temp_data = [infile.split('.')[0],str_A, len(A),phi_pred_A,phi_original,time]
            print(temp_data)
            #filepath = 'comm_bt_results.xlsx'
            filepath = 'comm_compete_results.xlsx'
            if not os.path.exists(filepath):
                wb = Workbook()
                sheet = wb.active
                sheet.append(['name', 'A', 'A num','phi_pred_A','phi_original','time'])
                wb.save(filepath)
            wb = load_workbook(filepath)
            sheet = wb.active
            sheet.append(temp_data)
            wb.save(filepath)


    # run_all('try.txt')
    # print('#####################')
    # run_all('try2.txt')
    # print('#####################')
    # run_all('try3.txt')
    # # print('#####################')
    # # run_all('try4.txt') # not good!!!!!
    # print('#####################')
    # run_all('try5.txt')
    # # print('#####################')
    # # run_all('try6.txt')

    # print('#####################')
    # greedy('try.txt')
    # #brute_force('try.txt')
    # print('#####################')
    # greedy('try2.txt')
    # #brute_force('try2.txt')
    # print('#####################')
    # greedy('try3.txt')
    # #brute_force('try3.txt')
    # print('#####################')
    # greedy('try4.txt')
    # #brute_force('try4.txt')
    # print('#####################')
    # greedy('try5.txt')
    # #brute_force('try5.txt')
    # print('#####################')
    # greedy('try6.txt')
    # #brute_force('try6.txt')
    # print('#####################')