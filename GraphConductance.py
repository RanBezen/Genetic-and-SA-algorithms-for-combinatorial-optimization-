import argparse
from utils_comb import bruteForce_graphCond, greedy_graphCond, jocker_graphCond, simulated_annealing_graphCond, lower_bound_cheeger
import glob
import os
from openpyxl import Workbook, load_workbook

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='',
                        help='What operation? Can choose name or parse')
    parser.add_argument('--infile', type=str, default='inputGraph.txt',
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

    edges_list = read_txt(path)
    vertices_list_flat = [val for sublist in edges_list for val in sublist]
    vertices_list = set(vertices_list_flat)
    print(len(vertices_list))
    print(len(edges_list))

def read_txt(path):
    file = open(path, 'r')
    rows = file.read().splitlines()
    edges_list = []
    for row in rows:
        edge = row.split(' ')
        temp = []
        for ver in edge:
            temp.append(int(ver))
        edges_list.append(temp)
    return edges_list

def brute_force(path):
    edges = read_txt(path)
    phi,edge_cut = bruteForce_graphCond(edges)
    print(str(phi))

    #print('brute force:',phi)


def greedy(path):
    edges = read_txt(path)
    phi,edge_cut = greedy_graphCond(edges)
    #print('greedy:',phi)
    print(str(phi))

def jocker(path):
    edges = read_txt(path)
    phi, edge_cut = jocker_graphCond(edges)
    #print('jocker:',phi)
    print(str(phi))

def compete(path,timeout):
    edges = read_txt(path)
    phi, cutted_edges_min = simulated_annealing_graphCond(edges, timeout)
    #print('compete:',phi)
    print(str(phi))
    return phi

def run_all(path):
    # greedy(path)
    # jocker(path)
    brute_force(path)
    #lower_bound(path)
    compete(path,5)

def lower_bound(path):
    edges = read_txt(path)
    lower,upper = lower_bound_cheeger(edges)
    print(str(lower),str(upper))

if __name__ == '__main__':
    main()


    timeouts = [0.5, 1, 5, 10, 30]
    for infile in glob.glob1('D:/PycharmProjects/optComb/test', 'inputGraph*.txt'):
        path = 'test/' + infile
        print(infile, '#####################')
        temp_data = []
        temp_data.append(infile.split('.')[0])
        for time in timeouts:

            phi = compete(path, timeout=time)

            #temp_data = [infile.split('.')[0], phi, time]
            temp_data.append(phi)
            print(temp_data)

        filepath = 'cond_compete_results.xlsx'
        if not os.path.exists(filepath):
            wb = Workbook()
            sheet = wb.active
            sheet.append(['name', '0.5', '1', '5', '10', '30'])
            wb.save(filepath)
        wb = load_workbook(filepath)
        sheet = wb.active
        sheet.append(temp_data)
        wb.save(filepath)

    # print('#####################')
    # upper_bound('test.txt')
    # print('#####################')
    # upper_bound('test2.txt')
    # print('#####################')
    # upper_bound('test3.txt')
    # print('#####################')
    # upper_bound('test4.txt')
    # print('#####################')
    # upper_bound('test5.txt')
    # print('#####################')
    # upper_bound('test6.txt')


    # print('#####################')
    # run_all('test.txt')
    # print('#####################')
    # run_all('test2.txt')
    # print('#####################')
    # run_all('test3.txt')
    # print('#####################')
    # run_all('test3.txt')
    # print('#####################')
    # run_all('test4.txt')
    # print('#####################')
    # run_all('test5.txt')
    # print('#####################')
    # run_all('test6.txt')
    # print('#####################')





    # parse('inputGraph.txt')
    # parse('inputGraph2.txt')
    # parse('inputGraph3.txt')
