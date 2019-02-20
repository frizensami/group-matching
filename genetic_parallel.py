#! /usr/bin/env python3
'''
Runs multiple instances of genetic.py in parallel (SPMD) with different parameters
'''
import sys
if sys.version_info < (3, 0):
    sys.stdout.write("Sorry, requires Python 3.x, not Python 2.x\n")
    sys.exit(1)
import subprocess
import multiprocessing
import os
from time import strftime, localtime

CMD = './genetic.py --numparticipants {} --groupsize {} --populationsize {} --generations {} --numelitism {} --numrest {} --positiveweight {} --negativeweight {} --mutationchance {} --mutationswaps {} --numhalloffame {} --graphdir {}'

TEST_FOLDER = 'parallel_runs/'


def run_genetic(logfilename, numparticipants, groupsize, populationsize, generations, numelitism, numrest, positiveweight, negativeweight, mutationchance, mutationswaps, numhalloffame, graphdir, debug=False, notest=False, graphhide=False):
    formatted_cmd = CMD.format(numparticipants, groupsize, populationsize, generations, numelitism, numrest, positiveweight, negativeweight, mutationchance, mutationswaps, numhalloffame, graphdir)
    if debug:
        formatted_cmd += " --debug"
    if notest:
        formatted_cmd += " --notest"
    if graphhide:
        formatted_cmd += " --graphhide"
    proc = subprocess.Popen(formatted_cmd, shell=True, stdout=open(logfilename, 'w'), universal_newlines=True)
    return proc


if __name__ == "__main__":
    cpus = multiprocessing.cpu_count()
    print("Detected {} CPUs".format(cpus))
    time_for_dir = strftime("%Y-%m-%d--%H-%M-%S/", localtime()) 
    test_dir = TEST_FOLDER + time_for_dir
    os.makedirs(test_dir)
    print("Created directory {} for this test".format(test_dir))

    procs = []
    gens_start = 10
    gens_fx = lambda prevgen: prevgen + 5
    cpus = 8
    for cpu in range(cpus):
        logfilename = test_dir + str(cpu) + ".txt"
        proc = run_genetic(logfilename=logfilename, numparticipants=30, groupsize=3, populationsize=1000, generations=gens_start, numelitism=250, numrest=250, positiveweight=100, negativeweight=-1000, mutationchance=0.1, mutationswaps=1, numhalloffame=5, graphdir=test_dir + 'graphs/')
        gens_start = gens_fx(gens_start)
        print("Process #{} started".format(cpu))
    print("All processes started: exiting")

