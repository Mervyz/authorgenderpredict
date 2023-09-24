

import csv
import numpy
import time
import selector as slctr
from sklearn.model_selection import train_test_split
import pandas as pd
import fitnessFUNs

def main():
    adres="..//NLP_ResultFile//FemalesMales"
    yenidosyaadresi="n2_Char300"
    farkliAlgoritma(0,adres,yenidosyaadresi)
    farkliAlgoritma(1,adres,yenidosyaadresi)
   

def farkliAlgoritma(tip,datasetsadres,yenidosyaadieki):

    # Select optimizers
    PSO = True
    MVO = False
    GWO = True
    MFO = True
    WOA = False
    FFA = True
    BAT = True

    optimizer = [PSO, MVO, GWO, MFO, WOA, FFA, BAT]
    datasets = [datasetsadres]
    # benchmarkfunc=[Fs1,Fs2,Fs3,Fs4,Fs5,Fs6,Fs7,Fs8,Fs9,Fs10]

    # Select number of repetitions for each experiment.
    # To obtain meaningful statistical results, usually 30 independent runs
    # are executed for each algorithm.
    NumOfRuns = 1

    # Select general parameters for all optimizers (population size, number of iterations)
    PopulationSize = 5
    Iterations = 3

    # Export results ?
    Export = True

    # ExportToFile="YourResultsAreHere.csv"
    # Automaticly generated file name by date and time
    ExportToFile = "..//Op_ResultFile//experiment"+yenidosyaadieki + time.strftime("%Y-%m-%d-%H-%M-%S") + ".csv"

    # Check if it works at least once
    Flag = False

    # CSV Header for for the cinvergence
    CnvgHeader1 = []
    CnvgHeader2 = []

    for l in range(0, Iterations):
        CnvgHeader1.append("Iter" + str(l + 1))

    for l in range(0, Iterations):
        CnvgHeader2.append("Iter" + str(l + 1))

    for j in range(0, len(datasets)):  # specfiy the number of the datasets
        for i in range(0, len(optimizer)):

            if optimizer[i]:  # start experiment if an optimizer and an objective function is selected
                for k in range(0, NumOfRuns):

                    # func_details=["costNN",-1,1]
                    func_details = fitnessFUNs.getFunctionDetails(tip)
                    completeData = datasets[j] + ".csv"
                    x,reducedfeatures = slctr.selector(i, func_details, PopulationSize, Iterations, completeData)

                    if Export:
                        with open(ExportToFile, 'a', newline='\n') as out:
                            writer = csv.writer(out, delimiter=',')

                            if not Flag:  # just one time to write the header of the CSV file
                                header = numpy.concatenate([["Optimizer", "Dataset", "objfname", "Experiment", "startTime",
                                                             "EndTime", "ExecutionTime", "trainAcc", "testAcc"],
                                                            CnvgHeader1, CnvgHeader1])
                                writer.writerow(header)
                            a = numpy.concatenate([[x.optimizer, datasets[j], x.objfname, k + 1, x.startTime, x.endTime,
                                                    x.executionTime, x.trainAcc, x.testAcc], x.convergence1,
                                                   x.convergence2])
                            writer.writerow(a)
                            writer.writerow(reducedfeatures)

                            print("------------ALGORİTMA SONU------------")
                        out.close()
                    Flag = True  # at least one experiment

    if not Flag:  # Faild to run at least one experiment
        print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions")



if __name__ == "__main__":
    main()