import os
import numpy as np
import pandas as pd
from numpy import matlib
import random as rn
from sklearn import preprocessing
from CHOA import CHOA
from DHOA import DHOA
from GSO import GSO
from Glob_Vars import Glob_Vars
from Model_AE_LSTM import Model_AE_LSTM
from Model_AE_LSTM_MLP import Model_AE_LSTM_MLP
from Model_GRU import Model_GRU
from Model_MLP import Model_MLP
from Model_PROPOSED import Model_PROPOSED
from PROPOSED import PROPOSED
from SOA import SOA
from objective_function import Objfun
from Plot_Results import plot_results_learnperc, plot_results_kfold

# Read Dataset
an = 0
if an == 1:
    Data = []
    Data_col = ['guarantees', 'reason', 'other_credits', 'credit_report', 'marital_status', 'employment',
                'qualification', 'immigrant', 'accommodation', 'estate', 'savings', 'phone', 'status']
    Directory = './Dataset/'
    dir1 = os.listdir(Directory)
    file1 = Directory + dir1[2]
    read = pd.read_csv(file1, encoding='latin-1')
    tar = read['bankruptcy']
    tar = np.asarray(tar)
    tar = np.reshape(tar, [-1, 1])
    read.drop('bankruptcy', inplace=True, axis=1)
    for i in range(len(Data_col)):
        uni = np.unique(read[Data_col[i]])
        for j in range(len(uni)):
            ind = np.where(read[Data_col[i]] == uni[j])
            read[Data_col[i]][ind[0]] = j + 1
    np.save('Data.npy', read)
    np.save('Target.npy', tar)

# Data Tranformation
an = 0
if an == 1:
    Datas = []
    Data = np.load('Data.npy', allow_pickle=True)
    for i in range(len(Data)):  # loop to read all dataset
        data = Data[i]
        pd.isnull('data')  # locates missing data
        df = pd.DataFrame(data)
        # Replace with 0 values. Accepts regex.
        df.replace(np.NAN, 0, inplace=True)
        # Replace with zero values
        df.fillna(value=0, inplace=True)
        df.drop_duplicates()  # removes the duplicates
        data = np.array(df)
        # Normalization
        scaler = preprocessing.MinMaxScaler()
        for i in range(data.shape[1]):
            print(i, np.unique(data[:, i]))
        normalized = scaler.fit_transform(data)
        Datas.append(normalized)
    np.save('Data_transformation.npy', Datas)

# Optimization for  Classifier
an = 0
if an == 1:
    Bestsol = []
    Data = np.load('Data_transformation.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Glob_Vars.Data = Data
    Glob_Vars.Target = Target
    Npop = 10
    Chlen = 5
    xmin = matlib.repmat([5, 50, 5, 5, 0.01], Npop, 1)
    xmax = matlib.repmat([255, 100, 255, 255, 0.9], Npop, 1)
    fname = Objfun

    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 25

    print("DHOA...")
    [bestfit1, fitness1, bestsol1, time1] = DHOA(initsol, fname, xmin, xmax, Max_iter)

    print("CHOA...")
    [bestfit2, fitness2, bestsol2, time2] = CHOA(initsol, fname, xmin, xmax, Max_iter)

    print("GSO...")
    [bestfit4, fitness4, bestsol4, time3] = GSO(initsol, fname, xmin, xmax, Max_iter)

    print("SOA...")
    [bestfit3, fitness3, bestsol3, time4] = SOA(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    Bestsol = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])

    np.save('Bestsol.npy', Bestsol)

## Classification
an = 0
if an == 1:
    Eval_all = []
    Learnper = [0.35, 0.55, 0.65, 0.75, 0.85]
    Data = np.load('Data_transformation.npy', allow_pickle=True)
    sol = np.load('Bestsol.npy', allow_pickle=True)
    Targets = np.load('Target.npy', allow_pickle=True)
    for i in range(len(Learnper)):
        Eval = np.zeros((10, 14))
        learnper = round(Targets.shape[0] * Learnper[i])
        for j in range(sol.shape[0]):
            learnper = round(Data.shape[0] * 0.75)
            train_data = Data[learnper:, :]
            train_target = Targets[learnper:, :]
            test_data = Data[:learnper, :]
            test_target = Targets[:learnper, :]
            Eval = Model_PROPOSED(train_data, train_target, test_data, test_target, sol)
        Train_Data1 = Data[:learnper, :]
        Test_Data1 = Data[learnper:, :]
        Train_Target = Targets[learnper:, :]
        Test_Target = Targets[:learnper, :]
        Eval[5, :] = Model_GRU(Train_Data1, Train_Target, Test_Data1, Test_Target)
        Eval[6, :] = Model_MLP(Train_Data1, Train_Target, Test_Data1, Test_Target)
        Eval[7, :] = Model_AE_LSTM(Train_Data1, Train_Target, Test_Data1, Test_Target)
        Eval[8, :] = Model_AE_LSTM_MLP(Train_Data1, Train_Target, Test_Data1, Test_Target)
        Eval[9, :] = Eval[4, :]
        Eval_all.append(Eval)
    np.save('Eval_all.npy', np.asarray(Eval_all))

## K-Fold
an = 0
if an == 1:
    k_fold = 5
    Eval_all = []
    Data = np.load('Data_transformation.npy', allow_pickle=True)
    sol = np.load('Bestsol.npy', allow_pickle=True)
    Targets = np.load('Target.npy', allow_pickle=True)
    for i in range(k_fold):
        Eval = np.zeros((10, 14))
        for j in range(sol.shape[0]):
            Total_Index = np.arange(Data.shape[0])
            Test_index = np.arange(((i - 1) * (Data.shape[0] / k_fold)) + 1, i * (Data.shape[0] / k_fold))
            Train_Index = np.setdiff1d(Total_Index, Test_index)
            Train_Data = Data[Train_Index, :]
            Train_Target = Targets[Train_Index, :]
            Test_Data = Data[Test_index, :]
            Test_Target = Targets[Test_index, :]
            Eval[i, :] = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target, sol.astype('int'))
        Train_Data1 = Data[Train_Index, :]
        Test_Data1 = Data[Test_index:, :]
        Train_Target = Targets[Train_Index:, :]
        Test_Target = Targets[:Test_index, :]
        Eval[5, :] = Model_GRU(Train_Data1, Train_Target, Test_Data1, Test_Target)
        Eval[6, :] = Model_MLP(Train_Data1, Train_Target, Test_Data1, Test_Target)
        Eval[7, :] = Model_AE_LSTM(Train_Data1, Train_Target, Test_Data1, Test_Target)
        Eval[8, :] = Model_AE_LSTM_MLP(Train_Data1, Train_Target, Test_Data1, Test_Target)
        Eval[9, :] = Eval[4, :]
        Eval_all.append(Eval)
    np.save('Eval_Fold.npy', Eval_all)

plot_results_learnperc()
plot_results_kfold()
