import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

def plot_results_learnperc():
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term =[0,3,4,5,7,8]
    Algorithm = ['TERMS', 'DHOA', 'COA', 'GSO', 'SOA', 'PROPOSED']
    Classifier = ['TERMS', 'GRU', 'MLP', 'AE_LSTM', 'AE_LSTM_MLP', 'PROPOSED']
    # Dataset = ['Eclipse JDT Core Dataset','Eclipse PDE UI Dataset','Equinox Framework Dataset']

    value = Eval[ 4, :, 4:]
    value[:, :-1] = value[:, :-1] * 100
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value[j, :])
    print('--------------------------------------------------  Algorithm Comparison - ',
          'Learning Percentage --------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
    print('-------------------------------------------------- Classifier Comparison - ',
          'Learning Percentage --------------------------------------------------')
    print(Table)

    Eval = np.load('Eval_all.npy', allow_pickle=True)
    learnper = [35, 55, 65, 75, 85]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[1]):
                if Graph_Term[j] == 9:
                    Graph[k, l] = Eval[ k, l, Graph_Term[j] + 4]
                else:
                    Graph[k, l] = Eval[ k, l, Graph_Term[j] + 4] * 100






        plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                 label="DHOA-ASCALSMLP")
        plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                 label="COA-ASCALSMLP")
        plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                 label="GSO-ASCALSMLP")
        plt.plot(learnper, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                 label="SOA-ASCALSMLP")
        plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                 label="SGSO-ASCALSMLP")
        plt.xlabel('Learning Percentage (%)')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc=4)
        path1 = "./Results/Dataset_%s_line_1.png" % ( Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()



        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="GRU")
        ax.bar(X + 0.10, Graph[:, 6], color='#cc9f3f', width=0.10, label="MLP")
        ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="AE_LSTM")
        ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="AE_LSTM_MLP")
        ax.bar(X + 0.40, Graph[:, 9], color='c', width=0.10, label="SGSO - ASCALSMLP")
        plt.xticks(X + 0.25, ('35', '55', '65', '75', '85'))
        plt.xlabel('Learning Percentage (%)')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc=1)
        path1 = "./Results/Dataset_%s_bar_1.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()



def plot_results_kfold():
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term =[0,3,4,5,7,8]

    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    learnper = [1, 2, 3, 4, 5]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        # Graph = np.zeros(eval.shape[1:3])
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                if Graph_Term[j] == 9:
                    Graph[k, l] = eval[k, l, Graph_Term[j]+4]
                else:
                    Graph[k, l] = eval[k, l, Graph_Term[j] + 4] * 100

        plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                 label="DHOA-ASCALSMLP")
        plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                 label="COA-ASCALSMLP")
        plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                 label="GSO-ASCALSMLP")
        plt.plot(learnper, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                 label="SOA-ASCALSMLP")
        plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                 label="SGSO-ASCALSMLP")
        plt.xlabel('K - Fold')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc=4)
        # "./Results/%s_line_1.png" % (Terms[Graph_Term[j]])
        path1 = "./Results/Dataset_%s_line_2.png" % ( Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="GRU")
        ax.bar(X + 0.10, Graph[:, 6], color='#cc9f3f', width=0.10, label="MLP")
        ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="AE_LSTM")
        ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="AE_LSTM_MLP")
        ax.bar(X + 0.40, Graph[:, 9], color='c', width=0.10, label="SGSO - ASCALSMLP")
        plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
        plt.xlabel('K - Fold')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc=1)
        path1 = "./Results/Dataset_%s_bar_2.png" % ( Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

if __name__ == "__main__":

    plot_results_learnperc()
    plot_results_kfold()


