import os
import pickle
import datetime
import numpy as np
from model import RNALocator
from sklearn import preprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
batch_size = 512
dataset = 'cefra-seq'

if dataset == 'cefra-seq':
    print(f"Loading Dataset {dataset} From Cache... ")
    
    with open("pkl_data\ppi_pca_train_cefra.pkl", 'rb') as ppitr:
        ppi_train = pickle.load(ppitr)
    with open("pkl_data\ppi_pca_test_cefra.pkl", 'rb') as ppite:
        ppi_test = pickle.load(ppite)

    with open("pkl_data\Bp_train_cefra.pkl", 'rb') as bptr:
        bp_train = pickle.load(bptr)
    with open("pkl_data\Bp_test_cefra.pkl", 'rb') as bpte:
        bp_test = pickle.load(bpte)

    with open("pkl_data\kTF_NPR_pca_cefra.pkl", 'rb') as ktftr:
        ktf_train = pickle.load(ktftr)
    with open("pkl_data\kTF_pca_cefra.pkl", 'rb') as ktfte:
        ktf_test = pickle.load(ktfte)
    
    with open("pkl_data\y_train_cefra.pkl", 'rb') as ytr:
        y_train = pickle.load(ytr)
    with open("pkl_data\y_test_cefra.pkl", 'rb') as yte:
        y_test = pickle.load(yte)
elif dataset == 'rnalocate':
    print(f"Loading Dataset {dataset} From Cache... ")
    
    with open("pkl_data\ppi_pca_train_rnalocate.pkl", 'rb') as ppitr:
        ppi_train = pickle.load(ppitr)
    with open("pkl_data\ppi_pca_test_rnalocate.pkl", 'rb') as ppite:
        ppi_test = pickle.load(ppite)

    with open("pkl_data\Bp_train_rnalocate.pkl", 'rb') as bptr:
        bp_train = pickle.load(bptr)
    with open("pkl_data\Bp_test_rnalocate.pkl", 'rb') as bpte:
        bp_test = pickle.load(bpte)

    with open("pkl_data\kTF_NPR_pca_rnalocate.pkl", 'rb') as ktftr:
        ktf_train = pickle.load(ktftr)
    with open("pkl_data\kTF_pca_rnalocate.pkl", 'rb') as ktfte:
        ktf_test = pickle.load(ktfte)
        
    with open("pkl_data\y_train_rnalocate.pkl", 'rb') as ytr:
        y_train = pickle.load(ytr)
    with open("pkl_data\y_test_rnalocate.pkl", 'rb') as yte:
        y_test = pickle.load(yte)

    
scaler_bp = preprocessing.MinMaxScaler()
bp_train = scaler_bp.fit_transform(bp_train)
bp_test = scaler_bp.transform(bp_test)

scaler_ktf = preprocessing.MinMaxScaler()
ktf_train = scaler_ktf.fit_transform(ktf_train)
ktf_test = scaler_ktf.transform(ktf_test)

scaler_ppi = preprocessing.MinMaxScaler()
ppi_train = scaler_ppi.fit_transform(ppi_train)
ppi_test = scaler_ppi.transform(ppi_test)

newDataTrain = np.concatenate((bp_train, ktf_train, ppi_train), axis = 1)
newDataTest = np.concatenate((bp_test, ktf_test, ppi_test), axis = 1)
    
y_train = (y_train == y_train.max(axis=1, keepdims=True)).astype(int)
y_test = (y_test == y_test.max(axis=1, keepdims=True)).astype(int)


OUTPATH = os.path.join(basedir,'Results/' + dataset+ '/' + str(datetime.datetime.now()).split('.')[0].replace(':', '-').replace(' ', '-'))

if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)

if dataset == 'cefra-seq':
    nb_classes = 4
    NucP_dim, kNF_PR_dim, ppi_dim = 32, 1000, 3275
elif dataset == 'rnalocate':
    nb_classes = 5
    NucP_dim, kNF_PR_dim, ppi_dim = 32, 1000, 1


model = RNALocator(nb_classes, OUTPATH, max_len=40000, index=0)
print("Build RNALocator")
model.build_neural_network(NucP_dim=NucP_dim, kNF_PR_dim=kNF_PR_dim, ppi_dim=ppi_dim, dataset=dataset)
model.train(newDataTrain, y_train, batch_size, NucP_dim=NucP_dim, kNF_PR_dim=kNF_PR_dim, ppi_dim=ppi_dim, dataset=dataset, epochs=300)
model.balance_class_eval(newDataTrain, y_train,newDataTest,y_test, dataset)