import os
import pickle
import datetime
import argparse
import numpy as np
from sklearn.decomposition import PCA
from transcript_info import Gene_Wrapper
from collections import Counter, defaultdict
import Data.rnalocate.processData as processor


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
# sys.path.append(basedir)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# gpu_options = tf.compat.v1.GPUOptions()
# gpu_options.allow_growth = True
# config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
# tf.compat.v1.keras.backend.set_session(sess)

temp = []
batch_size = 512

# Preprocess_data
def label_dist(dist):
    assert (len(dist) == 4)
    return np.array(dist) / np.sum(dist)

def preprocess_data(lower_bound, upper_bound, use_annotations, dataset, max_len):
    
    ''' import data CEFRA-SEQ: CDNA_SCREENED.FA using GENE_WRAPPER calss fromtranascript_gene_data
    '''
    if(dataset == "rnalocate"):
        X, y, gene_info = processor.process_data()
        gene_info = [gene[1] for gene in gene_info]
    if(dataset == "cefra-seq"):
        gene_data = Gene_Wrapper.seq_data_loader(use_annotations, dataset, lower_bound, upper_bound)
        
        X= [gene.seq for gene in gene_data]
        y = np.array([label_dist(gene.dist) for gene in gene_data])
        gene_info = [gene.id for gene in gene_data]

    print("Shape of X", np.shape(X))
    print("Shape of y", np.shape(y))
    
    return X, y, gene_info

# PPI_Feature
def ppi_preparation(dataset):
    if(dataset == "cefra-seq"):
        print("Cefra-seq PPI inforamtion")
        with open('Data/ppiMatrixScoress.npy', 'rb') as f:
            ppi = np.load(f)
            
    if(dataset == "rnalocate"):
        print("Loadin ppi info")
        with open('Data/ppiHomosapRNALocate.npy', 'rb') as f:
            ppi = np.load(f)
    
    with open("ppiData.pkl", 'wb') as ppis:
        pickle.dump(ppi, ppis)
    
    return ppi

# kNF-PR_Feature
def create_ktf_matrix(sequences, k=4):
    """
    Create the kTF matrix from sequences.
    
    Parameters:
    -----------
    sequences : list of str
        List of mRNA sequences
    k : int, default=4
        Size of the k-mer window
    
    Returns:
    --------
    numpy.ndarray
        kTF matrix with shape (len(sequences), k*max_seq_length)
    """
    sequences = [str(seq).replace("'", "").replace("[", "").replace("]", "").replace(" ", "") for seq in sequences]
    # Add $ at the beginning and two $ at the end of each sequence
    modified_sequences = ["$" + seq + "$$" for seq in sequences]
    
    # Find the maximum sequence length (before adding $ symbols)
    max_seq_length = max(len(seq) for seq in sequences)
    
    # Initialize the kTF matrix
    ktf_matrix = np.zeros((len(sequences), 4 * max_seq_length))
    
    # Process each sequence
    for i, seq in enumerate(modified_sequences):
        # Generate overlapping k-mers
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j+k]
            
            # Count nucleotides in the k-mer
            counts = Counter(kmer)
            
            # Store counts in the matrix (ignoring $ symbols)
            ktf_matrix[i, j*4 + 0] = counts.get('A', 0)
            ktf_matrix[i, j*4 + 1] = counts.get('C', 0)
            ktf_matrix[i, j*4 + 2] = counts.get('G', 0)
            ktf_matrix[i, j*4 + 3] = counts.get('T', 0)
    
    return ktf_matrix

def create_pr_matrix(sequences, labels, k=4):
    """
    Create the PR matrix (PPR/NPR) from sequences and their class labels.
    
    Parameters:
    -----------
    sequences : list of str
        List of mRNA sequences
    labels : list of int
        Class labels for each sequence
    k : int, default=4
        Size of the k-mer window
    
    Returns:
    --------
    numpy.ndarray
        PR matrix with shape (number of classes, 4*max_seq_length)
    """
    sequences = [str(seq).replace("'", "").replace("[", "").replace("]", "").replace(" ", "") for seq in sequences]
    # # Add $ at the beginning and two $ at the end of each sequence
    # modified_sequences = ["$" + seq + "$$" for seq in sequences]
    
    # Find the maximum sequence length (before adding $ symbols)
    max_seq_length = max(len(seq) for seq in sequences)
    
    num_classes = labels.shape[1]
    
    # Initialize PPR and NPR matrices
    ppr_matrix = np.zeros((num_classes, 4 * max_seq_length))
    npr_matrix = np.zeros((num_classes, 4 * max_seq_length))
    
    # Process each position in the sequences
    for pos in range(max_seq_length):  # +2 for the $ symbols
        # Count nucleotides at this position for each class
        for c_idx in range(num_classes):
            # Get sequences in this class
            class_indices = [i for i, label in enumerate(labels) if np.argmax(label) == c_idx]
            class_sequences = [sequences[i] for i in class_indices]
            class_count = len(class_sequences)
            
            # Get sequences not in this class
            other_indices = [i for i, label in enumerate(labels) if np.argmax(label) != c_idx]
            other_sequences = [sequences[i] for i in other_indices]
            other_count = len(other_sequences)
            
            # Count nucleotides at this position
            for nuc_idx, nuc in enumerate(['A', 'C', 'G', 'T']):
                # PPR: Count in this class
                ppr_count = sum(1 for seq in class_sequences if pos < len(seq) and seq[pos] == nuc)
                ppr_matrix[c_idx, pos*4 + nuc_idx] = ppr_count / class_count if class_count > 0 else 0
                
                # NPR: Count in other classes
                npr_count = sum(1 for seq in other_sequences if pos < len(seq) and seq[pos] == nuc)
                npr_matrix[c_idx, pos*4 + nuc_idx] = npr_count / other_count if other_count > 0 else 0
    
    # Avoid division by zero
    npr_matrix = np.where(npr_matrix == 0, 1e-10, npr_matrix)
    
    # Calculate PR matrix
    pr_matrix = ppr_matrix / npr_matrix
    
    return pr_matrix

def multiply_ktf_pr(ktf_matrix, pr_matrix):
    n_samples, F1 = ktf_matrix.shape
    num_classes, F2 = pr_matrix.shape

    if F1 != F2:
        raise ValueError(f"Feature mismatch: ktf has {F1}, pr has {F2}")

    # ضرب المنت به المنت با broadcasting
    result = ktf_matrix[:, None, :] * pr_matrix[None, :, :]  # shape: (n_samples, num_classes, F)

    # reshape to 2D: (n_samples * num_classes, F)
    result_flat = result.reshape(n_samples * num_classes, F1)

    return result_flat

def extract_features(sequences, labels, k=4):
    # Create kTF matrix
    ktf_matrix = create_ktf_matrix(sequences, k)
    
    # Create PR matrix
    pr_matrix = create_pr_matrix(sequences, labels, k)
    
    # Initialize the feature matrix
    feature_matrix = np.zeros_like(ktf_matrix)
    
    # Calculate the final feature matrix
    for i, label in enumerate(labels):
        # Get the class index
        class_idx = np.argmax(label)
        # Multiply the kTF row with the corresponding PR row
        feature_matrix[i] = ktf_matrix[i] * pr_matrix[class_idx]
    
    return feature_matrix

# NucleotidePair_Feature
def extract_basepair_features(sequences):
    sequences = [str(seq).replace("'", "").replace("[", "").replace("]", "").replace(" ", "") for seq in sequences]
    # Define all possible basepairs
    nucleotides = ['A', 'C', 'G', 'T']
    basepairs = [n1 + n2 for n1 in nucleotides for n2 in nucleotides]
    
    # Initialize feature matrix
    feature_matrix = np.zeros((len(sequences), 32))
    
    # Process each sequence
    for i, seq in enumerate(sequences):
        # Handle odd-length sequences (adding $ at the end)
        if len(seq) % 2 != 0:
            seq = seq + '$'
        
        # Cut the sequence in half
        mid_point = len(seq) // 2
        first_half = seq[:mid_point]
        second_half = seq[mid_point:]
        
        # Count basepairs in direct alignment
        direct_counts = count_basepairs(first_half, second_half)
        
        # Count basepairs in reverse alignment
        reverse_counts = count_basepairs(first_half, second_half[::-1])
        
        # Store counts in feature matrix
        for j, bp in enumerate(basepairs):
            feature_matrix[i, j] = direct_counts.get(bp, 0)
            feature_matrix[i, j + 16] = reverse_counts.get(bp, 0)
    
    return feature_matrix

def count_basepairs(first_half, second_half):
    """
    Count basepairs between two halves of a sequence.
    
    Parameters:
    -----------
    first_half : str
        First half of the sequence
    second_half : str
        Second half of the sequence
    
    Returns:
    --------
    dict
        Dictionary with basepairs as keys and counts as values
    """
    counts = defaultdict(int)
    
    # Ensure both halves are the same length
    min_length = min(len(first_half), len(second_half))
    
    # Count basepairs
    for i in range(min_length):
        # Skip if either nucleotide is not A, C, G, or T
        if first_half[i] not in 'ACGT' or second_half[i] not in 'ACGT':
            continue
        
        # Create basepair key and increment count
        bp = first_half[i] + second_half[i]
        counts[bp] += 1
    
    return counts


def run_model(lower_bound, upper_bound, max_len, dataset, **kwargs):
    '''load data into the playground'''

    X, y, gene_info = preprocess_data(lower_bound, upper_bound,max_len, dataset, max_len)
    X = np.array(X)
    gene_info = np.array(gene_info)
    y = np.array(y)
    np.random.seed(42)
    indices = np.arange(len(X), dtype=int)
    np.random.shuffle(indices)
    split_point = int(0.8 * len(indices))
    train_indices, test_indices = indices[:split_point], indices[split_point:]
    X_train, X_test = X[train_indices], X[test_indices]
    gene_test = gene_info[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
###############################################################################################################################
    if dataset == 'cefra-seq':
        # Add PPI :
        if os.path.exists("pkl_data\ppi_cefra.pkl"):
            with open("pkl_data\ppi_cefra.pkl", 'rb') as ppce:
                ppi = pickle.load(ppce)
        else:
            ppi = ppi_preparation(dataset)
            with open('pkl_data\ppi_cefra.pkl', 'wb') as ppce:
                pickle.dump(ppi, ppce)
        
        if os.path.exists("pkl_data\ppi_train_cefra.pkl") and os.path.exists("pkl_data\ppi_test_cefra.pkl") and os.path.exists("pkl_data\ppi_pca_test_cefra.pkl") and os.path.exists("pkl_data\ppi_pca_test_cefra.pkl"):
            with open('pkl_data\ppi_train_cefra.pkl', 'rb') as ppitr:
                ppi_train = pickle.load(ppitr)
            with open('pkl_data\ppi_test_cefra.pkl', 'rb') as ppite:
                ppi_test = pickle.load(ppite)
            
            with open('pkl_data\ppi_pca_train_cefra.pkl', 'rb') as ppipcatr:
                ppi_train = pickle.load(ppipcatr)
            with open('pkl_data\ppi_pca_test_cefra.pkl', 'rb') as ppipcate:
                ppi_test = pickle.load(ppipcate) 
        else:
            ppi_train, ppi_test = ppi[train_indices], ppi[test_indices]
            
            with open('pkl_data\ppi_train_cefra.pkl', 'wb') as ppitr:
                pickle.dump(ppi_train, ppitr)
            with open('pkl_data\ppi_test_cefra.pkl', 'wb') as ppite:
                pickle.dump(ppi_test, ppite)
            
            ppipca = PCA(n_components=0.95)
            ppi_train = ppipca.fit_transform(ppi_train)
            ppi_test = ppipca.transform(ppi_test)
            
            with open('pkl_data\ppi_pca_train_cefra.pkl', 'wb') as ppipcatr:
                pickle.dump(ppi_train, ppipcatr)
            with open('pkl_data\ppi_pca_test_cefra.pkl', 'wb') as ppipcate:
                pickle.dump(ppi_test, ppipcate)
        
        # bp-check
        if os.path.exists("pkl_data\Bp_train_cefra.pkl") and os.path.exists("pkl_data\Bp_test_cefra.pkl"):
            with open("pkl_data\Bp_train_cefra.pkl", 'rb') as bptra:
                bp_train = pickle.load(bptra)
            with open("pkl_data\Bp_test_cefra.pkl", 'rb') as bptes:
                bp_test = pickle.load(bptes)
        else:
            bp_train = extract_basepair_features(X_train)
            bp_test = extract_basepair_features(X_test)
            with open('pkl_data\Bp_train_cefra.pkl', 'wb') as bptra:
                pickle.dump(bp_train, bptra)
            with open('pkl_data\Bp_test_cefra.pkl', 'wb') as bptes:
                pickle.dump(bp_test, bptes)
        
        # kTF-NucleotidePR :
        if os.path.exists("pkl_data\kTF_NPR_cefra.pkl") and os.path.exists("pkl_data\kTF_cefra.pkl") and os.path.exists("pkl_data\kTF_NPR_pca_cefra.pkl") and os.path.exists("pkl_data\kTF_pca_cefra.pkl") and os.path.exists("pkl_data\kTFPR_cefra.pkl") and os.path.exists("pkl_data\kTFPR_pca_cefra.pkl"):
            with open("pkl_data\kTF_NPR_cefra.pkl", 'rb') as cef:
                kTF_NPR = pickle.load(cef)
            with open("pkl_data\kTF_cefra.pkl", 'rb') as kcef:
                kTF = pickle.load(kcef)
            with open("pkl_data\kTF_NPR_pca_cefra.pkl", 'rb') as cefpca:
                kTF_NPR_pca = pickle.load(cefpca)
            with open("pkl_data\kTF_pca_cefra.pkl", 'rb') as kcefpca:
                kTF_pca = pickle.load(kcefpca)
            with open("pkl_data\kTFPR_cefra.pkl", 'rb') as kpcef:
                kTFPR = pickle.load(kpcef)
            with open("pkl_data\kTFPR_pca_cefra.pkl", 'rb') as kpcefpca:
                kTFPR_pca = pickle.load(kpcefpca)
        else:
            kTF_NPR = extract_features(X_train, y_train, k=4)
            with open('pkl_data\kTF_NPR_cefra.pkl', 'wb') as ktfnpr:
                pickle.dump(kTF_NPR, ktfnpr)
                
            kTF = create_ktf_matrix(X_test, k=4)
            with open('pkl_data\kTF_cefra.pkl', 'wb') as ktf:
                pickle.dump(kTF, ktf)
            
            PRmatrix = create_pr_matrix(X_test, y_test, k=4)
            kTFPR = multiply_ktf_pr(kTF, PRmatrix)
            with open('pkl_data\kTFPR_cefra.pkl', 'wb') as ktfprr:
                pickle.dump(kTFPR, ktfprr)
            with open("pkl_data\kTF_NPR_cefra.pkl", 'rb') as cef:
                kTF_NPR = pickle.load(cef)
            with open("pkl_data\kTF_cefra.pkl", 'rb') as kcef:
                kTF = pickle.load(kcef)
            with open("pkl_data\kTFPR_cefra.pkl", 'rb') as kpcef:
                kTFPR = pickle.load(kpcef)
            padding = np.zeros((kTF.shape[0], kTF_NPR.shape[1] - kTF.shape[1]))
            kTF = np.hstack((kTF, padding))
                
            kpca = PCA(n_components=1000)
            kTF_NPR_pca = kpca.fit_transform(kTF_NPR)
            kTF_pca = kpca.transform(kTF)
            kTFPR_pca = kpca.transform(kTFPR)

            with open('pkl_data\kTF_NPR_pca_cefra.pkl', 'wb') as kcef:
                pickle.dump(kTF_NPR_pca, kcef)
            with open('pkl_data\kTF_pca_cefra.pkl', 'wb') as cef:
                pickle.dump(kTF_pca, cef)
            with open('pkl_data\kTFPR_pca_cefra.pkl', 'wb') as kpcef:
                pickle.dump(kTFPR_pca, kpcef)
        
        with open('pkl_data\y_train_cefra.pkl', 'wb') as ytr:
            pickle.dump(y_train, ytr)
        with open('pkl_data\y_test_cefra.pkl', 'wb') as yte:
            pickle.dump(y_test, yte)
        
        print(f"{dataset} data and features prepared succesfully!")
###############################################################################################################################
    elif dataset == 'rnalocate':
        # Add PPI :
        if os.path.exists("pkl_data\ppi_rnalocate.pkl"):
            with open("pkl_data\ppi_rnalocate.pkl", 'rb') as ppce:
                ppi = pickle.load(ppce)
        else:
            ppi = ppi_preparation(dataset)
            with open('pkl_data\ppi_rnalocate.pkl', 'wb') as ppce:
                pickle.dump(ppi, ppce)
        
        if os.path.exists("pkl_data\ppi_train_rnalocate.pkl") and os.path.exists("pkl_data\ppi_test_rnalocate.pkl") and os.path.exists("pkl_data\ppi_pca_train_rnalocate.pkl") and os.path.exists("pkl_data\ppi_pca_test_rnalocate.pkl"):
            with open('pkl_data\ppi_train_rnalocate.pkl', 'rb') as ppitr:
                ppi_train = pickle.load(ppitr)
            with open('pkl_data\ppi_test_rnalocate.pkl', 'rb') as ppite:
                ppi_test = pickle.load(ppite)
            
            with open('pkl_data\ppi_pca_train_rnalocate.pkl', 'rb') as ppipcatr:
                ppi_train = pickle.load(ppipcatr)
            with open('pkl_data\ppi_pca_test_rnalocate.pkl', 'rb') as ppipcate:
                ppi_test = pickle.load(ppipcate) 
        else:
            ppi_train, ppi_test = ppi[train_indices], ppi[test_indices]
            
            with open('pkl_data\ppi_train_rnalocate.pkl', 'wb') as ppitr:
                pickle.dump(ppi_train, ppitr)
            with open('pkl_data\ppi_test_rnalocate.pkl', 'wb') as ppite:
                pickle.dump(ppi_test, ppite)
            
            ppipca = PCA(n_components=0.95)
            ppi_train = ppipca.fit_transform(ppi_train)
            ppi_test = ppipca.transform(ppi_test)
            
            with open('pkl_data\ppi_pca_train_rnalocate.pkl', 'wb') as ppipcatr:
                pickle.dump(ppi_train, ppipcatr)
            with open('pkl_data\ppi_pca_test_rnalocate.pkl', 'wb') as ppipcate:
                pickle.dump(ppi_test, ppipcate)
        
        # bp-check
        if os.path.exists("pkl_data\BP_Train_rnalocate.pkl") and os.path.exists("pkl_data\BP_Test_rnalocate.pkl"):
            with open("pkl_data\BP_Train_rnalocate.pkl", 'rb') as bptra:
                bp_train = pickle.load(bptra)
            with open("pkl_data\BP_Test_rnalocate.pkl", 'rb') as bptes:
                bp_test = pickle.load(bptes)
        else:
            bp_train = extract_basepair_features(X_train)
            bp_test = extract_basepair_features(X_test)
            with open('pkl_data\BP_Train_rnalocate.pkl', 'wb') as bptra:
                pickle.dump(bp_train, bptra)
            with open('pkl_data\BP_Test_rnalocate.pkl', 'wb') as bptes:
                pickle.dump(bp_test, bptes)

        
        # kTF-NucleotidePR :
        if os.path.exists("pkl_data\kTF_NPR_rnalocate.pkl") and os.path.exists("pkl_data\kTF_rnalocate.pkl") and os.path.exists("pkl_data\kTF_NPR_pca_rnalocate.pkl") and os.path.exists("pkl_data\kTF_pca_rnalocate.pkl") and os.path.exists("pkl_data\kTFPR_rnalocate.pkl") and os.path.exists("pkl_data\kTFPR_pca_rnalocate.pkl"):
            with open("pkl_data\kTF_NPR_rnalocate.pkl", 'rb') as cef:
                kTF_NPR = pickle.load(cef)
            with open("pkl_data\kTF_rnalocate.pkl", 'rb') as kcef:
                kTF = pickle.load(kcef)
            with open("pkl_data\kTF_NPR_pca_rnalocate.pkl", 'rb') as cefpca:
                kTF_NPR_pca = pickle.load(cefpca)
            with open("pkl_data\kTF_pca_rnalocate.pkl", 'rb') as kcefpca:
                kTF_pca = pickle.load(kcefpca)
            with open("pkl_data\kTFPR_rnalocate.pkl", 'rb') as kpcef:
                kTFPR = pickle.load(kpcef)
            with open("pkl_data\kTFPR_pca_rnalocate.pkl", 'rb') as kpcefpca:
                kTFPR_pca = pickle.load(kpcefpca)
        else:
            kTF_NPR = extract_features(X_train, y_train, k=4)
            with open('pkl_data\kTF_NPR_rnalocate.pkl', 'wb') as ktfnpr:
                pickle.dump(kTF_NPR, ktfnpr)
                
            kTF = create_ktf_matrix(X_test, k=4)
            with open('pkl_data\kTF_rnalocate.pkl', 'wb') as ktf:
                pickle.dump(kTF, ktf)
            
            PRmatrix = create_pr_matrix(X_test, y_test, k=4)
            kTFPR = multiply_ktf_pr(kTF, PRmatrix)
            with open('pkl_data\kTFPR_rnalocate.pkl', 'wb') as ktfprr:
                pickle.dump(kTFPR, ktfprr)
            padding = np.zeros((kTF.shape[0], kTF_NPR.shape[1] - kTF.shape[1]))
            kTF = np.hstack((kTF, padding))
                
            kpca = PCA(n_components=1000)
            kTF_NPR_pca = kpca.fit_transform(kTF_NPR)
            kTF_pca = kpca.transform(kTF)
            kTFPR_pca = kpca.transform(kTFPR)

            with open('pkl_data\kTF_NPR_pca_rnalocate.pkl', 'wb') as kcef:
                pickle.dump(kTF_NPR_pca, kcef)
            with open('pkl_data\kTF_pca_rnalocate.pkl', 'wb') as cef:
                pickle.dump(kTF_pca, cef)
            with open('pkl_data\kTFPR_pca_rnalocate.pkl', 'wb') as kpcef:
                pickle.dump(kTFPR_pca, kpcef)
        
        with open('pkl_data\y_train_rnalocate.pkl', 'wb') as ytr:
            pickle.dump(y_train, ytr)
        with open('pkl_data\y_test_rnalocate.pkl', 'wb') as yte:
            pickle.dump(y_test, yte)
        
        print(f"{dataset} data and features prepared succesfully!")
###############################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''Model parameters'''
    parser.add_argument('--lower_bound', type=int, default=0, help='set lower bound on sample sequence length')
    parser.add_argument('--upper_bound', type=int, default= 40000, help='set upper bound on sample sequence length') #default=4000
    parser.add_argument('--max_len', type=int, default=40000,
                        help="dummy, pad or slice sequences to a fixed length in preprocessing")
    parser.add_argument('--dataset', type=str, default='cefra-seq', choices=['cefra-seq', 'rnalocate'],
                        help='choose from cefra-seq and rnalocate')
    parser.add_argument('--epochs', type=int, default=300, help='')
    parser.add_argument("--message", type=str, default="", help="append to the dir name")
    parser.add_argument("--load_pretrain", action="store_true",
                        help="load pretrained CNN weights to the first convolutional layers")
    parser.add_argument("--weights_dir", type=str, default="",
                        help="Must specificy pretrained weights dir, if load_pretrain is set to true. Only enter the relative path respective to the root of this project.")
    parser.add_argument("--randomization", type=int, default=None,
                        help="Running randomization test with three settings - {1,2,3}.")
    # parser.add_argument("--nb_epochs", type=int, default=20, help='choose the maximum number of iterations over training samples')
    args = parser.parse_args()

        
    OUTPATH = os.path.join(basedir,
                               'Results/' + args.dataset+ '/' + str(datetime.datetime.now()).split('.')[0].replace(':', '-').replace(' ', '-') + '-' + args.message + '/')
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    print('OUTPATH:', OUTPATH)

    args.weights_dir = os.path.join(basedir, args.weights_dir)

    for k, v in vars(args).items():
        print(k, ':', v)

    
    run_model(**vars(args))
