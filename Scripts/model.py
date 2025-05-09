import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python.keras
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Model
from imblearn.over_sampling import RandomOverSampler
from tensorflow.python.keras.layers.core import Dense
from imblearn.ensemble import BalancedBaggingClassifier
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.layers import Input, Dense, Concatenate, Attention
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
OUTPATH = None

class RNALocator:
    def __init__(self, max_len, nb_classes, save_path, index):
        print("Constructing RNALocator class")
        print("Number of classes is", nb_classes)
        self.max_len = max_len
        self.nb_classes = nb_classes
        self.is_built = False
        global OUTPATH
        OUTPATH = save_path
        self.index = index


    def build_neural_network(self, NucP_dim, kNF_PR_dim, ppi_dim, dataset):
        NucP_input = Input(shape=(NucP_dim,), name='NucP')
        kNF_PR_input = Input(shape=(kNF_PR_dim,), name='kNF_PR')
        ppi_input = Input(shape=(ppi_dim,), name='ppi')
        dense_layer = 0
        if dataset == 'cefra-seq':
            dense_layer = 100
        else:
            dense_layer = 30
        
        # 1st Attention Set :
        NucP_proj = Dense(dense_layer, activation='linear', name='NucP_NucP_projection')(NucP_input)
        NucP_proj = tf.keras.layers.BatchNormalization()(NucP_proj)
        NucP_expanded = tf.expand_dims(NucP_proj, axis=1)
        kNF_PR_proj = Dense(dense_layer, activation='linear', name='kNF_PR_NucP_projection')(kNF_PR_input)
        kNF_PR_proj = tf.keras.layers.BatchNormalization()(kNF_PR_proj)
        kNF_PR_expanded = tf.expand_dims(kNF_PR_proj, axis=1)
        ppi_proj = Dense(dense_layer, activation='linear', name='ppi_NucP_projection')(ppi_input)
        ppi_proj = tf.keras.layers.BatchNormalization()(ppi_proj)
        ppi_expanded = tf.expand_dims(ppi_proj, axis=1)
        NucP_att = Attention(name='NucP_self_attention')([NucP_expanded, NucP_expanded, NucP_expanded])
        kNF_PR_att = Attention(name='kNF_PR_NucP_attention')([kNF_PR_expanded, NucP_expanded, NucP_expanded])
        ppi_att = Attention(name='ppi_NucP_attention')([ppi_expanded, NucP_expanded, NucP_expanded])
        NucP_att = tf.squeeze(NucP_att, axis=1)
        kNF_PR_att = tf.squeeze(kNF_PR_att, axis=1)
        ppi_att = tf.squeeze(ppi_att, axis=1)
        feature_vector1 = Concatenate(name='feature_vector1')([NucP_att, kNF_PR_att, ppi_att])
        
        # 2nd Attention Set :
        NucP_proj = Dense(dense_layer, activation='linear', name='NucP_kNF_PR_projection')(NucP_input)
        NucP_proj = tf.keras.layers.BatchNormalization()(NucP_proj)
        NucP_expanded = tf.expand_dims(NucP_proj, axis=1)
        kNF_PR_proj = Dense(dense_layer, activation='linear', name='kNF_PR_kNF_PR_projection')(kNF_PR_input)
        kNF_PR_proj = tf.keras.layers.BatchNormalization()(kNF_PR_proj)
        kNF_PR_expanded = tf.expand_dims(kNF_PR_proj, axis=1)
        ppi_proj = Dense(dense_layer, activation='linear', name='ppi_kNF_PR_projection')(ppi_input)
        ppi_proj = tf.keras.layers.BatchNormalization()(ppi_proj)
        ppi_expanded = tf.expand_dims(ppi_proj, axis=1)
        NucP_att = Attention(name='kNF_PR_self_attention')([kNF_PR_expanded, kNF_PR_expanded, kNF_PR_expanded])
        kNF_PR_att = Attention(name='NucP_kNF_PR_attention')([NucP_expanded, kNF_PR_expanded, kNF_PR_expanded])
        ppi_att = Attention(name='ppi_kNF_PR_attention')([ppi_expanded, kNF_PR_expanded, kNF_PR_expanded])
        NucP_att = tf.squeeze(NucP_att, axis=1)
        kNF_PR_att = tf.squeeze(kNF_PR_att, axis=1)
        ppi_att = tf.squeeze(ppi_att, axis=1)
        feature_vector2 = Concatenate(name='feature_vector2')([NucP_att, kNF_PR_att, ppi_att])
        
        # 3rd Attention Set :
        NucP_proj = Dense(dense_layer, activation='linear', name='NucP_ppi_projection')(NucP_input)
        NucP_proj = tf.keras.layers.BatchNormalization()(NucP_proj)
        NucP_expanded = tf.expand_dims(NucP_proj, axis=1)
        kNF_PR_proj = Dense(dense_layer, activation='linear', name='kNF_PR_ppi_projection')(kNF_PR_input)
        kNF_PR_proj = tf.keras.layers.BatchNormalization()(kNF_PR_proj)
        kNF_PR_expanded = tf.expand_dims(kNF_PR_proj, axis=1)
        ppi_proj = Dense(dense_layer, activation='linear', name='ppi_ppi_projection')(ppi_input)
        ppi_proj = tf.keras.layers.BatchNormalization()(ppi_proj)
        ppi_expanded = tf.expand_dims(ppi_proj, axis=1)
        NucP_att = Attention(name='ppi_self_attention')([ppi_expanded, ppi_expanded, ppi_expanded])
        kNF_PR_att = Attention(name='kNF_PR_ppi_attention')([kNF_PR_expanded, ppi_expanded, ppi_expanded])
        ppi_att = Attention(name='NucP_ppi_attention')([NucP_expanded, ppi_expanded, ppi_expanded])
        NucP_att = tf.squeeze(NucP_att, axis=1)
        kNF_PR_att = tf.squeeze(kNF_PR_att, axis=1)
        ppi_att = tf.squeeze(ppi_att, axis=1)
        feature_vector3 = Concatenate(name='feature_vector3')([NucP_att, kNF_PR_att, ppi_att])
        
        concat_feature_vectors = Concatenate(name='Concatenated_Vectors')([feature_vector1, feature_vector2, feature_vector3])
        
        first = Dense(1024, activation='relu', name='First_Dense')(concat_feature_vectors)
        second = Dense(512, activation='relu', name='Second_Dense')(first)
        last = Dense(self.nb_classes, activation='softmax', name='Last_Dense')(second)
        
        model = Model(inputs=[NucP_input, kNF_PR_input, ppi_input], outputs=last)
        self.model = model
        self.model.compile(optimizer='nadam', loss=CategoricalCrossentropy(), metrics=['accuracy'])
        self.is_built = True
        self.bn = False
        self.model.summary()

        self.is_built = True
        self.bn = False
        self.model.summary()    

    
    def train(self, x_train, y_train, batch_size, NucP_dim, kNF_PR_dim, ppi_dim, dataset, epochs=300):
        os.makedirs("train_val_log", exist_ok=True)
        x_train = [x_train[:, :NucP_dim], x_train[:, NucP_dim:NucP_dim+kNF_PR_dim], x_train[:, NucP_dim+kNF_PR_dim:NucP_dim+kNF_PR_dim+ppi_dim]]
        best_model_path = OUTPATH + 'weights_Run_{}.h5'.format(self.index)
        model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, verbose=1)

        print("Initial evaluation:")
        print("Train:", self.model.evaluate(x_train, y_train, batch_size=batch_size))

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            hist = self.model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=1,
                verbose=1,
                callbacks=[model_checkpoint],
                shuffle=True
            )

            if epoch == 0:
                full_hist = hist.history
            else:
                for key in hist.history:
                    full_hist[key].extend(hist.history[key])

        if dataset == 'cefra-seq':
            self.model.save_weights('cefraseq\\trained_weights.h5')
        elif dataset == 'rnalocate':
            self.model.save_weights('rnalocate\\trained_weights.h5')
        
        metrics_df = pd.DataFrame(full_hist)
        metrics_df.to_csv(f"train_val_log/train_val_metrics_{dataset}.csv", index=False)

        plt.style.use('seaborn-v0_8-white')
        plt.rcParams.update({
            'savefig.facecolor': 'white',
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.labelweight': 'bold',
            'axes.linewidth': 1.5,
            'font.size': 20,
            'legend.fontsize': 13,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'figure.titlesize': 20,
            'axes.titlesize': 18
        })

        plt.figure(figsize=(15, 15))
        plt.plot(full_hist['loss'], label='Train Loss', color='tomato', linewidth=2)
        if 'val_loss' in full_hist:
            plt.plot(full_hist['val_loss'], label='Validation Loss', color='dodgerblue', linewidth=2)
        plt.title('Loss per Epoch', fontweight='bold')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('Loss', fontweight='bold')
        plt.xticks(rotation=0,fontsize=20)
        plt.yticks(rotation=0,fontsize=20)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(2)
        plt.tight_layout()
        plt.savefig(f"train_val_log/train_val_loss_{dataset}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(15, 15))
        plt.plot(full_hist['accuracy'], label='Train Accuracy', color='forestgreen', linewidth=2)
        if 'val_accuracy' in full_hist:
            plt.plot(full_hist['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
        plt.title('Accuracy per Epoch', fontweight='bold')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('Accuracy', fontweight='bold')
        plt.xticks(rotation=0,fontsize=20)
        plt.yticks(rotation=0,fontsize=20)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(2)
        plt.tight_layout()
        plt.savefig(f"train_val_log/train_val_accuracy_{dataset}.png", dpi=300)
        plt.close()
    
    
    def balance_class_eval(self, x_train, y_train, x_test, y_test, dataset):
        if dataset == 'cefra-seq':
            self.model.load_weights('cefraseq\\trained_weights.h5')
        elif dataset == 'rnalocate':
            self.model.load_weights('rnalocate\\trained_weights.h5')
        
        intermediate_model = Model(inputs=self.model.inputs,
                                outputs=self.model.get_layer('Second_Dense').output)

        train_features = intermediate_model.predict([x_train[:,:32],x_train[:,32:1032],x_train[:,1032:]])
        test_features = intermediate_model.predict([x_test[:,:32],x_test[:,32:1032],x_test[:,1032:]])
        
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
        
        with open(f"x_train_{dataset}.pkl", "wb") as xtr:
            pickle.dump(train_features, xtr)
        with open(f"x_test_{dataset}.pkl", "wb") as xte:
            pickle.dump(test_features, xte)
        with open(f"y_train_{dataset}.pkl", "wb") as ytr:
            pickle.dump(y_train, ytr)
        with open(f"y_test_{dataset}.pkl", "wb") as yte:
            pickle.dump(y_test, yte)
        
        print('Resampled dataset shape %s' % Counter(np.concatenate([y_train, y_test])))
            
        resamp_model = RandomOverSampler(random_state=42)
        X_augmented, y_augmented = resamp_model.fit_resample(np.concatenate([train_features, test_features]), np.concatenate([y_train, y_test]))
        model = BalancedBaggingClassifier(random_state=42)
        model.fit(X_augmented, y_augmented)
        y_pred = model.predict(test_features)
        
        print(accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        

        cm = confusion_matrix(y_test, y_pred)
        class_accuracy = cm.diagonal() / cm.sum(axis=1) 

        for i, acc in enumerate(class_accuracy):
            print(f"Accuracy for class {i}: {acc:.2f}")
        print("done !")
        
        

        encoder = OneHotEncoder(sparse_output=False) 
        y_true_onehot = encoder.fit_transform(y_test.reshape(-1, 1))
        y_pred_onehot = encoder.transform(y_pred.reshape(-1, 1))

        n_classes = y_true_onehot.shape[1]
        for i in range(n_classes):
            corr = np.corrcoef(y_true_onehot[:, i], y_pred_onehot[:, i])[0, 1]
            print(f"Pearson for class {i}: {corr:.3f}")
        
