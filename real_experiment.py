#!/usr/bin/env python

import numpy as np
import pandas as pd
from scm import SCM
from helper import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB

import networkx as nx
from networkx.algorithms.approximation import max_clique

np.random.seed(312253)
np.set_printoptions(precision=3)
class RealExperiment:
    def __init__(self, n_classes, n_experts, seed=44):
        self.rng = np.random.default_rng(seed)
        self.n_classes = n_classes
        self.n_experts = n_experts

        self.proba_models = []
        self.nb_baseline = []
        self.list_proba_func = []

        self.scm_naive = None
        self.scm_model = None
    
    def add_experts(self, expert_list, data, labels):
        new_proba_func_list = self.fit_proba_models(expert_list, data, labels)
        self.list_proba_func.extend(new_proba_func_list)
        self.n_experts = len(self.list_proba_func)

    def get_proba(self, x, clf):
        classes_not_trained = set(clf.classes_).symmetric_difference(range(self.n_classes))
        #print(classes_not_trained)
        prob = clf.predict_proba(np.expand_dims(x, axis=0))
        #print(prob)
        if len(classes_not_trained)>0:
            new_prob = np.empty((self.n_classes))
            new_prob[list(classes_not_trained)] =  0.00000001
            new_prob[clf.classes_] = prob[0]
            new_prob[new_prob==0.0] = 0.00000001
            return new_prob
            
        prob = np.squeeze(prob, axis=0)
        return np.where( prob > 0.0, prob,  0.00000001)

    def fit_proba_models(self,expert_list, data, labels):
      for expert in expert_list:
        X_train = data[labels[:, expert]!=-999]
        y_train = labels[labels[:,expert]!=-999][:,expert]

        gnb = GaussianNB()

        gnb.fit(X_train, y_train)
        self.proba_models.append(gnb)

      return [(lambda x, clf=clf: self.get_proba(x, clf)) for clf in self.proba_models]
       
    def update_model(self, data_train, labels_train):
        #our model
        if self.scm_model == None:
            self.scm_model = SCM("Trained", self.n_classes, self.list_proba_func)
            self.scm_model.fit( data_train, labels_train, val_ratio=0.0, max_rounds=10)
        
        #naive
        if self.scm_naive == None:
            self.scm_naive = SCM("Naive", self.n_classes, self.list_proba_func, naive= True)

    def evaluate_marginal_estimators(self, data, labels):
        scores = np.zeros((self.n_experts))
        test_predict = np.zeros((self.n_experts))
        for expert, clf in enumerate(self.proba_models):
            has_label=labels[:,expert]!=-999
            scores[expert] = clf.score(data[has_label], labels[has_label][:,expert])
            test_predict[expert] = clf.predict(data[1].reshape(1, -1))[0]
        print("mean accuracy of marginal estimators")
        print(scores)
        print(np.mean(scores))
        print(test_predict)

    def get_eval_matrix(self, data, labels, model_name="trained"):
        print("Evaluating Gumbel-Max SI-SCM "+model_name)
        model = self.scm_model
        if model_name=="naive":
            model = self.scm_naive
        n_labels_per_row = np.sum(labels!=-999, axis=1)
        total_predictions = np.sum( n_labels_per_row * (n_labels_per_row-1))
        eval_matrix = np.zeros((total_predictions, 17))
        #print(eval_matrix.shape)
        current_ind = 0
        for obs_exp in range(self.n_experts):
            obs_group = self.scm_model.get_group_index(obs_exp)
            has_data = labels[:,obs_exp]!=-999
            #predictions = model.predict_counterfactuals(data[has_data], np.repeat(obs_exp,np.sum(has_data)), labels[has_data][:,obs_exp])
            proba = model.predict_cfc_proba(data[has_data], np.repeat(obs_exp,np.sum(has_data)), labels[has_data][:,obs_exp])
            predictions = np.argmax(proba, axis=2)
            data_indices = np.arange(data.shape[0], dtype=int)[has_data]
            for i, x in enumerate(data_indices):
                has_labels = labels[x]!=-999
                has_labels[obs_exp]= False
                exp_indices = np.arange(self.n_experts, dtype=int)[has_labels]
                for exp in exp_indices:
                    same_group = self.scm_model.get_group_index(exp) == obs_group
                    eval_matrix[current_ind] = np.array([x,obs_exp,exp,labels[x,obs_exp], labels[x,exp], predictions[i,exp], same_group ]+ proba[i,exp].tolist(), dtype=float)
                    current_ind +=1

        df_eval_matrix = pd.DataFrame(eval_matrix, columns = ["data_index", "obs_expert", "pred_expert", "obs_label","expert_label", "prediction", "is_same_group"] + ['proba_' + str(i) for i in range(10)])
        df_eval_matrix.to_csv("results_real/evaluation_results_"+model_name+".csv", index=False)
 
        print(model_name)
        print("Accuracy same: ",np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,4]==eval_matrix[:,3]))
        print("Accuracy diff: ",np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,4]!=eval_matrix[:,3]))
        print("Accuracy : ",np.mean(eval_matrix[:,4]==eval_matrix[:,5]))
        
    def fit_nb_baseline(self,expert_list, data, labels):
      #proba_func_list = []
      for expert in expert_list:
        X_train = data[labels[:, expert]!=-999]
        y_train_old = labels[labels[:,expert]!=-999][:,expert]
        y_train = labels[labels[:,expert]!=-999]

        n_labels_per_row = np.sum(y_train!=-999, axis=1)-1
        total_training_points = np.sum( n_labels_per_row )
        enc_X_train = np.zeros((total_training_points, X_train.shape[1]), dtype=float)
        #enc_X_train_obs = np.full((total_training_points, self.n_experts), 0, dtype=int)
        enc_X_train_obs = np.full((total_training_points, self.n_experts), 10, dtype=int)
        enc_y_train = np.zeros((total_training_points), dtype=int)
        current_ind = 0
        for obs_exp in range(self.n_experts):
            if obs_exp != expert:
                has_data = y_train[:,obs_exp]!=-999
                for i,x in enumerate(X_train[has_data]):
                    enc = np.full(self.n_experts,10)
                    #enc = np.full(self.n_experts,0)
                    enc[obs_exp] = y_train[has_data][i,obs_exp]#+1
                    enc_X_train[current_ind] = x
                    enc_X_train_obs[current_ind] = enc.astype(int)
                    enc_y_train[current_ind] = y_train[has_data][i,expert]
                    current_ind +=1
        
        X_train = enc_X_train
        X_train_cat = enc_X_train_obs
        y_train = enc_y_train

        gnb = GaussianNB()
        catnb = CategoricalNB(min_categories=np.repeat(11,self.n_experts))

        gnb.fit(X_train, y_train)
        catnb.fit(X_train_cat, y_train)
        #self.nb_baseline.append((gnb,catnb))
        self.nb_baseline.append((self.proba_models[expert],catnb))
 
    def get_eval_matrix_nb_baseline(self, data, labels):
        
        print("Evaluating GNB+CNB Baseline")
        n_labels_per_row = np.sum(labels!=-999, axis=1)
        total_predictions = np.sum( n_labels_per_row * (n_labels_per_row-1))
        eval_matrix = np.zeros((total_predictions, 7))
        #print(eval_matrix.shape)
        current_ind = 0
        for exp in range(self.n_experts):
            exp_group = self.scm_model.get_group_index(exp)
            has_data = labels[:,exp]!=-999
            gnb, catnb = self.nb_baseline[exp]
            data_indices = np.arange(data.shape[0], dtype=int)[has_data]
            for x in data_indices:
                has_labels = labels[x]!=-999
                has_labels[exp] = False
                obs_indices = np.arange(self.n_experts, dtype=int)[has_labels]
                for obs_exp in obs_indices:
                        same_group = self.scm_model.get_group_index(obs_exp) == exp_group
                        has_data = labels[:,obs_exp]!=-999
                        #feat_enc = np.full(self.n_experts, 0)
                        feat_enc = np.full(self.n_experts, 10)
                        feat_enc[obs_exp] = labels[x,obs_exp]#+1
                        X = np.expand_dims(data[x],axis=0)
                        X_cat = np.expand_dims(feat_enc,axis=0)
                        proba = gnb.predict_proba(X) * catnb.predict_proba(X_cat) / gnb.class_prior_
                        prediction = np.argmax(proba)
                        eval_matrix[current_ind] = np.array([x,obs_exp,exp,labels[x,obs_exp], labels[x,exp], prediction, same_group ], dtype = int)
                        #eval_matrix[current_ind] = np.array([x,obs_exp,exp,labels[x,obs_exp], labels[x,exp], prediction, 0 ], dtype = int)
                        current_ind +=1

        df_eval_matrix = pd.DataFrame(eval_matrix, columns = ["data_index", "obs_expert", "pred_expert", "obs_label","expert_label", "prediction", "is_same_group"])
        df_eval_matrix.to_csv("results_real/evaluation_results_nb_baseline.csv", index=False)
        
        print("Accuracy same: ",np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,4]==eval_matrix[:,3]))
        print("Accuracy diff: ",np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,4]!=eval_matrix[:,3]))
        print("Accuracy : ",np.mean(eval_matrix[:,4]==eval_matrix[:,5]))
       
    def get_eval_matrix_base_model(self, data, labels):
        print("Evaluating GNB Base model")
        n_labels_per_row = np.sum(labels!=-999, axis=1)
        total_predictions = np.sum( n_labels_per_row)
        eval_matrix = np.zeros((total_predictions, 7))
        #print(eval_matrix.shape)
        current_ind = 0
        for exp in range(self.n_experts):
            has_data = labels[:,exp]!=-999
            data_indices = np.arange(data.shape[0], dtype=int)[has_data]
            predictions = self.proba_models[exp].predict(data[has_data])
            for i, x in enumerate(data_indices):
                eval_matrix[current_ind] = np.array([x,-1,exp,-1, labels[x,exp], predictions[i], -1 ], dtype = int)
                current_ind +=1

        df_eval_matrix = pd.DataFrame(eval_matrix, columns = ["data_index", "obs_expert", "pred_expert", "obs_label","expert_label", "prediction", "is_same_group"])
        df_eval_matrix.to_csv("results_real/evaluation_results_base_model.csv", index=False)

        print("Accuracy : ",np.mean(eval_matrix[:,4]==eval_matrix[:,5]))



def main():
    n_classes = 10
    seed = 44
    
    data = pd.read_csv('data/data_training.csv').to_numpy()
    data_test = pd.read_csv('data/data_test.csv').to_numpy()
    labels = pd.read_csv('data/labels_training.csv').to_numpy(dtype = 'int')
    labels_test = pd.read_csv('data/labels_test.csv').to_numpy(dtype = 'int')

    #print(data)
    #print(labels)
    print("Training data shape: ", data.shape)
    print("Labels shape: ", labels.shape)
       
    n_experts = labels.shape[1]

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    num_features=20
    pca = PCA(n_components=num_features)
    sc = StandardScaler()

    data = sc.fit_transform(data)
    data = pca.fit_transform(data)
    data_test = sc.transform(data_test)
    data_test = pca.transform(data_test)
     
    exp = RealExperiment(n_classes, n_experts, seed)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data_test = scaler.transform(data_test)

    exp.add_experts(range(n_experts), data, labels)
    #exp.evaluate_marginal_estimators(data_test, labels_test)
    print("Evaluating GNB")
    exp.get_eval_matrix_base_model( data_test, labels_test)
    exp.update_model(data, labels)

    row_idx_test = np.sum(labels_test!=-999, axis=1)>1
    data_test = data_test[row_idx_test]
    labels_test = labels_test[row_idx_test]
    print("Test data shape: ", data_test.shape)
    print("Evaluating Gumbel-Max SI-SCM")
    exp.get_eval_matrix( data_test, labels_test)
    print("Evaluating M(H)")
    exp.get_eval_matrix( data_test, labels_test, model_name="naive")

    exp.fit_nb_baseline( range(n_experts),data, labels)
    exp.get_eval_matrix_nb_baseline( data_test, labels_test)



if __name__ == "__main__":
    main()

