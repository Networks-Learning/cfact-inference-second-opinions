#!/usr/bin/env python

import numpy as np
import pandas as pd
from siscm import SISCM
from helper import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB


np.random.seed(312253)
np.set_printoptions(precision=3)

class RealExperiment:
    #class constructor
    def __init__(self, n_classes, n_experts, seed=44):
        self.rng = np.random.default_rng(seed)
        self.n_classes = n_classes
        self.n_experts = n_experts

        self.proba_models = []
        self.nb_baseline = []
        self.list_marginal_proba_func = []

        self.siscm_H = None
        self.siscm_Psi = None
        
    #add experts to the experiment by learning their marginal distribution models with the given data    
    def add_experts(self, expert_list, data, labels):
        new_proba_func_list = self.fit_proba_models(expert_list, data, labels)
        self.list_marginal_proba_func.extend(new_proba_func_list)
        self.n_experts = len(self.list_marginal_proba_func)

    #returns marginal probabilities of an expert's prediction for given features x
    #Note, if class observations were missing during training, this function returns a very small probability for these classes,
    def get_proba(self, x, clf):
        classes_not_trained = set(clf.classes_).symmetric_difference(range(self.n_classes))
        #print(classes_not_trained)
        proba = clf.predict_proba(np.expand_dims(x, axis=0))
        #print(proba)
        if len(classes_not_trained)>0:
            new_proba = np.empty((self.n_classes))
            new_proba[list(classes_not_trained)] =  0.00000001
            new_proba[clf.classes_] = proba[0]
            new_proba[new_proba==0.0] = 0.00000001
            return new_proba
            
        proba = np.squeeze(proba, axis=0)
        return np.where( proba > 0.0, proba,  0.00000001)

    #fit marginal distribution models, GNB, for each expert
    #returns array of distribution functions
    def fit_proba_models(self,expert_list, data, labels):
      for expert in expert_list:
        X_train = data[labels[:, expert]!=-999]
        y_train = labels[labels[:,expert]!=-999][:,expert]

        gnb = GaussianNB()

        gnb.fit(X_train, y_train)
        self.proba_models.append(gnb)

      return [(lambda x, clf=clf: self.get_proba(x, clf)) for clf in self.proba_models]

    #create and fit SISCM model M(Psi) and create M(H) model
    def update_model(self, data_train, labels_train):
        #create and fit SI-SCM M(Psi)
        if self.siscm_Psi == None:
            self.siscm_Psi = SISCM("SISCM_M(Psi)", self.n_classes, self.list_marginal_proba_func)
            self.siscm_Psi.fit( data_train, labels_train, val_ratio=0.0, max_rounds=10)
        
        #create SI-SCM M(H)
        if self.siscm_H == None:
            self.siscm_H = SISCM("SISCM_M(H)", self.n_classes, self.list_marginal_proba_func, siscm_H= True)

    #Evaluates the models estimating the marginal distibution of each expert
    #Prints mean accuracy per expert and in total
    def evaluate_marginal_estimators(self, data, labels):
        scores = np.zeros((self.n_experts))
        #test_predict = np.zeros((self.n_experts))
        for expert, clf in enumerate(self.proba_models):
            has_label=labels[:,expert]!=-999
            scores[expert] = clf.score(data[has_label], labels[has_label][:,expert])
            #test_predict[expert] = clf.predict(data[1].reshape(1, -1))[0]
        print("mean accuracy of marginal estimators")
        print(scores)
        print(np.mean(scores))
        #print(test_predict)

    #evaluate SI-SCM model on data 
    def evaluate_siscm(self, data, labels, model_name="SISCM_M(Psi)"):
        #pick correct SI-SCM
        model = self.siscm_Psi
        if model_name=="SISCM_M(H)":
            model = self.siscm_H
        print("Evaluating Gumbel-Max SI-SCM "+ model_name)

        #counterfactual inference of an expert's prediction using another expert's observed prediction
        n_labels_per_row = np.sum(labels!=-999, axis=1)
        total_predictions = np.sum( n_labels_per_row * (n_labels_per_row-1))
        eval_matrix = np.zeros((total_predictions, 17))
        current_ind = 0
        for obs_exp in range(self.n_experts):
            obs_group = self.siscm_Psi.get_group_index(obs_exp)
            has_data = labels[:,obs_exp]!=-999
            proba = model.predict_cfc_proba(data[has_data], np.repeat(obs_exp,np.sum(has_data)), labels[has_data][:,obs_exp])
            predictions = np.argmax(proba, axis=2)
            data_indices = np.arange(data.shape[0], dtype=int)[has_data]
            for i, x in enumerate(data_indices):
                has_labels = labels[x]!=-999
                has_labels[obs_exp]= False
                exp_indices = np.arange(self.n_experts, dtype=int)[has_labels]
                for exp in exp_indices:
                    same_group = self.siscm_Psi.get_group_index(exp) == obs_group
                    #set evaluation matrix entry with meta data and counterfactual inference result
                    eval_matrix[current_ind] = np.array([x,obs_exp,exp,labels[x,obs_exp], labels[x,exp], predictions[i,exp], same_group ]+ proba[i,exp].tolist(), dtype=float)
                    current_ind +=1

        #save evaluation results
        df_eval_matrix = pd.DataFrame(eval_matrix, columns = ["data_index", "obs_expert", "pred_expert", "obs_label","expert_label", "prediction", "is_same_group"] + ['proba_' + str(i) for i in range(10)])
        df_eval_matrix.to_csv("results_real/evaluation_results_"+model_name+".csv", index=False)
 
        #print overall mean
        print(model_name)
        print("Accuracy : ",np.mean(eval_matrix[:,4]==eval_matrix[:,5]))
        
    #create and fit GNB+CNB baseline
    def fit_nb_baseline(self, expert_list, data, labels):
      for expert in expert_list:
        X_train = data[labels[:, expert]!=-999]
        y_train = labels[labels[:,expert]!=-999]

        n_labels_per_row = np.sum(y_train!=-999, axis=1)-1
        total_training_points = np.sum( n_labels_per_row )
        img_features = np.zeros((total_training_points, X_train.shape[1]), dtype=float)
        #one-hot encoding of observation
        #zero-based: label 0 stands for unobserved, shifts all labels +1
        #enc_X_train_obs = np.full((total_training_points, self.n_experts), 0, dtype=int)
        #10-based: label 10 stands for unobserved
        observed_pred = np.full((total_training_points, self.n_experts), 10, dtype=int)
        new_y_train = np.zeros((total_training_points), dtype=int)
        current_ind = 0
        for obs_expert in range(self.n_experts):
            if obs_expert != expert:
                has_data = y_train[:,obs_expert]!=-999
                for i,x in enumerate(X_train[has_data]):
                    enc = np.full(self.n_experts,10)
                    #enc = np.full(self.n_experts,0)
                    enc[obs_expert] = y_train[has_data][i,obs_expert]#+1
                    img_features[current_ind] = x
                    observed_pred[current_ind] = enc.astype(int)
                    new_y_train[current_ind] = y_train[has_data][i,expert]
                    current_ind +=1
        
        X_train_gnb = img_features
        X_train_cnb = observed_pred
        y_train = new_y_train

        #fit CNB given observed predictions
        catnb = CategoricalNB(min_categories=np.repeat(11,self.n_experts))
        catnb.fit(X_train_cnb, y_train)

        #Uncomment to train and use new GNB with the images features
        #gnb = GaussianNB()
        #gnb.fit(X_train_gnb, y_train)
        #self.nb_baseline.append((gnb,catnb))
        
        #Saving the previously learned, marginal distribution GNB models and CNB as baseline
        self.nb_baseline.append((self.proba_models[expert],catnb))

    #evaluate GNB+CNB baseline model on data 
    def evaluate_nb_baseline(self, data, labels):
        
        print("Evaluating GNB+CNB Baseline")
        n_labels_per_row = np.sum(labels!=-999, axis=1)
        total_predictions = np.sum( n_labels_per_row * (n_labels_per_row-1))
        eval_matrix = np.zeros((total_predictions, 7))
        #print(eval_matrix.shape)
        current_ind = 0
        for expert in range(self.n_experts):
            expert_group = self.siscm_Psi.get_group_index(expert)
            has_data = labels[:,expert]!=-999
            gnb, catnb = self.nb_baseline[expert]
            data_indices = np.arange(data.shape[0], dtype=int)[has_data]
            for x in data_indices:
                has_labels = labels[x]!=-999
                has_labels[expert] = False
                obs_indices = np.arange(self.n_experts, dtype=int)[has_labels]
                for obs_expert in obs_indices:
                        same_group = self.siscm_Psi.get_group_index(obs_expert) == expert_group
                        has_data = labels[:,obs_expert]!=-999
                        #feat_enc = np.full(self.n_experts, 0)
                        feat_enc = np.full(self.n_experts, 10)
                        feat_enc[obs_expert] = labels[x,obs_expert]#+1
                        X_gnb = np.expand_dims(data[x],axis=0)
                        X_catnb = np.expand_dims(feat_enc,axis=0)
                        #combine marginal probabilities of both NB models
                        proba = gnb.predict_proba(X_gnb) * catnb.predict_proba(X_catnb) / gnb.class_prior_
                        #predict likeliest label
                        prediction = np.argmax(proba)
                        #set evaluation matrix entry with meta data and baseline prediction
                        eval_matrix[current_ind] = np.array([x, obs_expert, expert, labels[x,obs_expert], labels[x,expert], prediction, same_group ], dtype = int)
                        current_ind +=1

        #save evaluation results
        df_eval_matrix = pd.DataFrame(eval_matrix, columns = ["data_index", "obs_expert", "pred_expert", "obs_label","expert_label", "prediction", "is_same_group"])
        df_eval_matrix.to_csv("results_real/evaluation_results_nb_baseline.csv", index=False)
        
        #print overall mean
        print("Accuracy : ", np.mean(eval_matrix[:,4]==eval_matrix[:,5]))

    #evaluate GNB model on data 
    def evaluate_proba_models(self, data, labels):
        print("Evaluating GNB marginal probability models")
        n_labels_per_row = np.sum(labels!=-999, axis=1)
        total_predictions = np.sum( n_labels_per_row)
        eval_matrix = np.zeros((total_predictions, 7))
        #print(eval_matrix.shape)
        current_ind = 0
        for expert in range(self.n_experts):
            has_data = labels[:,expert]!=-999
            data_indices = np.arange(data.shape[0], dtype=int)[has_data]
            predictions = self.proba_models[expert].predict(data[has_data])
            for i, x in enumerate(data_indices):
                #set evaluation matrix entry with meta data and prediction
                eval_matrix[current_ind] = np.array([x,-1,expert,-1, labels[x,expert], predictions[i], -1 ], dtype = int)
                current_ind +=1

        #save evaluation results
        df_eval_matrix = pd.DataFrame(eval_matrix, columns = ["data_index", "obs_expert", "pred_expert", "obs_label","expert_label", "prediction", "is_same_group"])
        df_eval_matrix.to_csv("results_real/evaluation_results_proba_models.csv", index=False)

        #print overall mean
        print("Accuracy : ",np.mean(eval_matrix[:,4]==eval_matrix[:,5]))

    #save set Psi of expert groups found by fit algorithm to file
    def save_groups(self):
        groups = { i: g for i,g in enumerate(self.siscm_Psi.group_members_sorted)}
        groups = { i: g for i,g in enumerate(groups)}
        df_groups = pd.DataFrame.from_dict(groups, orient='index')
        df_groups.to_csv('results_real/SI-SCM_groups.csv',index=False)


def main():
    n_classes = 10
    seed = 44
    
    #read data
    data = pd.read_csv('data/data_training.csv').to_numpy()
    data_test = pd.read_csv('data/data_test.csv').to_numpy()
    labels = pd.read_csv('data/labels_training.csv').to_numpy(dtype = 'int')
    labels_test = pd.read_csv('data/labels_test.csv').to_numpy(dtype = 'int')

    #print data size
    print("Training data shape: ", data.shape)
    print("Labels shape: ", labels.shape)
    #number of experts in data
    n_experts = labels.shape[1]

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    #preprocess the image features using PCA and scalers
    num_features=20
    pca = PCA(n_components=num_features)
    sc = StandardScaler()

    data = sc.fit_transform(data)
    data = pca.fit_transform(data)
    data_test = sc.transform(data_test)
    data_test = pca.transform(data_test)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data_test = scaler.transform(data_test)

    #create experiment setup
    exp = RealExperiment(n_classes, n_experts, seed)
    exp.add_experts(range(n_experts), data, labels)
    #exp.evaluate_marginal_estimators(data_test, labels_test)
    exp.update_model(data, labels)
    exp.save_groups()

    #evaluate models on the test data
    print("Test data shape: ", data_test.shape)
    exp.evaluate_siscm( data_test, labels_test)
    exp.evaluate_siscm( data_test, labels_test, model_name="SISCM_M(H)")
    exp.evaluate_proba_models( data_test, labels_test)

    #fit and evaluate baseline
    exp.fit_nb_baseline( range(n_experts),data, labels)
    exp.evaluate_nb_baseline( data_test, labels_test)


if __name__ == "__main__":
    main()

