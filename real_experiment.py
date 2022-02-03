#!/usr/bin/env python

import numpy as np
import pandas as pd
from scm import SCM
from helper import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

class RealExperiment:
    def __init__(self, n_classes, n_experts, seed=44):
        self.rng = np.random.default_rng(seed)
        self.n_classes = n_classes
        #self.n_features = n_features
        self.n_experts = 0

        self.proba_models = []
        self.list_proba_func = []

        self.scm_naive = None
        self.scm_model = None
    
    def add_experts(self, expert_list, data, labels):
        n_experts = self.n_experts
        new_proba_func_list = self.fit_proba_models(expert_list, data, labels)
        self.n_experts += len(expert_list)
        self.list_proba_func.extend(new_proba_func_list)
        #return index of added experts
        return range(n_experts, n_experts+len(expert_list))

    def get_proba(self, x, clf):
        classes_not_trained = set(clf.classes_).symmetric_difference(range(self.n_classes))
        #print(classes_not_trained)
        prob = clf.predict_proba(np.expand_dims(x, axis=0))
        #print(prob)
        if len(classes_not_trained)>0:
            new_prob = np.empty((self.n_classes))
            new_prob[list(classes_not_trained)]= 0.0000001
            new_prob[clf.classes_] = prob[0]
            return new_prob
            """ 
            new_prob = []
            prob_per_class = {c: prob[0,ind] for ind, c in enumerate(clf.classes_)}
            prob_class_not_trained = {c: 0.0000001 for c in classes_not_trained}
            # put the probabilities in class order
            prob_per_class.update(prob_class_not_trained)
            new_prob = np.hstack([p for k, p in sorted(prob_per_class.items())])
            #print(new_prob)
            return new_prob.astype(float)
            """
        
        return np.squeeze(prob, axis=0)

    def fit_proba_models(self,expert_list, data, labels):
      proba_func_list = []
      for expert in expert_list:
        X_train = data[labels[:, expert]!=-999]
        y_train = labels[labels[:,expert]!=-999][:,expert]

        lr = LogisticRegression(C=1.0)
        gnb = GaussianNB()
        gnb_isotonic = CalibratedClassifierCV(gnb, cv=4, method="isotonic")
        gnb_sigmoid = CalibratedClassifierCV(gnb, cv=4, method="sigmoid")

        clf_list = [
            (lr, "Logistic"),
            #(gnb, "Naive Bayes"),
            #(gnb_isotonic, "Naive Bayes + Isotonic"),
            #(gnb_sigmoid, "Naive Bayes + Sigmoid"),
        ]
        proba_function = None
        for i, (clf, name) in enumerate(clf_list):
            clf.fit(X_train, y_train)
            self.proba_models.append(clf)
            #if expert==175: #only 9 classes in labels of expert 175
            #    print("prob for expert 175")
            #    print(self.get_proba(data[0], clf))
            #    print(labels[0])
    
        #return [(lambda x, clf=clf: np.squeeze(clf.predict_proba(np.expand_dims(x, axis=0)), axis=0)) for clf in self.proba_models]
      return [(lambda x, clf=clf: self.get_proba(x, clf)) for clf in self.proba_models]
       
    def update_model(self, data_train, labels_train):
        #our model
        if self.scm_model == None:
            self.scm_model = SCM("Trained", self.n_classes, self.list_proba_func)
            self.scm_model.fit( data_train, labels_train)
        
        #naive
        if self.scm_naive == None:
            self.scm_naive = SCM("Naive", self.n_classes, self.list_proba_func, naive= True)

    def evaluate_experiment(self, data_test, labels_test):
        N_test = data_test.shape[0]
        test_inds = [self.rng.choice(a=np.flatnonzero(labels_test[x]!=-999)) for x in range(N_test)]
        # score of counterfactual labels for the real group of the observed expert and non cf. labels for remaining experts in the trained scm group of the observed expert
        print("Evaluating Gumbel CSCM...")
        scores_trained = self.scm_model.score_counterfactuals_rand(data_test, labels_test, test_inds, labels_test[range(N_test),test_inds])
        print("Gumbel-Max CSCM: ", scores_trained)

        print("Evaluating Naive CSCM...")
        scores_naive = self.scm_naive.score_counterfactuals_rand(data_test, labels_test, test_inds, labels_test[range(N_test),test_inds])
        print("Naive: ", scores_naive)

    def evaluate_experiment_all(self, data_test, labels_test):
        print("Score of predicting all experts with labels")
        N_test = data_test.shape[0]
        test_inds = [self.rng.choice(a=np.flatnonzero(labels_test[x]!=-999)) for x in range(N_test)]
        # score of counterfactual labels for the real group of the observed expert and non cf. labels for remaining experts in the trained scm group of the observed expert
        print("Evaluating Gumbel CSCM...")
        scores_trained = self.scm_model.score_counterfactuals(data_test, labels_test, test_inds, labels_test[range(N_test),test_inds])
        print("Gumbel-Max CSCM: ", scores_trained)

        print("Evaluating Naive CSCM...")
        scores_naive = self.scm_naive.score_counterfactuals(data_test, labels_test, test_inds, labels_test[range(N_test),test_inds])
        print("Naive: ", scores_naive)

    def evaluate_experiment_seen(self, data_test, labels_test, labels_seen):
        #filter out experts pairs without any data with label prediction overlap
        def check_label_overlap(labels):
            has_no_overlap = np.empty((self.n_experts, self.n_experts))
            for ind in range(self.n_experts):
                n_overlaps = np.count_nonzero(labels[labels[np.arange(labels.shape[0]), ind]!=-999]!=-999, axis=0)
                has_no_overlap[ind] = n_overlaps==0
            print(has_no_overlap)
            return has_no_overlap.astype(int)

        has_no_overlap = check_label_overlap(labels_test)
        N_test = data_test.shape[0]

        print("Score of predicting all experts with labels")
        obs_inds = [self.rng.choice(a=np.flatnonzero(labels_test[x]!=-999)) for x in range(N_test)]
        obs_labels =labels_test[range(N_test),obs_inds]
        labels_test[ has_no_overlap[obs_inds]] = -999
        print(labels_test)
        print(obs_labels)
        # score of counterfactual labels for the real group of the observed expert and non cf. labels for remaining experts in the trained scm group of the observed expert
        print("Evaluating Gumbel CSCM...")
        scores_trained = self.scm_model.score_counterfactuals(data_test, labels_test, obs_inds, obs_labels)
        print("Gumbel-Max CSCM: ", scores_trained)

        print("Evaluating Naive CSCM...")
        scores_naive = self.scm_naive.score_counterfactuals(data_test, labels_test, obs_inds, obs_labels)
        print("Naive: ", scores_naive)

    def evaluate_experiment_seen_top_k(self, data_test, labels_test, labels_seen):
        #filter out experts pairs without any data with label prediction overlap
        def check_label_overlap(labels):
            has_no_overlap = np.empty((self.n_experts, self.n_experts))
            for ind in range(self.n_experts):
                n_overlaps = np.count_nonzero(labels[labels[np.arange(labels.shape[0]), ind]!=-999]!=-999, axis=0)
                has_no_overlap[ind] = n_overlaps==0
            print(has_no_overlap)
            return has_no_overlap.astype(int)

        has_no_overlap = check_label_overlap(labels_test)
        N_test = data_test.shape[0]

        print("Score of predicting all experts with labels")
        obs_inds = [self.rng.choice(a=np.flatnonzero(labels_test[x]!=-999)) for x in range(N_test)]
        obs_labels =labels_test[range(N_test),obs_inds]
        labels_test[ has_no_overlap[obs_inds]] = -999
        print(labels_test)
        print(obs_labels)
        # score of counterfactual labels for the real group of the observed expert and non cf. labels for remaining experts in the trained scm group of the observed expert
        print("Evaluating Gumbel CSCM...")
        scores_trained = self.scm_model.score_counterfactuals_top_k(2, data_test, labels_test, obs_inds, obs_labels)
        print("Gumbel-Max CSCM: ", scores_trained)

        print("Evaluating Naive CSCM...")
        scores_naive = self.scm_naive.score_counterfactuals_top_k(2, data_test, labels_test, obs_inds, obs_labels)
        print("Naive: ", scores_naive)

        out = np.full_like(labels, -999)
        same_pred = np.repeat(np.expand_dims(obs_labels, axis=1),self.n_experts,axis=1)
        loss_same_pred = np.not_equal(same_pred, labels, out=out, where= labels!=-999)
        print("Mean loss for same pred estimation: ", np.mean(loss_same_pred, where= labels!=-999))

    def evaluate_experiment_top_k(self, data_test, labels_test, labels_seen):
        N_test = data_test.shape[0]

        print("Score of predicting all experts with labels")
        obs_inds = [self.rng.choice(a=np.flatnonzero(labels_test[x]!=-999)) for x in range(N_test)]
        obs_labels =labels_test[range(N_test),obs_inds]
        #print(labels_test)
        #print(obs_labels)
        # score of counterfactual labels for the real group of the observed expert and non cf. labels for remaining experts in the trained scm group of the observed expert
        print("Evaluating Gumbel CSCM...")
        scores_trained = self.scm_model.score_counterfactuals_top_k(2, data_test, labels_test, obs_inds, obs_labels)
        print("Gumbel-Max CSCM top 2 accuracy: ", scores_trained)

        print("Evaluating Naive CSCM...")
        scores_naive = self.scm_naive.score_counterfactuals_top_k(2, data_test, labels_test, obs_inds, obs_labels)
        print("Naive top 2 accuracy: ", scores_naive)

        out = np.full_like(labels_test, -999)
        same_pred = np.repeat(np.expand_dims(obs_labels, axis=1),self.n_experts,axis=1)
        loss_same_pred = np.not_equal(same_pred, labels_test, out=out, where= labels_test!=-999)
        print("Mean loss for same pred estimation: ", np.mean(loss_same_pred, where= labels_test!=-999))

def main():
    n_experts = 500
    n_classes = 10
    seed = 44
    """
    df = pd.read_csv('data/cifar10_feat+labels.csv').fillna(-999)
    data_all = df.filter(like='feature', axis=1).to_numpy()
    labels_all = df.filter(like='chosen_label', axis=1).to_numpy(dtype = 'int')
    for i in range(int(2571/n_experts)):
        min_index = (i*n_experts)
        max_index = min((i*n_experts)+n_experts, 2571)
        print("Modeling experts ", min_index, " to ", max_index)
        labels = labels_all[:, min_index:max_index]
        mask = np.any(labels!=-999, axis=1)
        labels = labels[mask]
        data = data_all[mask]
        print("Data shape: ", data.shape)
        print("Labels shape: ", labels.shape)
        data, data_test, labels, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        data_test = scaler.transform(data_test)

        exp = RealExperiment(n_classes, n_experts, seed)
        exp.add_experts(range(n_experts), data, labels)
        exp.update_model(data, labels)
        exp.evaluate_experiment( data_test, labels_test)
        exp.evaluate_experiment_seen_top_k( data_test, labels_test, labels)
        #exp.evaluate_experiment_seen( data_test, labels_test, labels)
    """
    data = pd.read_csv('data/data_training.csv').to_numpy()
    data_test = pd.read_csv('data/data_test.csv').to_numpy()
    labels = pd.read_csv('data/labels_training.csv').to_numpy(dtype = 'int')
    labels_test = pd.read_csv('data/labels_test.csv').to_numpy(dtype = 'int')

    #print(data)
    #print(labels)
    print("Data shape: ", data.shape)
    print("Labels shape: ", labels.shape)

    n_experts = labels.shape[1]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data_test = scaler.transform(data_test)

    exp = RealExperiment(n_classes, n_experts, seed)
    exp.add_experts(range(n_experts), data, labels)
    exp.update_model(data, labels)
    exp.evaluate_experiment( data_test, labels_test)
    exp.evaluate_experiment_top_k( data_test, labels_test, labels)
        
if __name__ == "__main__":
    main()

#to do:
#pre_process labels:
#choose 200 experts at random, have a set of pic_idx that have been labeled
#group: data matrix: #pic_idx x #features, labels: #pic_idx x 200

#Here:
#log regression, and calibrated naive gaussian (+calibration curve)
#call scm
#evaluation plots

#Possible Problem:
#check if there are even different guesses, most seem to be correct 
