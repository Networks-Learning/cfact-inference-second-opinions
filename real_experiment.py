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

np.set_printoptions(precision=3)
class RealExperiment:
    def __init__(self, n_classes, n_experts, seed=44):
        self.rng = np.random.default_rng(seed)
        self.n_classes = n_classes
        #self.n_features = n_features
        self.n_experts = n_experts

        self.proba_models = []
        self.logreg_baseline = []
        self.nb_baseline = []
        self.minmaxsc_models = []
        self.list_proba_func = []

        self.scm_naive = None
        self.scm_model = None
    
    def add_experts(self, expert_list, data, labels):
        n_experts = self.n_experts
        new_proba_func_list = self.fit_proba_models(expert_list, data, labels)
        self.list_proba_func.extend(new_proba_func_list)
        self.n_experts = len(self.list_proba_func)
        #return index of added experts
        return range(n_experts, n_experts+len(expert_list))

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
        prob = np.squeeze(prob, axis=0)
        return np.where( prob > 0.0, prob,  0.00000001)

    def fit_proba_models(self,expert_list, data, labels):
      #proba_func_list = []
      for expert in expert_list:
        X_train = data[labels[:, expert]!=-999]
        y_train = labels[labels[:,expert]!=-999][:,expert]

        lr = LogisticRegression(random_state=33,  max_iter=500,class_weight='balanced', penalty='l1', solver='saga')
        gnb = GaussianNB()
        #gnb = GaussianNB(priors = np.repeat(0.1,10))
        gnb_isotonic = CalibratedClassifierCV(gnb, cv=2, method="isotonic")
        gnb_sigmoid = CalibratedClassifierCV(gnb, cv=2, method="sigmoid")

        clf_list = [
            #(lr, "Logistic"),
            (gnb, "Naive Bayes"),
            #(gnb_isotonic, "Naive Bayes + Isotonic"),
            #(gnb_sigmoid, "Naive Bayes + Sigmoid"),
        ]
        #proba_function = None
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

        has_no_overlap = check_label_overlap(labels_seen)
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

        has_no_overlap = check_label_overlap(labels_seen)
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

        out = np.full_like(labels_test, -999)
        same_pred = np.repeat(np.expand_dims(obs_labels, axis=1),self.n_experts,axis=1)
        loss_same_pred = np.not_equal(same_pred, labels_test, out=out, where= labels_test!=-999)
        print("Mean loss for same pred estimation: ", np.mean(loss_same_pred, where= labels_test!=-999))

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

    def get_eval_matrix(self, data, labels, model_name="trained"):
        model = self.scm_model
        if model_name=="naive":
            model = self.scm_naive
        print(np.sum(labels!=-999, axis=1))
        n_labels_per_row = np.sum(labels!=-999, axis=1)
        print(n_labels_per_row)
        total_predictions = np.sum( n_labels_per_row * (n_labels_per_row-1))
        eval_matrix = np.zeros((total_predictions, 17))
        print(eval_matrix.shape)
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
        df_eval_matrix.to_csv("data/evaluation_results_"+model_name+".csv", index=False)
 
        print(model_name)
        print("Accuracy same: ",np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,4]==eval_matrix[:,3]))
        print("Accuracy diff: ",np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,4]!=eval_matrix[:,3]))
        print("Accuracy : ",np.mean(eval_matrix[:,4]==eval_matrix[:,5]))
        print("Accuracy per expert: ")
        acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,2]==exp) for exp in range(self.n_experts)])
        print(acc)
        print(np.nanmean(acc))
        print("Accuracy per expert for group obs: ")
        acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,2]==exp) & (eval_matrix[:,6]==1)) for exp in range(self.n_experts)])
        print(acc)
        print(np.nanmean(acc))

        print("Accuracy diff per expert for group obs: ")
        acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,4]!=eval_matrix[:,3]) & (eval_matrix[:,2]==exp) & (eval_matrix[:,6]==1)) for exp in range(self.n_experts)])
        print(acc)
        print(np.nanmean(acc))
        print("Accuracy per expert for non group obs: ")
        acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,2]==exp) & (eval_matrix[:,6]==0)) for exp in range(self.n_experts)])
        print(acc)
        print(np.nanmean(acc))
        #print("Accuracy per obs expert for group prediction: ")
        #print(np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,1]==exp) & (eval_matrix[:,6]==1)) for exp in range(self.n_experts)]))
  


    def fit_nb_baseline(self,expert_list, data, labels):
      #proba_func_list = []
      for expert in expert_list:
        X_train = data[labels[:, expert]!=-999]
        y_train = labels[labels[:,expert]!=-999]

        n_labels_per_row = np.sum(y_train!=-999, axis=1)-1
        total_training_points = np.sum( n_labels_per_row )
        #enc_X_train = np.zeros((total_training_points, X_train.shape[1]+self.n_experts), dtype=float)
        enc_X_train = np.zeros((total_training_points, X_train.shape[1]), dtype=float)
        enc_X_train_obs = np.full((total_training_points, self.n_experts), 0, dtype=int)
        #enc_X_train_obs = np.full((total_training_points, self.n_experts), 10, dtype=int)
        enc_y_train = np.zeros((total_training_points), dtype=float)
        current_ind = 0
        for obs_exp in range(self.n_experts):
            if obs_exp != expert:
                has_data = y_train[:,obs_exp]!=-999
                for i,x in enumerate(X_train[has_data]):
                    #enc = np.full(self.n_experts,10)
                    enc = np.full(self.n_experts,0)
                    enc[obs_exp] = y_train[has_data][i,obs_exp]+1
                    #enc_X_train[current_ind] = np.concatenate((x,enc.astype(float)), axis=None)
                    enc_X_train[current_ind] = x
                    enc_X_train_obs[current_ind] = enc.astype(int)
                    enc_y_train[current_ind] = y_train[has_data][i,expert]
                    current_ind +=1
        
        X_train = enc_X_train
        X_train_cat = enc_X_train_obs
        y_train = enc_y_train

        gnb = GaussianNB()
        catnb = CategoricalNB(min_categories=np.repeat(11,self.n_experts))

        clf_list = [
            (gnb, "Naive Bayes"),
        ]
        for i, (clf, name) in enumerate(clf_list):
            clf.fit(X_train, y_train)
            catnb.fit(X_train_cat, y_train)
            self.nb_baseline.append((clf,catnb))
 
    def get_eval_matrix_nb_baseline(self, data, labels):
        
        n_labels_per_row = np.sum(labels!=-999, axis=1)
        total_predictions = np.sum( n_labels_per_row * (n_labels_per_row-1))
        eval_matrix = np.zeros((total_predictions, 7))
        print(eval_matrix.shape)
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
                        feat_enc = np.full(self.n_experts, 0)
                        #feat_enc = np.full(self.n_experts, 10)
                        feat_enc[obs_exp] = labels[x,obs_exp]+1
                        X = np.expand_dims(data[x],axis=0)
                        X_cat = np.expand_dims(feat_enc,axis=0)
                        proba = gnb.predict_proba(X) * catnb.predict_proba(X_cat) / gnb.class_prior_
                        prediction = np.argmax(proba, axis=1)
                        eval_matrix[current_ind] = np.array([x,obs_exp,exp,labels[x,obs_exp], labels[x,exp], prediction[0], same_group ], dtype = int)
                        current_ind +=1

        df_eval_matrix = pd.DataFrame(eval_matrix, columns = ["data_index", "obs_expert", "pred_expert", "obs_label","expert_label", "prediction", "is_same_group"])
        df_eval_matrix.to_csv("data/evaluation_results_nb_baseline.csv", index=False)
        
        print("Accuracy same: ",np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,4]==eval_matrix[:,3]))
        print("Accuracy diff: ",np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,4]!=eval_matrix[:,3]))
        print("Accuracy : ",np.mean(eval_matrix[:,4]==eval_matrix[:,5]))
        
        print("Accuracy per expert: ")
        acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,2]==exp) for exp in range(self.n_experts)])
        print(acc)
        print(np.nanmean(acc))
        print("Accuracy per expert in same group: ")
        acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,2]==exp) &(eval_matrix[:,6]==1) ) for exp in range(self.n_experts)])
        print(acc)
        print(np.nanmean(acc))

        print("Accuracy diff per expert in same group: ")
        acc=np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,4]!=eval_matrix[:,3]) & (eval_matrix[:,2]==exp) &(eval_matrix[:,6]==1) ) for exp in range(self.n_experts)])
        print(acc)
        print(np.nanmean(acc))
        #print("Accuracy per obs expert:")
        #print(np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,1]==exp) for exp in range(self.n_experts)]))
  

    def fit_logreg_baseline(self,expert_list, data, labels):
      #proba_func_list = []
      for expert in expert_list:
        X_train = data[labels[:, expert]!=-999]
        y_train = labels[labels[:,expert]!=-999]

        n_labels_per_row = np.sum(y_train!=-999, axis=1)-1
        total_training_points = np.sum( n_labels_per_row )
        enc_X_train = np.zeros((total_training_points, X_train.shape[1]), dtype=float)
        enc_X_train_obs = np.full((total_training_points, self.n_experts), 0, dtype=int)
        #enc_X_train_obs = np.full((total_training_points, self.n_experts), 10, dtype=int)
        enc_y_train = np.zeros((total_training_points), dtype=float)
        current_ind = 0
        for obs_exp in range(self.n_experts):
            if obs_exp != expert:
                has_data = y_train[:,obs_exp]!=-999
                for i,x in enumerate(X_train[has_data]):
                    #enc = np.full(self.n_experts,10)
                    enc = np.full(self.n_experts,0)
                    enc[obs_exp] = y_train[has_data][i,obs_exp]+1
                    #enc_X_train[current_ind] = np.concatenate((x,enc.astype(float)), axis=None)
                    enc_X_train[current_ind] = x
                    enc_X_train_obs[current_ind] = enc.astype(int)
                    enc_y_train[current_ind] = y_train[has_data][i,expert]
                    current_ind +=1
        
        X_train = np.hstack([enc_X_train, enc_X_train_obs])
        #self.minmaxsc_models.append(MinMaxScaler())
        #X_train = self.minmaxsc_models[expert].fit_transform(X_train)
        y_train = enc_y_train

        #lr = LogisticRegression(random_state=33, max_iter=2000, class_weight='balanced', penalty='elasticnet', solver='saga', l1_ratio= 1.0)
        lr = LogisticRegression(random_state=33, max_iter=1000, class_weight='balanced')

        lr.fit(X_train, y_train)
        self.logreg_baseline.append(lr)
 
    def get_eval_matrix_logreg_baseline(self, data, labels):
        
        n_labels_per_row = np.sum(labels!=-999, axis=1)
        total_predictions = np.sum( n_labels_per_row * (n_labels_per_row-1))
        eval_matrix = np.zeros((total_predictions, 7))
        print(eval_matrix.shape)
        current_ind = 0
        for exp in range(self.n_experts):
            exp_group = self.scm_model.get_group_index(exp)
            has_data = labels[:,exp]!=-999
            model = self.logreg_baseline[exp]
            data_indices = np.arange(data.shape[0], dtype=int)[has_data]
            for x in data_indices:
                has_labels = labels[x]!=-999
                has_labels[exp] = False
                obs_indices = np.arange(self.n_experts, dtype=int)[has_labels]
                for obs_exp in obs_indices:
                        same_group = self.scm_model.get_group_index(obs_exp) == exp_group
                        has_data = labels[:,obs_exp]!=-999
                        feat_enc = np.full(self.n_experts, 0)
                        #feat_enc = np.full(self.n_experts, 10)
                        feat_enc[obs_exp] = labels[x,obs_exp]+1
                        X = np.expand_dims(np.hstack([data[x],feat_enc]),axis=0)
                        #X = self.minmaxsc_models[exp].transform(X)
                        prediction = model.predict(X)
                        eval_matrix[current_ind] = np.array([x,obs_exp,exp,labels[x,obs_exp], labels[x,exp], prediction[0], same_group ], dtype = int)
                        current_ind +=1

        df_eval_matrix = pd.DataFrame(eval_matrix, columns = ["data_index", "obs_expert", "pred_expert", "obs_label","expert_label", "prediction", "is_same_group"])
        df_eval_matrix.to_csv("data/evaluation_results_logreg_baseline.csv", index=False)
        
        print("Accuracy same: ",np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,4]==eval_matrix[:,3]))
        print("Accuracy diff: ",np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,4]!=eval_matrix[:,3]))
        print("Accuracy : ",np.mean(eval_matrix[:,4]==eval_matrix[:,5]))
        
        print("Accuracy per expert: ")
        acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,2]==exp) for exp in range(self.n_experts)])
        print(acc)
        print(np.nanmean(acc))
        print("Accuracy per expert in same group: ")
        acc = np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,2]==exp) &(eval_matrix[:,6]==1) ) for exp in range(self.n_experts)])
        print(acc)
        print(np.nanmean(acc))

        print("Accuracy diff per expert in same group: ")
        acc=np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= (eval_matrix[:,4]!=eval_matrix[:,3]) & (eval_matrix[:,2]==exp) &(eval_matrix[:,6]==1) ) for exp in range(self.n_experts)])
        print(acc)
        print(np.nanmean(acc))
        #print("Accuracy per obs expert:")
        #print(np.array([np.mean(eval_matrix[:,4]==eval_matrix[:,5], where= eval_matrix[:,1]==exp) for exp in range(self.n_experts)]))
  

def main():
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
    """
    n_experts = labels.shape[1]
    has_data= np.zeros((n_experts,n_experts))
    for exp in range(n_experts):
        for obs in range(n_experts):
            has_data[exp,obs] = np.sum((labels[:,exp]!=-999) & (labels[:,obs]!=-999))
    
    print(np.sum(has_data<10, axis=1))
    exps = np.arange(n_experts, dtype=int)[has_data[4]<10]
    print(has_data[:,exps])

    G = nx.Graph(has_data<50)
    clique = list(max_clique(G))
    print(clique)

    row_idx = np.any(labels[:,clique]!=-999, axis=1)
    data = data[row_idx]
    labels = labels[row_idx][:,clique]
    print(data.shape)
    row_idx_test = np.any(labels_test[:,clique]!=-999, axis=1)
    data_test = data_test[row_idx_test]
    labels_test = labels_test[row_idx_test][:,clique]
    print(data_test.shape)


    """    
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
    #exp.fit_baseline_model( range(n_experts), data, labels)
    #exp.get_eval_matrix_baseline( data_test, labels_test)
    

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data_test = scaler.transform(data_test)

    exp.add_experts(range(n_experts), data, labels)
    exp.evaluate_marginal_estimators(data_test, labels_test)
    exp.update_model(data, labels)
    #exp.evaluate_experiment_top_k( data_test, labels_test, labels)
    #exp.evaluate_experiment_seen_top_k( data_test, labels_test, labels)

    row_idx_test = np.sum(labels_test!=-999, axis=1)>1
    data_test = data_test[row_idx_test]
    labels_test = labels_test[row_idx_test]
    print(data_test.shape)
    exp.get_eval_matrix( data_test, labels_test)
    exp.get_eval_matrix( data_test, labels_test, model_name="naive")

    exp.fit_logreg_baseline( range(n_experts),data, labels)
    exp.get_eval_matrix_logreg_baseline( data_test, labels_test)
    exp.fit_nb_baseline( range(n_experts),data, labels)
    exp.get_eval_matrix_nb_baseline( data_test, labels_test)
    #exp.evaluate_experiment( data_test, labels_test)



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
