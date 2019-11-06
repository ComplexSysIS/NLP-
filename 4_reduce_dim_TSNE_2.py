# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:35:41 2017

@author: PWOLFF
"""
import numpy as np
import numpy as Math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def make_vector_dic(vectors_filename):
    vector_dic = {}
    with open(vectors_filename,'r') as infile:
        for row in infile:
            row = row.rstrip('\n')
            row = row.rstrip('\t')
            line = row.split('\t')
            vector = []
            for i in range(1,len(line)):
                vector.append(float(line[i]))
            vector = vector / np.linalg.norm(vector)
            vector_dic[line[0]] = vector
    return vector_dic


def retrieve_important_words(keyword_file):
    importantwords = []
    with open(keyword_file,'r') as infile:
        for row in infile:
            row = row.rstrip('\n')
            importantwords.append(row) 
    return importantwords


def construct_matrix(importantwords,vector_dic):
    matrix_dic = {}
    for i in range(len(importantwords)):
        if importantwords[i] in vector_dic:
            matrix_dic[importantwords[i]] = vector_dic[importantwords[i]]
    return matrix_dic


def read_in_vectors_make_array(vec_dic_sample):
    row_list = []
    label_list = []
    for key in vec_dic_sample:
        row_list.append(vec_dic_sample[key])
        label_list.append(key)
    return np.array(row_list), label_list

def build_reduced_dim_matrix(Y,verbs,dimensionality):
    vec_dic_reduced = {}
    for i in range(len(Y)):
        vec_dic_reduced[verbs[i]] = Y[i]
    return vec_dic_reduced

#********************** TSNE BEGIN ********************************

def Hbeta(D = Math.array([]), beta = 1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""
    # Compute P-row and corresponding perplexity
    P = Math.exp(-D.copy() * beta);
    sumP = sum(P);
    if sumP < .00001:  #my addition to avoid errors
        sumP = .00001
    H = Math.log(sumP) + beta * Math.sum(D * P) / sumP;
    P = P / sumP;
    return H, P;

def x2p(X = Math.array([]), tol = 1e-5, perplexity = 30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""
    # Initialize some variables
    #print ("Computing pairwise distances...")
    (n, d) = X.shape;
    sum_X = Math.sum(Math.square(X), 1);
    D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X);
    P = Math.zeros((n, n));
    beta = Math.ones((n, 1));
    logU = Math.log(perplexity);
    
    # Loop over all datapoints
    for i in range(n):
        # Print progress
        if i % 500 == 0:
            print ("Computing P-values for point ", i, " of ", n, "...")
        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -Math.inf; 
        betamax =  Math.inf;
        Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))];
        (H, thisP) = Hbeta(Di, beta[i]);
        
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU;
        tries = 0;
        while Math.abs(Hdiff) > tol and tries < 50:		
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i];
                if betamax == Math.inf or betamax == -Math.inf:
                    beta[i] = beta[i] * 2;
                else:
                    beta[i] = (beta[i] + betamax) / 2;
            else:
                betamax = beta[i];
                if betamin == Math.inf or betamin == -Math.inf:
                    beta[i] = beta[i] / 2;
                else:
                    beta[i] = (beta[i] + betamin) / 2;
            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i]);
            Hdiff = H - logU;
            tries = tries + 1;
        # Set the final row of P
        P[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))] = thisP;
    # Return final P-matrix
    #print ("Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta)))
    return P;
	
	
def pca(X = Math.array([]), no_dims = 25):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	#print ("Preprocessing the data using PCA...")
	(n, d) = X.shape;
	X = X - Math.tile(Math.mean(X, 0), (n, 1));
	(l, M) = Math.linalg.eig(Math.dot(X.T, X));
	Y = Math.dot(X, M[:,0:no_dims]);
	return Y;


def tsne(X = Math.array([]), no_dims = 3, max_iter = 1000, initial_dims = 50, perplexity = 30.0):

	# Check inputs
	if X.dtype != "float64":
		print ("Error: array X should have type float64.");
		return -1;

	X = pca(X, initial_dims);
	(n, d) = X.shape;
	#max_iter = 3000;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	Y = Math.random.randn(n, no_dims);
	dY = Math.zeros((n, no_dims));
	iY = Math.zeros((n, no_dims));
	gains = Math.ones((n, no_dims));
	
	# Compute P-values
	P = x2p(X, 1e-5, perplexity);
	P = P + Math.transpose(P);
	P = P / Math.sum(P);
	P = P * 4;									# early exaggeration
	P = Math.maximum(P, 1e-12);
	
	# Run iterations
	for iter in range(max_iter):
		
		# Compute pairwise affinities
		sum_Y = Math.sum(Math.square(Y), 1);		
		num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
		num[range(n), range(n)] = 0;
		Q = num / Math.sum(num);
		Q = Math.maximum(Q, 1e-12);
		
		# Compute gradient
		PQ = P - Q;
		for i in range(n):
			dY[i,:] = Math.sum(Math.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);
			
		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		gains[gains < min_gain] = min_gain;
		iY = momentum * iY - eta * (gains * dY);
		Y = Y + iY;
		Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));
		
		# Compute current value of cost function
		if (iter + 1) % 100 == 0:
		#if (iter + 1) == max_iter:
        #if (iter + 1) == max_iter:
			C = Math.sum(P * Math.log(P / Q));
			print ("Iteration ", (iter + 1), ": error is ", C)			
		# Stop lying about P-values
		if iter == 100:
			P = P / 4;
	return Y;

def make_matrix(matrix_dic):
    X = []
    labels = []
    for key in matrix_dic:
        X.append(np.float64(matrix_dic[key]))
        labels.append(key)
    return np.array(X), labels

def TSNE_reduce_dimensionality(matrix_dic):    
    X, labels = read_in_vectors_make_array(matrix_dic)
    dimensionality = 2
    max_iter = 2000
    Y = tsne(X, dimensionality, max_iter, 20, 5.0);  # PCA dimensions; Perplexity = 3 dimensions = 5 plex = 30 dim must lower than plex when matrix is sparse
    matrix_dic_reduced = {}
    for i in range(len(Y)):
        matrix_dic_reduced[labels[i]] = Y[i]
    #return Y, labels
    return matrix_dic_reduced
    
#********************** TSNE END ********************************

def save_clusters_and_coords2(rulevecs_std,labels,hindex,highest_y_km,outfilename):
      
    outfile = open(outfilename, "w")
    for i in range(len(rulevecs_std)):
        row = ''
        if len(rulevecs_std[i]) == 2:
            row = str(rulevecs_std[i][0]) + ',' + '0' + ',' + str(rulevecs_std[i][1])
        else:
            for j in range(len(rulevecs_std[i])):
                row = row + str(rulevecs_std[i][j]) + ','
            row = row.rstrip(',')
        if (hindex > 0):
            outfile.write(labels[i]+','+str(highest_y_km[i])+','+row+'\n')
        else:
            outfile.write(labels[i]+','+str(0)+','+row+'\n')

    outfile.close()

################################################
    
def readin_matrix(inputfilename):
    matrix_dic = {}
    with open(inputfilename,"r") as infile:
        for row in infile:
            row = row.rstrip('\n')
            line = row.split(',')
            numlist = []
            for i in range(1,len(line)):
                if line[i] != '':
                    numlist.append(float(line[i]))
            matrix_dic[line[0]] = np.array(numlist)
    return matrix_dic
    

def save_coords(matrix_dic_reduced,output_file):
    with open(output_file,"w") as outfile:
        for key in matrix_dic:
            #outfile.write(key+','+'10'+','+str(matrix_dic_reduced[key][0])+','+'0'+','+str(matrix_dic_reduced[key][1])+'\n')
            outfile.write(key+','+str(matrix_dic_reduced[key][0]/gain)+','+'0,'+str(matrix_dic_reduced[key][1]/gain)+'\n')
        


################  Begin here #################

input_file = 'diary_even_vectors_words2.txt'
output_file = 'diary_even_vectors_words2_tsne_20_5.txt'
gain = 100

matrix_dic = readin_matrix(input_file)

matrix_dic_reduced = TSNE_reduce_dimensionality(matrix_dic)

save_coords(matrix_dic_reduced,output_file)

#save_clusters_and_coords2(high_coord_std,high_labels,high_hindex,high_highest_y_km,output_file)
