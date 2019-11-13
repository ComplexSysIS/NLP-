
"""
Twins diary data/ identify text clusters in the twins diary by even/odd numbering
of the twins.
@author: YSHAO45
"""
"""
Word2vec
"""
# Word2vec as an alternative for principal component analysis
# Potential cons of using Word2vec for this dataset: if frequence < 2, leave out
# gensim modules
# Use 
import gensim
from gensim.models import Word2Vec

# logging
import logging
import os.path
import sys
import numpy as np

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

import os            
os.chdir('H:\\Twins Data Diary\\1_word2vec')
os.getcwd()
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in open(self.filename):
            self.sentences = line.split()
            yield line.split()

def save_vectors(model,outfilename):
    with open(outfilename,'w') as outfile:
        for i in range(len(model.wv.syn0)):
            vecstr = ''
            for j in range(len(model.wv.syn0[i])):
                vecstr = vecstr + str(model.wv.syn0[i][j]) + '\t'
            vecstr = vecstr.rstrip('\t')
            outfile.write(model.wv.index2word[i]+'\t'+vecstr+'\n')
            
def save_normed_vectors(model,outfilename):
    with open(outfilename,'w') as outfile:
        for i in range(len(model.wv.syn0)):
            vecstr = ''
            normed_vec = model.wv.syn0[i] / np.linalg.norm(model.wv.syn0[i])
            for j in range(len(normed_vec)):
                vecstr = vecstr + str(normed_vec[j]) + '\t'
            vecstr = vecstr.rstrip('\t')
            outfile.write(model.wv.index2word[i]+'\t'+vecstr+'\n')    

def save_word_counts(model,outfilename):
    with open(outfilename,'w') as outfile:       
        for i in range(len(model.wv.syn0)):
            vocab_obj = model.wv.vocab[model.wv.index2word[i]]
            count = vocab_obj.count
            outfile.write(model.wv.index2word[i]+'\t'+str(count)+'\n')


############### Import diary with even ID  #########################
# diary txt has been cleaned prior to this step.
sentences = MySentences('./diary_even.txt') # LabeledLineSentence(sources)
outfilename = 'diary_even_vectors_updated.txt'
outfilename_normed = 'diary_even_normed_vectors_updated.txt'
save_word_counts_file = 'diary_even_word_counts_final.txt'

model = Word2Vec(sentences, size=100, window=5, min_count=3, workers=8, sg=1, iter=4)
### Set min_count to three since our diary trial is small, size = 100 default since we do n't have 
## bigger size training data
model.save('./diary_even.w2v')

model = Word2Vec.load('./diary_even.w2v')

save_vectors(model,outfilename)
save_normed_vectors(model,outfilename_normed)
save_word_counts(model,save_word_counts_file)

"""
Vector selection, decided to use text libraries of emotion words, adjectives,
nouns and a comprehensive vocabulary library as a comparison point.
"""
def fill_selection_list(select_list_file):
    s_file = open(select_list_file,'r')
    select_list = s_file.read().splitlines()
    return select_list

def open_and_resave_vectors_file(selection_list,infilename,outfilename):
    outfile = open(outfilename, "w")
    with open(infilename,'r') as infile:
        for row in infile:
            line = row.split('\t')
            if line[0] in selection_list:
                outfile.write(row)                
    outfile.close()
################# Import dataset #################
infilename = 'diary_even_vectors_updated.txt'
outfilename = 'diary_even_vectors_allwords.txt'
select_list_file = '47155words.csv' 
selection_list = fill_selection_list(select_list_file)
open_and_resave_vectors_file(selection_list,infilename,outfilename)
"""
Emotion words did not work out probably because of the setting where diary is recorded.
It is a guided interview. I also did not choose to use preposition since they are mainly functional.
Ended up with 6159 verbs library, 36832 nouns, and 47155 words
Maybe worth it to filter name-related nouns and name-related verbs
"""

"""
Insert Comma prior to PCA and dimensionality reduction
"""
def open_and_resave_vectors_file(infilename,outfilename):
    count = 0
    outfile = open(outfilename, "w")
    with open(infilename,'r') as infile:
        for row in infile:
            if count > 0:
                vector = ''
                line = row.split(delimiter)
                line[len(line)-1] = line[len(line)-1].rstrip('\n')
                for j in range(len(line)):
                    vector = vector + str(line[j]) + ','
                vec = vector.rstrip(',')
                outfile.write(vec+'\n')
            count += 1
    outfile.close()

################# Insert comma  #################

infilename = 'diary_even_vectors_allwords.txt'
outfilename = 'diary_even_vectors_allwords2.txt' ## vector file with comma separation 
# cleaned before principal component analysis and dimensionality reduction using
# tSNE
delimiter = '\t'
open_and_resave_vectors_file(infilename,outfilename)


"""
Dimensionality Reduction using tSNE
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
    if sumP < .00001:  # addition to avoid errors
        sumP = .00001
    H = Math.log(sumP) + beta * Math.sum(D * P) / sumP;
    P = P / sumP;
    return H, P;

def x2p(X = Math.array([]), tol = 1e-5, perplexity = 30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""
    # Initialize variables
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
    
#********************** TSNE ENDS here ********************************

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
        


###############Import dataset  #################

input_file = 'diary_even_vectors_allwords2.txt'
output_file = 'diary_even_vectors_allwords2_tsne_11_11.txt'
gain = 100

matrix_dic = readin_matrix(input_file)

matrix_dic_reduced = TSNE_reduce_dimensionality(matrix_dic)

save_coords(matrix_dic_reduced,output_file)
#save_clusters_and_coords2(high_coord_std,high_labels,high_hindex,high_highest_y_km,output_file)

################### Cluster using kmeans ###################################
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import glob
	
def obtain_data(filename):
    rulelabel = []
    rulevecs = []
    rulevecs_array = []
    rulevecs_std = []
    with open(filename,'r') as infile:
        for row in infile:
            rulelabel.append(row.split(',')[0])
            frow = []
            for j in range(1,len(row.split(','))):
                frow.append(float(row.split(',')[j]))
            rulevecs.append(np.array(frow))
    rulevecs_array = np.array(rulevecs)
    if rulelabel != []:        
        if len(rulelabel) > 1:
            sc = StandardScaler()
            rulevecs_std = sc.fit_transform(rulevecs_array)
        else:
            rulevecs_std = rulevecs_array
    return rulelabel, rulevecs_std, rulevecs_array 


def determine_k(rulevecs_std,n_clusterlimit):
    from sklearn.metrics import silhouette_samples
    third = int(n_clusterlimit/3)
    silhouette_avg = [0,0]
    highest_index = 0
    for nclusters in range(2,third):
        km = KMeans(nclusters, 
                    init='k-means++', 
                    n_init=50, 
                    max_iter=300,
                    tol=1e-04,
                    random_state=0)
        y_km = km.fit_predict(rulevecs_std)
        cluster_labels = np.unique(y_km)
        silhouette_vals = silhouette_samples(rulevecs_std, y_km, metric='euclidean')
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[y_km == c]
            c_silhouette_vals.sort()
        silhouette_avg.append(np.mean(silhouette_vals))
        last_avg_index = len(silhouette_avg)-1
        if silhouette_avg[last_avg_index] > silhouette_avg[last_avg_index-1]:
            highest_index = last_avg_index
        else:
            break
    return silhouette_avg, highest_index

def save_clusters_and_coords(rulevecs_std,rulelabel,hindex,outfilename):
    if hindex > 0:
        km = KMeans(hindex, 
                    init='k-means++', 
                    n_init=50, 
                    max_iter=300,
                    tol=1e-04,
                    random_state=0)
        y_km = km.fit_predict(rulevecs_std)     
    
    outfile = open(outfilename, "w")
    for i in range(len(rulevecs_orig)):
        row = ''
        for j in range(len(rulevecs_std[i])):
            row = row + str(rulevecs_std[i][j]) + ','
        row = row.rstrip(',')
        if hindex > 0:
            outfile.write(rulelabel[i]+','+str(y_km[i])+','+row+'\n')
        else:
            outfile.write(rulelabel[i]+','+str(0)+','+row+'\n')
    outfile.close()    


#################### Import data  #################

rulelabel = []
rulevecs = []

#input_files = glob.glob('*.csv')
input_files = ['diary_even_vectors_allwords2_tsne_11_11.txt']
for infilename in input_files:
    outfilename = infilename.replace('.txt','_final.txt')

    rulelabel, rulevecs_std, rulevecs_orig = obtain_data(infilename)
    k,hindex = determine_k(rulevecs_std,len(rulelabel))
    print (hindex,infilename)
    save_clusters_and_coords(rulevecs_std,rulelabel,hindex,outfilename)
