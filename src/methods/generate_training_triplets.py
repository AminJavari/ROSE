import numpy
import scipy.io
import pickle

mat = scipy.io.loadmat('/home/suhang/Documents/MATLAB/SINE_Attr/dataset/epinions/split2/epinions_80_percent_1.mat')
A = mat['A'].todense()
testA = mat['testA'].todense()

threshold = 200

triplets = []

for i in range(A.shape[0]):
    poss = numpy.where(A[i,:] == 1)[1]
    negs = numpy.where(A[i,:] == -1)[1]

    # for links vi + vj, vi - vk add (vi vj vk)
    # if no negative, add (vi vj 0)
    tuples = []
    for pos_ind in poss:
        for neg_ind in negs:
            tuples.append((pos_ind+1,neg_ind+1))

    if len(negs) == 0:
        for pos_ind in poss:
            tuples.append((pos_ind+1,0))

    if len(tuples) > threshold:
        randind = numpy.random.permutation(len(tuples))[0:threshold]
        newtuples = []
        for ind in randind:
            newtuples.append(tuples[ind])
        tuples = newtuples

    for tuple in tuples:
        triplets.append([i+1, tuple[0], tuple[1]])

triplets = numpy.asarray(triplets, dtype='int32')

pickle.dump(triplets, open('epinions_large.p', 'w'))