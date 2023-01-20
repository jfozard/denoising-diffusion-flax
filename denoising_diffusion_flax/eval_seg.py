
import numpy as np
from scipy.sparse import coo_matrix
import scipy.ndimage as nd
import numpy.random as npr
from math import log, sqrt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from scipy.ndimage import distance_transform_edt, find_objects, binary_erosion


def precision_recall_IoU(X, Y, return_M=False):
    IoU_map = matching_IoU(X,Y)
    M = len(IoU_map)
    lX = np.unique(X)
    lY = np.unique(Y)
    nX = len(lX[lX>0])
    nY = len(lY[lY>0])
    print(nX, nY, M)
    if return_M:
        return M/float(nX), M/float(nY), M
    else:
        return M/float(nX), M/float(nY)


def gen_ndata(X, Y):
    # Calculation of terms used in Rand / Jaccard
    
    w = np.ones(X.shape)

    n = np.sum(w)

    lX, iX= np.unique(X, return_inverse=True)
    lY, iY = np.unique(Y, return_inverse=True)

    # have lX = [0, 2, 3] iX = [0, 1, 1, 2, 2]
    # want cX = [np.sum(w[iX==j]) for j in range(len(lX))]

#    cX = np.array([np.sum(w[X==j]) for j in lX])
#    cY = np.array([np.sum(w[Y==j]) for j in lY])
 
    
    I = iX.flatten() #[X.flatten()]
    J = iY.flatten() #[Y.flatten()]

    nX = lX.shape[0]
    nY = lY.shape[0]

    M = coo_matrix((w.flatten(), (I, J)), shape=(nX, nY))
    cX = np.ravel(M.sum(axis=1))
    cY = np.ravel(M.sum(axis=0))
    
#    M = np.array(M.todense())

    M2 = (M.multiply(M)).sum()
    cX2 = np.sum(cX*cX)
    cY2 = np.sum(cY*cY)
    
    n11 = 0.5*(M2-n)
    n10 = 0.5*(cX2 - M2)
    n01 = 0.5*(cY2 - M2)
    n00 = 0.5*n*(n-1)-n11-n10-n01

    return n00, n01, n10, n11


def rand_index(X,Y):
    n00, n01, n10, n11 = gen_ndata(X,Y)
    n = np.product(X.shape)
    R = 1 - (n11+n00)/(0.5*n*(n-1))
    return R
    
def jaccard_index(X,Y):
    n00, n01, n10, n11 = gen_ndata(X,Y)
    n = np.product(X.shape)
    J = 1 - n11/(n11+n01+n10)
    return J

    
def calc_cM(seg, exact, weights=None):

    X = seg
    Y = exact
    
    # Calculation of IoU
    if weights is None:
        w = np.ones(X.shape)
        n = np.product(X.shape)
    else:
        w = weights
        n = np.sum(w)
# 
    lX, iX = np.unique(X, return_inverse=True)
    lY, iY = np.unique(Y, return_inverse=True)

    # have lX = [0, 2, 3] iX = [0, 1, 1, 2, 2]
    # want cX = [np.sum(w[iX==j]) for j in range(len(lX))]

#    cX = np.array([np.sum(w[X==j]) for j in lX])
#    cY = np.array([np.sum(w[Y==j]) for j in lY])
 
    cX = nd.sum(w, labels=X, index=lX)
    cY = nd.sum(w, labels=Y, index=lY)
    
    I = iX.flatten() #[X.flatten()]
    J = iY.flatten() #[Y.flatten()]

#    print w.shape, I.shape, J.shape
    
    M = coo_matrix((w.flatten(), (I, J))).tocsr()

    # loop over all objects in seg

    return lX, lY, cX, cY, n, M




def calc_acme_criterion(X, Y, w=None, threshold=0.75, return_criterion=False):
    # Max_g ((Ra ^ Rg) /Rg) for each a in X
    lX, lY, aX, aY, n, M = calc_cM(X, Y, w)
    acme_map = {}
    acme_vals = {}
    for i, li in enumerate(lX):
        i_area = aX[i]
        if li>0:
            j = M.indices[M.indptr[i]:M.indptr[i+1]]
            intersection_area = M.data[M.indptr[i]:M.indptr[i+1]]
            lj = lY[j]
            mask = lj>0
            lj = lj[mask]
            j = j[mask]
            intersection_area = intersection_area[mask]
            j_area = aY[j]
            union_area = i_area + j_area - intersection_area
            c = intersection_area/j_area
            if len(c>0):
                k = np.argmax(c)
                if c[k]>=threshold:
                    acme_map[li] = lj[k]
                acme_vals[li] = c[k]
            else:
                acme_vals[li] = 1.0
    if return_criterion:
        return acme_map, acme_vals
    else:
        return acme_map
        


def matching_IoU(X, Y, threshold=0.75, w=None, return_best=False):
    lX, lY, aX, aY, n, M = calc_cM(X, Y, w)
    IoU_map = {}
    IoU_best = {}
    for i, li in enumerate(lX):
        i_area = aX[i]
        if li>0:
            j = M.indices[M.indptr[i]:M.indptr[i+1]]
            intersection_area = M.data[M.indptr[i]:M.indptr[i+1]]
            lj = lY[j]
            mask = lj>0
            lj = lj[mask]
            j = j[mask]
            intersection_area = intersection_area[mask]
            if len(j)>0:
                j_area = aY[j]
                union_area = i_area + j_area - intersection_area
                IoU = intersection_area/union_area
                k = np.argmax(IoU)
                if IoU[k]>=threshold:
                    IoU_map[li] = lj[k]
                IoU_best[li] = IoU[k]
            else:
                IoU_best[li] = None
    if return_best:
        return IoU_map, IoU_best
    else:
        return IoU_map
            



def max_freq(u, w):
    # Find most frequent (weighted) value in u

    v, indices = np.unique(u, return_inverse=True)

    c = np.bincount(indices, weights=w)
    i = np.argmax(c)
    overlap = c[i]
    
    s = v[i]

    return s, overlap

def test_max_freq():
    print(max_freq([2,3,4],[0.1, 0.2, 0.3]),  (4, 0.3))

    print(max_freq([2,3,4,2,2],[0.1, 0.2, 0.3, 0.1, 0.15]),  (2, 0.35))
    

def calc_overlaps(u, w):
    # Find most frequent (weighted) value in u

    v, indices = np.unique(u, return_inverse=True)

    c = np.bincount(indices, weights=w)

    return v, c

    
def per_cell_JI(R, S, w=None):
    if w is None:
        w = np.ones(R.shape)

    l_R = np.unique(R)
    per_cell_JI = {}

    seg = 0.0
    
    for l in l_R:
        
        u = S[R==l]
        w2 = w[R==l]
        
        area_l = np.sum(w2)



        s, overlap = max_freq(u, w2)
                
        if overlap>0.5*area_l:
            union_area = np.sum(w[np.logical_or(R==l, S==s)])
            seg += overlap/union_area

        v, c = calc_overlaps(u, w2)

        unions = [ np.sum(w[np.logical_or(R==l, S==t)]) for t in v ]

        JI = np.array(c)/np.array(unions)
        i = np.argmax(JI)
        per_cell_JI[l] = JI[i]
        
    seg = seg/len(l_R)
            
    return np.mean(list(per_cell_JI.values())), per_cell_JI



def test_2D():
    R = np.zeros((10,10))
    R[2:6,2:6] = 1
    R[6:9, 6:9] = 2

    
    #S = npr.randint(10, size=(10,10))
    S = np.array(R)
    
    S[S>0] += 1

    S[0,0] = 10
    

    print('nmi sklearn', normalized_mutual_info_score(R.flatten(), S.flatten()))
    print('anmi sklearn', adjusted_mutual_info_score(R.flatten(), S.flatten()))

    print('nrand', adjusted_rand_score(R.flatten(),S.flatten()))

    print('rand', rand_index(R,S))
    print('jaccard', jaccard_index(R,S))

    print('SEG', per_cell_JI(R.flatten(), S.flatten())[0])
    

def remove_outside(A, seg, level=0.5):
    A = np.array(A)
    l = np.unique(A)
    overlap = nd.mean(seg==0, labels=A, index=l)
    for i, v in zip(l, overlap):
        if v>level:
            A[A==i] = 0
    return A


