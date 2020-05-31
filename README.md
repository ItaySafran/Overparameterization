# Overparameterization
Code used in the experiment performed in the paper "The Effects of Mild Over-parameterization on the Optimization Landscape of Shallow ReLU Neural Networks".


Experiment description:

In the experiment, for each n=k between 6 and 100, we ran 500 instantiations of gradient descent on the objective (Equation 2 in the paper), each using an independent and standard Xavier random initialization and a fixed step size of 5/k, till the norm of the gradient was at most 10^-12. For each point found, we computed the spectrum of its Hessian to ensure that its minimal eigenvalue is positive (using floating point computations), which was always the case. We identified points that were equivalent up to permutations of the neurons and their coordinates (up to Frobenius norm of at most 5*10^-9). For further information on the experiment results see section 4.2 in the paper.




Minimum class structure:

The Minimum class was made to exploit the symmetric structure of the minima found, so we can store them in a compact manner and allow a quick and efficient analysis of the points found (without this compression, the 'Converged minima' files created weigh approximately 1GB whereas the compressed files weigh less than 1MB).


threshold - the maximal Frobenius norm between a pair of points to be considered equivalent up to permutations of the neurons and coordinates. If the distance is larger than 10^-6, then the two points are considered different and a new item will be added to the list of minima found when using the 'add_minimum' function.

vals - the values from which the minimum point consists.
p - the number of times each diagonal value in 'vals' appears.
multiplicity - a counter for the number of times this point was converged to.
max_dist - the largest Frobenius distance between the point found and the point reconstructed from its compact representation. In a sense, this measures how much accuracy is lost by using the compact representation over using the actual points found by gradient descent.
min_eig - the smallest eigenvalue of the Hessian of the minimum point.
block_eigs - the eigenvalues of the diagonal block components H_{i,i}' (defined at the first displayed equation in Section 4).
pred_eigs - an upper bound on the smallest eigenvalue of each diagonal block component, computed using the definition of lambda in the beginning of the proof of Theorem 4.1 (Equation 63).
grad_norm - the norm of the gradient of this point. It can be smaller than the requirement for gradient descent convergence since some accuracy is lost when compressing the point to its compact form.
norm_sum - the sum of the Euclidean norms of the neurons of the point.


The following provides an explanation on how the points found are stored.

Suppose n=k=6 and that we have 6 distinct values in the minimum we converged to: a, b, c, d, e, f (e.g. see Example 1 in paper "Spurious Local Minima are Common in Two-Layer ReLU Neural Networks" by Safran and Shamir which has 5 distinct values), where the point is given explicitly by the matrix:

a b f f f f
b a f f f f
e e c d d d
e e d c d d
e e d d c d
e e d d d c

Such a point will be stored with values
vals = array([[a, f, b],
              [e, c, d]])
p = array([2, 4])

More generally: 
vals[i, i] stores the i-th main diagonal value.
vals[i, j] for i!=j<k stores the off-diagonal value appearing at entries where the diagonal values at vals[i, i] and vals[j, j] intersect.
vals[i, -1] stores the off-diagonal values appearing at the main diagonal block of the value vals[i, i].

p[i] stores the number of appearances of each diagonal value vals[i, i].
For any minimum found, we have:
np.sum(minimum.p) == k

if p[i] == 1 then vals[i, -1] = 0, since this value is not used. 


A few concrete examples:

minima=pickle.load(open("Processed minima k=7.p", "rb"))
In[7]: minima[0].vals
Out[7]: array([1, 0])
In[8]: minima[0].p
Out[8]: array([7])

The above is the global minimum. If a non-global minimum is found for k=7, then it will be (with overwhelming probability) the following:

In[9]: minima[1].vals
Out[9]: 
array([[-0.6327039049,  0.2616910758,  0.          ],
       [ 0.1859910463,  0.9969227988, -0.0350801559]])
In[10]: minima[1].p
Out[10]: array([1, 6])
