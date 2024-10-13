# Backpropagation notes
Let's define:
$w_{jk}^{l}$ - weight to neuron $j$ of layer $l$ from neuron $k$ of layer $(l-1)$.
$b_{j}^{l}$ - bias of neuron $j$ of layer $l$.
$a_{j}^{l}$ - activation of neuron $j$ of layer $l$.

So,
$$
a_{j}^{l}= \sigma  \left ({ \sum_{k} w_{jk}^{l} a_{j}^{l} + b_{j}^{l} } \right )
$$
Now, let's define $w_{j}^{l}$ - vector containing all weights going in neuron $j$ of layer $l$:
$$
w_{j}^{l} = \begin{bmatrix} w_{j1}^{l} \\ w_{j2}^{l} \\ \vdots \\ w_{jk}^{l} \\ \vdots \\ w_{jn}^{l} \end{bmatrix} 
$$
where $n$ is the number of neurons in layer $(l-1)$, i.e. the number of weights going in neuron $j$ of layer $l$.

Analogously, $a^{l}$ is the vector of all activations in layer $l$:
$$
a^{l} = \begin{bmatrix} a_{1}^{l} \\ a_{2}^{l} \\ \vdots \\ a_{j}^{l} \\ \vdots \\ a_{n}^{l} \end{bmatrix}
$$
where $n$ is the number of neurons in layer $l$.

And $b^{l}$ - the vector of all biases for neurons in layer $l$:
$$
b^{l} = \begin{bmatrix} b_{1}^{l} \\ b_{2}^{l} \\ \vdots \\ b_{j}^{l} \\ \vdots \\ b_{n}^{l} \end{bmatrix}
$$
where $n$ is the number of neurons in layer $l$.

Then the previous expression can be rewritten as (notice the dot product):
$$
a_{j}^{l} =  \sigma  \left ({ w_{j}^{l} \cdot a^{l-1} + b_{j}^{l} } \right )
$$
If we instead use the transpose of $w_{j}^{l}$:
$$
(w_{j}^{l})^{T} = \begin{pmatrix} w_{j1}^{l} w_{j2}^{l}  \dots w_{jk}^{l}  \dots w_{jn}^{l} \end{pmatrix}
$$
where $n$ is the number of weights going in neuron $j$ of layer $l$.

Then, we can rewrite the expression again as:
$$
a_{j}^{l} =  \sigma  \left ({ (w_{j}^{l})^{T} a^{l-1} + b_{j}^{l} } \right )
$$
(this is now matrix multiplication).

Now, we can define $w^{l}$ to be the matrix containing all rows $(w_{j}^{l})^{T}$:
$$
w^{l} = \begin{bmatrix} (w_{1}^{l})^{T} \\ (w_{2}^{l})^{T} \\ \vdots \\ (w_{j}^{l})^{T} \\ \vdots \\ (w_{n}^{l})^{T} \end{bmatrix}
$$
where $n$ is the number of neurons in layer $l$.

Or, more explicitly:
$$
w^{l} = 
\begin{bmatrix} 
w_{11}^{l} w_{12}^{l} \dots w_{1k}^{l} \dots w_{1m}^{l}  \\
w_{21}^{l} w_{22}^{l} \dots w_{2k}^{l} \dots w_{2m}^{l}  \\
\vdots \\
w_{j1}^{l} w_{j2}^{l} \dots w_{jk}^{l} \dots w_{jm}^{l}  \\
\vdots \\
w_{n1}^{l} w_{n2}^{l} \dots w_{nk}^{l} \dots w_{nm}^{l}  \\
\end{bmatrix}_{\substack{n \times m}}
$$
where $n$ is the number of neurons in layer $l$ and $m$ is the number of neurons in layer $(l-1)$.

So, an equation for the vector $a^{l}$ containing all activations in layer $l$ would be:
$$
a^{l} = \sigma \left({ w^{l}a^{l-1} + b^{l} }\right)
$$
Let $z_{j}^{l}$ be the weighted input to neuron $j$ in layer $l$. This is basically the activation before it is ran through $\sigma$:
$$
z_{j}^{l} = (w_{j}^{l})^{T}a_{j}^{l-1} + b_{j}^{l}
$$
And $z^{l}$ be the vector of all the weighted inputs of neurons in layer $l$:
$$
z^{l} = w^{l}a^{l-1} + b^{l}
$$
Our cost function is:
$$
C = \frac{1}{n} \sum_{x} \frac{1}{2} \lVert y(x)-a^{L}(x) \rVert^{2}
$$
A few clarifications:
$L$ - number of layers in the neural network, so $a^{L}$ just means the vector of activations of the neurons in the last layer (output) $(l = L)$.
$a^{L}(x)$ - vector of activations in the last (output) layer L _for a given input_ $x$.
$y(x)$ - vector containing all _expected_ activations in the last (output) layer.
$\lVert y(x) - a^{L}(x) \rVert^{2}$ - magnitude of the vector, squared (or the dot product of the vector with itself).
$\frac{1}{n} \sum_{x} \dots$ - average of all those magnitudes over all $n$ inputs. Can be rewritten as:
$$
C = \frac{1}{n} \sum_{k=1}^{n} \frac{1}{2} \lVert y(x_{k})-a^{L}(x_{k}) \rVert^{2}
$$
$\frac{1}{2}$ - scaling factor for better-looking derivative later on.

We need to make an assumption that the cost $C$ can be written as an average over cost functions $C_{x}$ for individual training inputs:
$$
C = \frac{1}{n} \sum_{x} C_{x}
$$
We need this assumption because backpropagation computes the partial derivatives $\partial C_{x}/\partial w$ and $\partial C_{x} /\partial b$ for a single training example $x$.

Second assumption is that the cost can be written as a function of the outputs $a^{L}$.
We are going to refer to the cost of a single training example $C_{x}$ as just $C$ from now on.

Imagine a small change $\Delta z_{j}^{l}$ to the weighted sum input $z_{j}^{l}$ of neuron $j$ of layer $l$. So, instead of the activation being
$$
a_{j}^{l} = \sigma (z_{j}^{l})
$$
The activation is
$$
a_{j}^{l} = \sigma (z_{j}^{l} + \Delta z_{j}^{l})
$$
This causes the overall cost to change by $\frac{\partial C}{\partial z_{j}^{l}} \Delta z_{j}^{l}$. If $\partial C / \partial z_{j}^{l}$ is large, $\Delta z_{j}^{l}$ can make a big difference to the cost in that direction. If $\partial C / \partial z_{j}^{l}$ is small, $\Delta z_{j}^{l}$ can't improve the cost much at all
So, let's define the error $\delta_{j}^{l}$ of a neuron $j$ in layer $l$:
$$
\delta_{j}^{l} = \frac{\partial C}{\partial z_{j}^{l}}
$$
Then, we can define $\delta^{l}$ as the vector containing all errors $\delta_{j}^{l}$ of neurons in layer $l$:
$$
\delta^{l} = \begin{bmatrix} \delta_{1}^{l} \\ \delta_{2}^{l} \\ \vdots \\ \delta_{j}^{l} \\ \vdots \\ \delta_{n}^{l} \end{bmatrix}
$$
where $n$ is the number of neurons in layer $l$

Backpropagation will compute $\delta^{l}$ for every layer

## First equation of backpropagation - equation for error $\delta^{L}$ of the output layer:
Equation for error of output layer: 
$$
\delta_{j}^{L} = \frac{\partial C}{\partial a_{j}^{L}} \cdot \frac{\partial a_{j}^{L}}{\partial z_{j}^{L}} = \frac{\partial C}{\partial a_{j}^{L}} \sigma'(z_{j}^{L})
$$
For a quadratic cost function $\partial C / \partial a_{j}^{L}$ would be:
$$
\frac{\partial C}{\partial a_{j}^{L}} = \\
\frac{\partial}{\partial a_{j}^{L}} \frac{1}{2} \lVert y(x)-a^{L}(x) \rVert^{2} = \\
\frac{\partial}{\partial a_{j}^{L}} \frac{1}{2} \sum_{k = 1}^{n} ( y_{k} - a_{k}^{L}) ^2 = \\
\frac{\partial}{\partial a_{j}^{L}} \frac{1}{2} (( y_{1} - a_{1}^{L}) ^2 + \dots + ( y_{j} - a_{j}^{L}) ^2 + \dots + ( y_{n} - a_{n}^{L}) ^2) = \\
\frac{\partial}{\partial a_{j}^{L}} \frac{1}{2}( y_{j} - a_{j}^{L}) ^2 = \\
(y_{j} - a_{j}^{L})\cdot(-1) = a_{j}^{L} - y_{j}
$$

If we use the nabla (del) operator:
$$
\nabla_{a} = \begin{bmatrix} \frac{\partial}{\partial a_{1}^{L}} \\ \frac{\partial}{\partial a_{1}^{L}} \\ \vdots \\ \frac{\partial}{\partial a_{j}^{L}} \\ \vdots \end{bmatrix}
$$
and the elementwise product of two vectors (a.k.a. Hadamard product):
$$
\begin{bmatrix} a_{1} \\ a_{2} \\ \vdots \\ a_{k} \\ \vdots \end{bmatrix}
\circ
\begin{bmatrix} b_{1} \\ b_{2} \\ \vdots \\ b_{k} \\ \vdots \end{bmatrix} =
\begin{bmatrix} a_{1}b_{1} \\ a_{2}b_{2} \\ \vdots \\ a_{k}b_{k} \\ \vdots \end{bmatrix}
$$
We can write the equation for the vector of component errors in the last layer $\delta^{L}$:
$$
\delta^{L} = \nabla_{a}C \circ \sigma'(z^{L})
$$
In the case of a quadratic cost function the error for the last layer would be:
$$
\delta^{L} = (a^{L} - y) \circ \sigma'(z^{L})
$$

## Second equation of backpropagation - equation for error of any layer $\delta^{l}$ in terms of error of the next layer $\delta^{l+1}$:
Let's look at the component of error of neuron $k$ in layer $l$ $\delta_{k}^{l}$ coming from the error of neuron $j$ in layer $(l+1)$ $\delta_{j}^{l+1}$:
We know that:
$$
\delta_{j}^{l+1} = \frac{\partial C}{\partial z_{j}^{l+1}}
$$
And we know that:
$$
z_{j}^{l+1} = w_{j}^{l+1}a^{l} + b_{j}^{l+1} = \\
\sum_{m}(w_{jm}^{l+1}a_{m}^{l}) + b_{j}^{l+1} = \\
w_{j1}^{l+1}a_{1}^{l} + w_{j2}^{l+1}a_{2}^{l} + \dots + w_{jk}^{l+1}a_{k}^{l} + \dots + b_{j}^{l+1}
$$
So the partial derivative $\partial z_{j}^{l+1} / \partial a_{k}^{l}$ is:
$$
\frac{\partial z_{j}^{l+1}}{\partial a_{k}^{l}} = \frac{\partial}{\partial a_{k}^{l}}(w_{j1}^{l+1}a_{1}^{l} + w_{j2}^{l+1}a_{2}^{l} + \dots + w_{jk}^{l+1}a_{k}^{l} + \dots + b_{j}^{l+1}) = \\
\frac{\partial}{\partial a_{k}^{l}} (w_{j1}^{l+1}a_{1}^{l}) + \frac{\partial}{\partial a_{k}^{l}} (w_{j2}^{l+1}a_{2}^{l}) + \dots + \frac{\partial}{\partial a_{k}^{l}} (w_{jk}^{l+1}a_{k}^{l}) + \dots + \frac{\partial}{\partial a_{k}^{l}} (b_{j}^{l+1}) = \\
\frac{\partial}{\partial a_{k}^{l}}(w_{jk}^{l+1}a_{k}^{l}) = w_{jk}^{l+1}
$$
So, using the chain rule:
$$
\text{component of } \frac{\partial C}{\partial a_{k}^{l}} \text{ coming from } \delta_{j}^{l+1} = \frac{\partial C}{\partial z_{j}^{l+1}} \cdot \frac{\partial z_{j}^{l+1}}{\partial a_{k}^{l}} = w_{jk}^{l+1} \delta_{j}^{l+1}
$$
So, the derivative of $C$ with respect to $a_{k}^{l}$ is:
$$
\frac{\partial C}{\partial a_{k}^{l}} = w_{1k}^{l+1} \delta_{1}^{l+1} + w_{2k}^{l+1} \delta_{2}^{l+1} + \dots + w_{jk}^{l+1} \delta_{j}^{l+1} + \dots = \\
\sum_{m} w_{mk}^{l+1} \delta_{m}^{l+1} = \begin{bmatrix} w_{1k}^{l+1} \\ w_{2k}^{l+1} \\ \vdots \\ w_{jk}^{l+1} \\ \vdots \end{bmatrix} \cdot \delta^{l+1}
$$
Then, using the chain rule again, the error $\delta_{k}^{l}$ would be:
$$
\delta_{k}^{l} = \frac{\partial C}{\partial z_{k}^{l}} = \frac{\partial C}{\partial a_{k}^{l}} \cdot \frac{\partial a_{k}^{l}}{\partial z_{k}^{l}} = \\
\begin{bmatrix} w_{1k}^{l+1} \\ w_{2k}^{l+1} \\ \vdots \\ w_{jk}^{l+1} \\ \vdots \end{bmatrix} \cdot \delta^{l+1} \cdot \sigma'(z_{k}^{l})
$$
This dot product can be rewritten as matrix multiplication:
$$
\delta_{k}^{l} = \begin{pmatrix} w_{1k}^{l+1} w_{2k}^{l+1} \dots w_{jk}^{l+1} \dots \end{pmatrix} \delta^{l+1} \cdot \sigma'(z_{k}^{l})
$$
Then finally, the vector $\delta^{l}$ containing all individual errors $\delta_{k}^{l}$ would be:
$$
\delta^{l} = \begin{bmatrix} 
\begin{pmatrix} w_{11}^{l+1} w_{21}^{l+1} \dots w_{j1}^{l+1} \dots w_{n1}^{l+1} \end{pmatrix} \delta^{l+1} \cdot \sigma'(z_{1}^{l}) \\ 
\begin{pmatrix} w_{12}^{l+1} w_{22}^{l+1} \dots w_{j2}^{l+1} \dots w_{n2}^{l+1} \end{pmatrix} \delta^{l+1} \cdot \sigma'(z_{2}^{l}) \\
\vdots \\
\begin{pmatrix} w_{1k}^{l+1} w_{2k}^{l+1} \dots w_{jk}^{l+1} \dots w_{nk}^{l+1} \end{pmatrix} \delta^{l+1} \cdot \sigma'(z_{k}^{l}) \\
\vdots \\
\begin{pmatrix} w_{1m}^{l+1} w_{2m}^{l+1} \dots w_{jm}^{l+1} \dots w_{nm}^{l+1} \end{pmatrix} \delta^{l+1} \cdot \sigma'(z_{k}^{l})
\end{bmatrix} = 
$$
$$
\left ({
\begin{bmatrix}
w_{11}^{l+1} w_{21}^{l+1} \dots w_{j1}^{l+1} \dots w_{n1}^{l+1} \\
w_{12}^{l+1} w_{22}^{l+1} \dots w_{j2}^{l+1} \dots w_{n2}^{l+1} \\
\vdots \\
w_{1k}^{l+1} w_{2k}^{l+1} \dots w_{jk}^{l+1} \dots w_{nk}^{l+1} \\
\vdots \\
w_{1m}^{l+1} w_{2m}^{l+1} \dots w_{jm}^{l+1} \dots w_{nm}^{l+1}
\end{bmatrix}_{\substack{m \times n}}
\delta^{l+1}_{\substack{n \times 1}}
}\right ) \circ \sigma'(z^{l})_{\substack{m \times 1}} = \\
$$
$$
\delta^{l} = \left ({ (w^{l+1})^{T} \delta^{l+1} } \right ) \circ \sigma'(z^{l})
$$

## Third equation of backpropagation - derivative (rate of change) of cost function $C$ with respect to a bias $b_{j}^{l}$ in terms of error $\delta_{j}^{l}$:

We can use the chain rule:
$$
\frac{\partial C}{\partial b_{j}^{l}} = 
\frac{\partial C}{\partial z_{j}^{l}} \cdot \frac{\partial z_{j}^{l}}{\partial b_{j}^{l}} = 
\delta_{j}^{l} \frac{\partial}{\partial b_{j}^{l}}(w_{j}^{l} \cdot a^{l-1} + b_{j}^{l}) = \delta_{j}^{l}
$$

## Fourth equation of backpropagation - derivative (rate of change) of cost function $C$ with respect to a weight $w_{jk}^{l}$ in terms of error $\delta_{j}^{l}$:

We can again use the chain rule:
$$
\frac{\partial C}{\partial w_{jk}^{l}} =
\frac{\partial C}{\partial z_{j}^{l}} \cdot \frac{\partial z_{j}^{l}}{\partial w_{jk}^{l}} =
\delta_{j}^{l} \frac{\partial}{\partial w_{jk}^{l}}(w_{j}^{l} \cdot a^{l-1} + b_{j}^{l}) = \\
\delta_{j}^{l} \frac{\partial}{\partial w_{jk}^{l}}(w_{j1}^{l} a_{1}^{l-1} + w_{j2}^{l} a_{2}^{l-1} + \dots + w_{jk}^{l} a_{k}^{l-1} + \dots + w_{jm}^{l} a_{m}^{l-1}  + b_{j}^{l})
$$

> Written with [StackEdit](https://stackedit.io/).
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5MzQ4MDU3MjgsLTIxMDUzMjg1NjFdfQ
==
-->