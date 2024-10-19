# Neural Network Improvements

## Cross-entropy cost function
We know from the backpropagation notes that the quadratic cost function is
$$
C = \frac{1}{n} \sum_{k=1}^{n} \frac{1}{2} \lVert y(x_{k})-a^{L}(x_{k}) \rVert^{2}
$$

And we also know from the backpropagation notes that if we imagine a small change $\Delta z_{j}^{l}$ to the weighted sum input $z_{j}^{l}$ of neuron $j$ of layer $l$, the activation changes from
$$
a_{j}^{l} = \sigma (z_{j}^{l})
$$
to
$$
a_{j}^{l} = \sigma (z_{j}^{l} + \Delta z_{j}^{l})
$$
This causes the overall cost to change by $\frac{\partial C}{\partial z_{j}^{l}} \Delta z_{j}^{l}$. If $\partial C / \partial z_{j}^{l}$ is large, $\Delta z_{j}^{l}$ can make a big difference to the cost in that direction. If $\partial C / \partial z_{j}^{l}$ is small, $\Delta z_{j}^{l}$ can't improve the cost much at all. 
We called $\partial C / \partial z_{j}^{l}$ the error $\delta_{j}^{l}$ of a neuron $j$ in layer $l$:
$$
\delta_{j}^{l} = \frac{\partial C}{\partial z_{j}^{l}}
$$
For a single neuron $j$ in the output layer $L$ over a single training example $x$ using the chain rule we can derive that the error is
$$
\delta_{j}^{L} = \frac{\partial C}{\partial z_{j}^{L}} = \frac{\partial C}{\partial a_{j}^{L}} \cdot \frac{\partial a_{j}^{L}}{\partial z_{j}^{L}} = \\
\frac{\partial}{\partial a_{j}^{L}} \left ({ \frac{1}{2} \lVert y(x)-a^{L}(x) \rVert^{2} }\right ) \frac{\partial}{\partial z_{j}^{L}} \sigma (z_{j}^{L}) = \\
\frac{\partial}{\partial a_{j}^{L}} \frac{1}{2} \sum_{k = 1}^{n} ( y_{k} - a_{k}^{L}) ^2 \sigma'(z_{j}^{L}) = \\
\frac{\partial}{\partial a_{j}^{L}} \frac{1}{2} (( y_{1} - a_{1}^{L}) ^2 + \dots + ( y_{j} - a_{j}^{L}) ^2 + \dots + ( y_{n} - a_{n}^{L}) ^2) \sigma'(z_{j}^{L}) = \\
\frac{\partial}{\partial a_{j}^{L}} \frac{1}{2}( y_{j} - a_{j}^{L}) ^2 \sigma'(z_{j}^{L}) = \\
(y_{j} - a_{j}^{L})\cdot(-1) \cdot \sigma'(z_{j}^{L})= (a_{j}^{L} - y_{j}) \sigma'(z_{j}^{L}) = \\
( \sigma(z_{j}^{L}) - y_{j}) \sigma'(z_{j}^{L})
$$
Dropping the indexes for clarity we get:
$$
( \sigma(z) - y) \sigma'(z)
$$
Let's now compute the derivative of the sigmoid function:
$$
\sigma'(x) = \frac{d \sigma}{dx} \left ({ \frac{1}{1 + e^{-x}} }\right)
$$
Substituting $u(x) = 1 + e^{-x}$ and using the chain rule:
$$
\sigma'(x) = \frac{d \sigma}{du} \cdot \frac{du}{dx} = \frac{d}{du}\left({\frac{1}{u}}\right ) \cdot \frac{d}{dx} \left({ 1 + e^{-x} }\right ) = \\
-\frac{1}{u^{2}}(-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^{2}} = \frac{1}{1+e^{-x}} \cdot \frac{1 + e^{-x} - 1}{1 + e^{-x}} = \\
\sigma(x) \left ({ 1 - \frac{1}{1 + e^{-x}} }\right) = \sigma(x) \left ({ 1 - \sigma(x) }\right )
$$

ToDo: show graphs of $\sigma (x)$ and $\sigma' (x)$

So, following our reasoning for what impact $\Delta z_{j}^{L}$ can have on our cost function based on the error $\delta_{j}^{l} = \partial C/\partial z_{j}^{l}$, we can see that if $z_{j}^{l}$ is both very high or very low, $\sigma'$ will be low and so the whole error $\delta_{j}^{L}$ will be low regardless of the expected value $y$. This means that if for some reason the weighted output is very far from the expected value, learning will be slow when it should be the fastest.

To fix this we can use the Cross-entropy cost function:
$$
C = \frac{1}{n}\sum_{x}\sum_{k} - \left[{ y_{k}ln(a_{k}^{L}) + (1-y_{k})ln(1-a_{k}^{L}) }\right]
$$
Where:
$\frac{1}{n}\sum_x$ is the average over all $n$ training inputs
$\sum_{k}$ is the sum over all output neurons  
$- \left[{ y_{k}ln(a_{k}^{L}) + (1-y_{k})ln(1-a_{k}^{L}) }\right]$ is the cost for each individual neuron in the output layer. 
$ln(a_{k}^{L})$ and $ln(1-a_{k}^{L})$ will always be negative, because $0 < a_{k}^{L} < 1$ and the $-$ sign in front means the cost will aways be positive
$y_{k}$ or $(1-y_{k})$ will be $0$ depending on whether the expected value for the particular neuron $k$ is $0$ or $1$
If the expected value is $0$ and the output $a_{k}^{L}$ is close to $0$, then the cost will be $-ln(1-a_{k}^{L})$ where $1-a_{k}^{L}$ is close to $1$, so the whole cost will be close to $0$
If the expected value is $0$ and the output $a_{k}^{L}$ is close to $1$, then the cost will be $-ln(1-a_{k}^{L})$ where $1-a_{k}^{L}$ is close to $0$, so the whole cost will be very high
Analogously, we can see that if the expected value is $1$ then the cost will be very low if the output is close to $1$ and very high if it's far away

Let's now calculate the error for a neuron in the output layer over a single training input if we're using the Cross-entropy cost function:
$$
\frac{\partial }{\partial a_{j}^{L}} \left({ \sum_{k} - \left[{ y_{k}ln(a_{k}^{L}) + (1-y_{k})ln(1-a_{k}^{L}) }\right] }\right) = \\
\frac{\partial }{\partial a_{j}^{L}}- \left[{ y_{1}ln(a_{1}^{L}) + (1-y_{1})ln(1-a_{1}^{L}) }\right] + \dots + \\
\frac{\partial }{\partial a_{j}^{L}}- \left[{ y_{j}ln(a_{j}^{L}) + (1-y_{j})ln(1-a_{j}^{L}) }\right] + \dots = \\
\frac{\partial }{\partial a_{j}^{L}}- \left[{ y_{j}ln(a_{j}^{L}) + (1-y_{j})ln(1-a_{j}^{L}) }\right] = \\
-\frac{\partial }{\partial a_{j}^{L}}  y_{j}ln(a_{j}^{L}) - \frac{\partial }{\partial a_{j}^{L}}(1-y_{j})ln(1-a_{j}^{L}) = \\
-\frac{y_{j}}{a_{j}^{L}} + \frac{1-y_{j}}{1-a_{j}^{L}} = \frac{a_{j}^{L}(1 - y_{j}) - y_{j}(1-a_{j}^{L})}{a_{j}^{L}(1-a_{j}^{L})} = \\
\frac{a_{j}^{L} - a_{j}^{L}y_{j} - y_{j} + a_{j}^{L}y_{j}}{a_{j}^{L}(1-a_{j}^{L})} = \frac{a_{j}^{L} - y_{j}}{a_{j}^{L}(1-a_{j}^{L})}
$$
Using the chain rule:
$$
\delta_{j}^{L} = \frac{\partial C}{\partial z_{j}^{L}} = \frac{\partial C}{\partial a_{j}^{L}} \cdot \frac{\partial a_{j}^{L}}{\partial z_{j}^{L}} = \frac{a_{j}^{L} - y_{j}}{a_{j}^{L}(1-a_{j}^{L})} \sigma'(z_{j}^{L}) = \\
\frac{\sigma(z_{j}^{L}) - y_{j}}{\sigma(z_{j}^{L})(1-\sigma(z_{j}^{L}))} \sigma(z_{j}^{L}) \left ({ 1 - \sigma(z_{j}^{L}) }\right ) = \\
\sigma(z_{j}^{L}) - y_{j}
$$
We can see that now the error will always be high if the difference between the output of the neuron and the expected output is high

## Softmax
Softmax is a type of output layer, where the activations form a probability distribution.
The weighted input is the same as for a sigmoid neuron:
$$
z_{j}^{L} = \sum_{k} w_{jk}^{L} a_{k}^{L-1} + b_{j}^{L}
$$
The activation for each neuron is:
$$
a_{j}^{L} = \frac{e^{z_{j}^{L}}}{\sum_{k}e^{z_{k}^{L}}}
$$
where in the denominator we sum over all the output neurons
So, the activations are always positive and they always sum to $1$. But, neurons with higher weighted input are interpreted as exponentially more likely

### Monotonicity of softmax
ToDo: calculate partial derivative of activation $a_{j}^{L}$ with respect to $z_{j}^{L}$ and $z_{k}^{L}$ where $k \ne j$ and show that the first is positive, while the second is negative.

### Non-locality of softmax
Sigmoid activations depend only on their own weighted input $a_{j}^{L} = \sigma(z_{j}^{L})$. Softmax activations on the other hand, depend on all weighted inputs because they are calculated as fraction of the sum of all activations:
$$
a_{j}^{L} = \frac{e^{z_{j}^{L}}}{\sum_{k} e^{z_{k}^{L}}}
$$

### Softmax inverse
$$
a_{j}^{L} = \frac{e^{z_{j}^{L}}}{\sum_{k} e^{z_{k}^{L}}} \\
e^{z_{j}^{L}} = a_{j}^{L} \sum_{k} e^{z_{k}^{L}} \\
z_{j}^{L} = ln \left ({ a_{j}^{L} \sum_{k} e^{z_{k}^{L}} } \right) \\
z_{j}^{L} = ln \left({ a_{j}^{L} }\right) + ln \left({ \sum_{k} e^{z_{k}^{L}} }\right) \\
z_{j}^{L} = ln \left({ a_{j}^{L} }\right) + C
$$
Where $C$ is the natural log of the sum of all activations, so it's independent of $j$

## Log-likelihood cost function
Let's define the log-likelihood cost function to be:
$$
C = - ln (a_{y}^{L})
$$
Where y is the index of the output neuron that is expected to be activated. For example, for an MNIST image of a 7, the activation $a_{7}^{L}$ should be high and the cost function is $- ln (a_{7}^{L})$

### Output layer error of softmax with log-likelihood cost function
ToDo: Find the error $\delta^{L}$, i.e. the derivative of $C$ with respect to each $z_{j}^{L}$ and then the derivative of $C$ with respect to a weight and bias of the output layer $w_{jk}^{L}$ and $b_{j}^{L}$
> Written with [StackEdit](https://stackedit.io/).
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU2ODcwNjE0OSwtOTkyMDc1MTgzLDE2OD
cxNDc5NTYsMjE4MjQ4MjQ2LC03NTMyNjQzMzAsLTE3NDc3Mjc1
MDgsLTUzNTYzODMyMiw5MjYyNjQ1NjldfQ==
-->