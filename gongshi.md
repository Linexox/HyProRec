$$
\begin{align}
  \pi &= \textbf{Softmax} ( 
  		\textbf{MLP} ( [u^{‘txt}; u^{img}; u^{vdo}; u^{ado} ])
  ) \\
  s   &= \sum_{ v\in\mathcal M } {\pi^v u^v}
\end{align}
$$

$$
\begin{align}\pi &= \textbf{Softmax} ( \textbf{MLP} ( [u^{‘txt}; u^{img}; u^{vdo}; u^{ado} ])) \\s   &= \sum_{ v\in\mathcal M } {\pi^v u^v}\end{align}
$$

$$
 \mathcal L_{view}^v = CE(y, s^v)
$$

$$
\mathcal L_{fuse} = CE(y, s)
$$

$$
\mathcal L_{rec} = \mathcal L_{fuse} + \eta \sum_{v \in \mathcal M}\mathcal L_{view}^v
$$

$$
\mathcal L = \mathcal L_{rec} + \lambda \mathcal L_{conv}
$$

$$
\mathcal L = \mathcal L _{rec} + \eta \sum_{v \in \mathcal M}{\mathcal L_{view}^v} + \lambda \mathcal L_{conv}
$$


$$
\begin{align}
\mathcal C &= [\text{Soft Prompt}; \text{Conversation History}; \text{HoT Prompt}]
\\ 
\mathcal L_{conv}
&=
-\frac1{|\mathcal T|}
\sum_{t\in\mathcal T}
\log
P(w_{t}\mid w_{<t},\mathcal C)
\end{align}
$$

$$
s'w
$$
