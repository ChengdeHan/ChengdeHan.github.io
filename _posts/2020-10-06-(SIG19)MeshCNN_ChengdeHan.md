---
layout: mypost
title: MeshCNN: A Network with an Edge
categories: [PaperNote]
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
      inlineMath: [['$','$']]
    }
  });
</script>
# MeshCNN: A Network with an Edge

## 1.Contributions

- we **present MeshCNN**: a neural network akin to the well-known CNN, but **designed specifically for meshes**. MeshCNN operates **directly on irregular triangular meshes**, performing convolution and pooling operations designed in harmony with the unique mesh properties.

- We choose to work with edges since every edge is incident to exactly two faces (triangles), which defines a natural fixed-sized **==convolutional==** neighborhood of four edges. We utilize **the consistent face normal order** to apply **a symmetric convolution operation**, which learns edge features that are invariant to transformations in rotation, scale and translation.

- In CNNs, pooling downsamples the number of features in the network, thereby learning to eliminate less informative features. Since features are on the edges, an intuitive approach for **downsampling** is to use the well-known mesh simplification technique **edge collapse** [Hoppe 1997]. However, **unlike conventional edge collapse**, which removes edges that **introduce a minimal geometric distortion**, ==**mesh pooling**== **delegates the choice of which edges to collapse to the network** in a **task-specific** manner. **The purged edges** are the ones whose **features contribute the least to the used objective**.

- To illustrate the aptitude of our method, we **perform a variety of experiments** on **shape classification** and **segmentation tasks** and demonstrate superior results to state-of-the-art approaches on common datasets and on highly non-uniform meshes.

## 2.Overview of CNN

**卷积神经网络一般由卷积层、汇聚层和全连接层构成．**

<img src="image-20200904085859786.png" alt="image-20200904085859786" style="zoom:80%;" />

### 2.1.卷积层

卷积层的作用是提取一个局部区域的特征，不同的卷积核相当于不同的特征提取器。

<center class="half">    <img src="image-20200904085139595.png" width="450"/>    <img src="(SIG19)MeshCNN_ChengdeHan.assets/image-20200904085158893.png" width="450"/> </center>

- **局部连接。**在卷积层（假设是第𝑙 层）中的每一个神经元都只和下一层（第𝑙 − 1层）中某个局部窗口内的神经元相连，构成一个局部连接网络．如图5.5b所示，卷积层和下一层之间的连接数大大减少，由原来的𝑀𝑙 × 𝑀𝑙−1 个连接变为𝑀𝑙 × 𝐾个连接，𝐾 为卷积核大小．

- **权重共享。**从公式(5.22) 可以看出，作为参数的卷积核𝒘(𝑙) 对于第𝑙 层的所有的神经元都是相同的．如图5.5b中，所有的同颜色连接上的权重是相同的．权重共享可以理解为一个卷积核只捕捉输入数据中的一种特定的局部特征．因此，如果要提取多种特征就需要使用多个不同的卷积核。

### 2.2.汇聚层

汇聚层（Pooling Layer）也叫子采样层（Subsampling Layer），其作用是进行特征选择，降低特征数量，从而减少参数数量．

<img src="image-20200904090115204.png" alt="image-20200904090115204" style="zoom: 67%;" />

## 3.Overview : apply CNN on meshes 

Realizing our goal to **apply the CNN paradigm directly onto triangular meshes**, **necessitates** an **analogous definition** and **implementation of the standard building blocks** of CNN: **==the convolution and pooling layers==**.

Accordingly, we design our network to deliberately **apply** convolution and pooling operations **directly on the constructs of the**
**mesh**, and **avoid conversion to a regular and uniform representation**.

Such an assumption guarantees that **each edge is incident to two faces (triangles)** at most, and is therefore **adjacent to either two or four other edges**. The vertices of a face are **ordered counterclockwise**, defining two possible orderings for the four neighbors of every edge. For example, see Figure 4, where **the 1-ring neighbors of e can be ordered as (a,b, c,d) or (c,d, a,b)**, depending on which face defines the first neighbor. This **ambiguates** the convolutional receptive field, **hindering** the formation of invariant features.

<img src="image-20200904100052850.png" alt="image-20200904100052850" style="zoom:50%;" />

We **take two actions** to **address** **`this issue`** and **guarantee** **`invariance to similarity transformations`** (rotation, translation and scale) within our network.

-  First, we carefully design the input descriptor of an edge to contain only relative geometric features that are inherently
  invariant to similarity transformations. 

- Second, we aggregate the four 1-ring edges into two pairs of edges which have an ambiguity (e.g., a and c, and b and d), and generate new features by applying simple symmetric functions on each pair (e.g., sum(a, c)). The convolution is applied on the new symmetric features, thereby eliminating any order ambiguity.

### 3.1.Input features.

 The input edge feature is a 5-dimensional vector for every edge: the dihedral angle, two inner angles and two edge-length ratios for each face.   三个维度是如下图蓝色的角度，另外两个分别是下图中该edge的长度与其相应两个面的上的垂线的长度比率。

We sort each of the two face-based features (inner angles and edge-length ratios), thereby **resolving the ordering ambiguity** and **guaranteeing invariance**.  Observe that **these features are all relative**, **making them invariant to translation, rotation and uniform scale**.  由于角度特征无关于顺序，只要角度相同，顺序不同还是代表相同的特征。边长的比率可以消除缩放的特征影响。

<img src="image-20200904100539026.png" alt="image-20200904100539026" style="zoom: 33%;" />



### **3.2.Global ordering.** 

**The global ordering of the edges** is the order in which **the edge data (input features) of a particular shape enters the network**. **This ordering** has **no influence** **during the convolutional stage** since **convolution is performed within local neighborhoods**. By extension, **fully convolutional tasks** e.g., segmentation **are unaffected by it**. For tasks that **require global feature aggregation**, such as classification, we **follow** the common practice suggested by Qi et al. [2017a] in **PointNet**, and **place a global average pooling layer** that connects between the convolutional part and the fully-connected part of the network. **This layer renders the initial ordering inconsequential and thus guarantees invariance to transformations**.

### 3.3.**Pooling.** 

**Mesh pooling is accomplished by an edge collapse process**, as illustrated in Figure 2 (b) and (c). In (b), the dashed edge
is collapsing to a point, and, subsequently, the four incident edges (blue) merge into the two (blue) edges in (c). Note that in this edge collapse operation, five edges are transformed into two. **The operator** is **prioritized by the (smallest norm) edge features**, thereby **allowing the network to select which parts of the mesh to simplify**, and **which to leave intact**. This creates a **task-aware process**, where the network learns to determine object part importance with respect to its mission (see Figure 1).

![image-20200904104101321](image-20200904104101321.png)

**A notable advantage** of **the nature of our simplification**, is that it **provides flexibility** with **regards to the output dimensions of the pooling layer**, just before it reaches the final fully connected layer. Pooling also contributes to **robustness** **to initial mesh triangulation**. While it does **not provide equivariance to triangulation**, in practice, by way of continuously collapsing edges and simplifying the mesh, we **observe convergence to similar representations despite differences in initial tessellation**.

<img src="image-20200904104625395.png" alt="image-20200904104625395" style="zoom:80%;" />

## 4.Methods

<img src="image-20200904112601568.png" alt="image-20200904112601568" style="zoom: 67%;" />

### 4.1.Mesh Convolution

We define a convolution operator for edges, where the spatial support is defined using the four incident neighbors (Figure 3). Recall that convolution is the dot product between a kernel k and a neighborhood, thus the convolution for an edge feature e and its four adjacent edges is:
$$
e\cdot k_{0}+\sum^{4}_{j=1} k_{j}\cdot e^{j}
$$
where $e_{j}$ is the feature of the $j$-th convolutional neighbor of $e$.

<img src="image-20200904100052850.png" alt="image-20200904100052850" style="zoom:50%;" />

To guarantee **convolution invariance to the ordering of the input data**, we apply a set of **simple symmetric functions** to **the ambiguous pairs**. This generates **a new set of convolutional neighbors** that are **guaranteed to be invariant**. In our setting, the receptive field for an edge $e$ is given by
$$
(e^{1}, e^{2}, e^{3},e^{4})=(|a-c|,a+c,|b-d|,b+d)
$$
This leads to a convolution operation that is oblivious to **the initial ordering of the mesh elements**, and will therefore **produce the same output regardless of it**.

### 4.2. Mesh Pooling

We **extend conventional pooling to irregular data**, by identifying **three core operations** that together **generalize the notion of pooling**:

- define pooling region given adjacency.

- merge features in each pooling region.

- redefine adjacency for the merged features.

Mesh pooling is another special case of generalized pooling, **where adjacency is determined by the topology**. Unlike images,
which have a natural reduction factor of, for example 4 for 2 × 2 pooling, we define mesh pooling as a series of edge collapse operations, where **each such edge collapse converts five edges into two**.

Therefore, we **can control the desired resolution of the mesh** after each pooling operator, by adding a hyper-parameter which **defines the number of target edges in the pooled mesh**. During runtime, extracting mesh adjacency information requires querying special data structures that are continually updated (see [Berg et al. 2008] for details).

We **prioritize** the **edge-collapse order** (using a priority queue) by **the magnitude of the edge features**, allowing the network to **select which parts of the mesh are relevant to solve the task**. This enables the network to **non-uniformly collapse certain regions** which are least important to the loss.

Edge collapse is prioritized according to the strength of the features of the edge, which is taken as their ℓ2-norm. The features are aggregated as illustrated in Figure 5, where there are two merge operations, one for each of the incident triangles of the minimum edge feature e, resulting in two new feature vectors (denoted p and q). The edge features in channel index i for both triangles is given by
$$
p_{i} = avg(a_{i},b_{i},e_{i}), and , q_{i} = avg(c_{i},d_{i},e_{i})
$$
<img src="image-20200904132645326.png" alt="image-20200904132645326" style="zoom:80%;" />

**After edge collapse**, the **half-edge data structure** is **updated for the subsequent edge collapses**.
Finally, **note that not every edge can be collapsed**. An **edge collapse yielding a non-manifold face is not allowed in our setting**, as it violates the four convolutional neighbors assumption. Therefore, **an edge is considered invalid** **to collapse** if it has three vertices on the intersection of its 1-ring, or if it has two boundary vertices.

## 5.EXPERIMENTS

### <img src="image-20200904134604548.png" alt="image-20200904134604548" style="zoom:80%;" />

<img src="image-20200904134635190.png" alt="image-20200904134635190" style="zoom:80%;" />

<img src="image-20200904134704655.png" alt="image-20200904134704655" style="zoom:80%;" />

### 5.1 Data Processing

![image-20200904135540778](image-20200904135540778.png)

![image-20200904135608442](image-20200904135608442.png)

### 5.2 Mesh Classification

