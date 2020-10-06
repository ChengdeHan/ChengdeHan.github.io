---
layout: mypost
title: (SIG19)Dynamic Graph CNN for Learning on Point Clouds
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
# (SIG19)Dynamic Graph CNN for Learning on Point Clouds

## 1  Contributions

- We present **a novel operation** for learning from point clouds, ==**EdgeConv**==, to **better capture local geometric features** of point clouds **while** still **maintaining permutation invariance**.
- We show the model can **learn to semantically group points** by ==**dynamically updating a graph of relationships**== from layer to layer.
- We demonstrate that **EdgeConv can be integrated into multiple existing pipelines** for point cloud processing.
- We present extensive analysis and testing of EdgeConv and show that it **achieves state-of-the-art performance** on benchmark datasets.

## 2  Method

### 2.1 Overview

本文最大创新点就是==**EdgeConv**==和==**dynamically updating a graph of relationships**== 。

1. We propose an approach inspired by PointNet and convolution operations. Instead of working on individual points like PointNet, however, we **exploit local geometric structures** by **constructing a local neighborhood graph** and **applying convolution-like operations on the edges connecting neighboring pairs of points**, in the spirit of graph neural networks. We show in the following that such an operation, dubbed edge convolution (EdgeConv), has properties lying between **translation-invariance** and **nonlocality**.

   ![image-20200921174936257]((SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921174936257.png)

   Fig. 2. Left: Computing an edge feature, $e_{ij}$(top), from a point pair,  $x_{i}$and $x_{j}$(bottom). In this example,  $h_{\varTheta}()$ is instantiated using a fully connected layer, and the learnable parameters are its associated weights. Right: The **EdgeConv** operation. The output of EdgeConv is calculated by aggregating the edge features associated with all the edges emanating from each connected vertex.

2. Unlike graph CNNs, our **graph** is **not fixed** but rather is **dynamically updated after each layer of the network**. That is, the set of k-nearest neighbors of a point changes from layer to layer of the network and is computed from the sequence of embeddings. ==**Proximity in feature space** differs from **proximity in the input**, **leading to nonlocal diffusion** of information throughout the point cloud.==

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921193550802.png" alt="image-20200921193550802" style="zoom: 80%;" />

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921193619272.png" alt="image-20200921193619272" style="zoom: 57%;" />

Fig. 4. Structure of the feature spaces produced at different stages of our shape classification neural network architecture, visualized as the distance between the red point to the rest of the points. For each set, **Left:** **Euclidean distance** in the input $\Bbb{R}^{ 3}$space; **Middle:** Distance after **the point cloud transform stage**, amounting to a global transformation of the shape; **Right:** Distance in the **feature space** of the last layer. Observe how in the feature space of deeper layers semantically similar structures such as shelves of a bookshelf or legs of a table are brought close together, although they are distant in the original space.

![image-20200921194542430]((SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921194542430.png)

Fig. 3. Model architectures: The model architectures used for **classification (top branch)** and **segmentation (bottom branch)**.

### 2.2  Edge Convolution

Consider an F-dimensional point cloud with n points, denoted by $X = \{\textbf{x}_{1}, . . . , \textbf{x}_{n}\} ⊆ \Bbb{R}^{F}$ . In the simplest setting of F = 3, each point contains 3D coordinates $\textbf{x}_{i}=\{x_{i},y_{i},z_{i}\}$;

![image-20200921174936257]((SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921174936257.png)

Fig. 2. Left: Computing an edge feature, $e_{ij}$(top), from a point pair,  $x_{i}$and $x_{j}$(bottom). In this example,  $h_{\varTheta}()$ is instantiated using a fully connected layer, and the learnable parameters are its associated weights. Right: The **EdgeConv** operation. The output of EdgeConv is calculated by aggregating the edge features associated with all the edges emanating from each connected vertex.



**EdgeConv** 模块的具体步骤：

1. 首先，以$n\times f$ 的一个张量作为该模块的输入。

2. 一个直接的图为 $\cal{G}=\{\cal{V},\cal{E}\}$, $\cal{V}=\{1,...,n\}$为图结构中的顶点，$\cal{E}\subseteq \cal{V} \times \cal{V}$ 为相应的边。 在$X = \{\textbf{x}_{1}, . . . , \textbf{x}_{n}\} ⊆ \Bbb{R}^{F}$ 中，对每个点使用 k-nearest neighbor (k-NN)创建 $\cal{G}$ 。对点$\textbf{x}_{i}$，构建的相应图的边以$(i,j_{i_{1}}),(i,j_{i_{2}}),...,(i,j_{i_{k}})$来进行组织，使其$\textbf{x}_{j_{i_{1}}},\textbf{x}_{j_{i_{2}}},...,\textbf{x}_{j_{i_{k}}}$ 刚好为离$\textbf{x}_{i}$逐近到远的点。

3. 对于每个点，融合一个局部特征, $（\textbf{x}_{i},\textbf{x}_{i}-\textbf{x}_{j_{i_{1}}}）,（\textbf{x}_{i},\textbf{x}_{i}-\textbf{x}_{j_{i_{2}}}）,...,（\textbf{x}_{i},\textbf{x}_{i}-\textbf{x}_{j_{i_{k}}}）$. This explicitly **combines global shape** **structure**, captured by the coordinates of the patch centers $\textbf{x}_{i}$ , **with local neighborhood informatio**n, captured by $\textbf{x}_{i}-\textbf{x}_{j_{i_{k}}}$. 然后根据 $（\textbf{x}_{i},\textbf{x}_{i}-\textbf{x}_{j_{i_{1}}}）,（\textbf{x}_{i},\textbf{x}_{i}-\textbf{x}_{j_{i_{2}}}）,...,（\textbf{x}_{i},\textbf{x}_{i}-\textbf{x}_{j_{i_{k}}}）$通过MLP层，学习到相应的edge features $e_{ij}$（$a_{n}$维的特征） 。

   <img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921205213481.png" alt="image-20200921205213481" style="zoom:80%;" />

4. 对于以上得到的对于点$\textbf{x}_{i}$的 k 个edge features $e_{ij}$ （$a_{n}$维的特征），使用一个max 操作，输出一个pooling后的 feature。

    <img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921205704974.png" alt="image-20200921205704974" style="zoom:67%;" />

5. 如此，通过该层之后输入的为一个$n \times f$ 的一个特征，输出的便是一个融合了局部信息和非局部信息（在特征空间进行EdgeConv体现）的特征$n \times a_{n}$。

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921195511072.png" alt="image-20200921195511072" style="zoom: 50%;" />

### 2.3  Dynamic Graph Update

==Our experiments suggest that **it is beneficial to recompute the graph** **using nearest neighbors in the feature space** produced by each layer.== This is a crucial distinction of our method from graph CNNs working on a fixed input graph. Such a dynamic graph update is the reason for the name of our architecture, the Dynamic Graph CNN (DGCNN). ==**With dynamic graph updates**, **the receptive field is as large as the diameter of the point cloud**, **while being sparse**.==

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921211131066.png" alt="image-20200921211131066" style="zoom:80%;" />

Fig. 4. Structure of the feature spaces produced at different stages of our shape classification neural network architecture, visualized as **the distance between the red point to the rest of the points**. For each set, **Left**: **Euclidean distance** in the input R3 space; **Middle:** Distance after the point cloud **transform stage**, amounting to a global transformation of the shape; **Right:** Distance in the **feature space** of the **last layer**. Observe how in the feature space of deeper layers semantically similar structures such as shelves of a bookshelf or legs of a table are brought close together, although they are distant in the original space.

### 2.4  Properties

- **Permutation Invariance.** 

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921213122786.png" alt="image-20200921213122786" style="zoom: 80%;" />

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921213137858.png" alt="image-20200921213137858" style="zoom:80%;" />

- **Translation Invariance.**

  <img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921213350954.png" alt="image-20200921213350954" style="zoom:80%;" />

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921213324475.png" alt="image-20200921213324475" style="zoom:80%;" />

## 3  Classification

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921214829511.png" alt="image-20200921214829511" style="zoom: 80%;" /><img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921214205354.png" alt="image-20200921214205354" style="zoom:80%;" />

![image-20200921215037583]((SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921215037583.png)

![image-20200921215614581]((SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921215614581.png)

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921215500467.png" alt="image-20200921215500467" style="zoom:80%;" />

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921215719204.png" alt="image-20200921215719204" style="zoom:80%;" />

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921215940696.png" alt="image-20200921215940696" style="zoom:80%;" />

## 4  Part Segmentation

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921220017864.png" alt="image-20200921220017864" style="zoom:80%;" />

![image-20200921220154959]((SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921220154959.png)

## 5 Indoor Scene Segmentation

<img src="(SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921220411491.png" alt="image-20200921220411491" style="zoom: 67%;" />

![image-20200921220458872]((SIG19)Dynamic Graph CNN for Learning on Point Clouds_chengdehan.assets/image-20200921220458872.png)





