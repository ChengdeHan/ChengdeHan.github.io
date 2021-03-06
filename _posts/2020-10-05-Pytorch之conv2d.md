---
layout: mypost
title: Pytorch之CONV2D
categories: [Pytorch]
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
# CONV2D

## CLASS

`torch.nn.Conv2d`(**in_channels**: int, **out_channels**: int, **kernel_size**: Union[T, Tuple[T, T]], **stride**: Union[T, Tuple[T, T]] = 1, **padding**: Union[T, Tuple[T, T]] = 0, **dilation**: Union[T, Tuple[T, T]] = 1, **groups**: int = 1, **bias**: bool = True, **padding_mode**: str = 'zeros')

## Parameters

- **in_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of channels in the input image
- **out_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of channels produced by the convolution
- **kernel_size** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)) – Size of the convolving kernel
- **stride** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)*,* *optional*) – Stride of the convolution. Default: 1
- **padding** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)*,* *optional*) – Zero-padding added to both sides of the input. Default: 0
- **padding_mode** (*string**,* *optional*) – `'zeros'`, `'reflect'`, `'replicate'` or `'circular'`. Default: `'zeros'`
- **dilation** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)*,* *optional*) – Spacing between kernel elements. Default: 1
- **groups** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – Number of blocked connections from input channels to output channels. Default: 1
- **bias** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If `True`, adds a learnable bias to the output. Default: `True`

## Shpae

- 输入：$（N,C_{in},H_{in},W_{in}）$      $N$ 是 batch size，$C_{in}$ 是输入的通道数量， $H_{in}$  是输入的二维信号的高度，$W_{in}$是输入的二维信号的宽度。  

- 输出：$（N,C_{out},H_{out},W_{out}）$      

  $
  H_{out}=\lfloor  \frac{H_{in}+2\times padding[0]-dilation[0]\times (kernel\_size[0]-1)-1}{stride[0]}+1   \rfloor 
  $
  
  $
  H_{out}=\lfloor  \frac{H_{in}+2\times padding[0]-dilation[0]\times (kernel\_size[0]-1)-1}{stride[0]}+1   \rfloor 
  $
  
  $N$ 是 batch size，$C_{out}$ 是输出的通道数量， $H_{out}$  是输出的二维信号的高度，$W_{out}$是输出的二维信号的宽度。  

## Variables

- **Conv2d.weight** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor))  模型的需要学习的权重的形状大小为：$（out\_channels,\frac{in\_channels}{groups},kernel\_size[0],kernel\_size[1]）$. 该权重的值的话是从均匀分布$\cal{U}(-\sqrt{k},\sqrt{k})$采样得到的，其中$k=\frac{groups}{C_{in}\times kernel\_size[0]\times kernel\_size[1]}$.

- **conv2d.bias**([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) 模型的需要学习的偏置的形状大小为 $out\_channels$. 若 bias是True 的话，那么其值是从均匀分布$\cal{U}(-\sqrt{k},\sqrt{k})$采样得到的，其中$k=\frac{groups}{C_{in}\times kernel\_size[0]\times kernel\_size[1]}$.

## 自我理解

二维卷积可以认为是对多通道的图像做的卷积。

<img src="conv2d.jpg" alt="conv2d" style="zoom:33%;" />

输出的维度是out\_channels，其卷积操作是out\_channels个 kernel\_size[0]*kernel\_size[1] 大小的卷积核分别按步长等参数对整个图像做填充。
