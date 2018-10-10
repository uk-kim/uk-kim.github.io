---
layout: post
title: Siamese Neural Network for One-shot Image Recognition (샴 네트워크)
tags: [Siamese Neural Network, One-shot Learning, Metric Learning]
---
![Siamese NN Paper](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-10-07-siamese_nn/siamese_paper_intro.png?raw=true)
[paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

## Introduction
ML 영역에서 좋은 Feature를 학습시키는 과정은 많은 연산량을 요구한다. 또한 <u>주어진 데이터의 수가 적은 상황에서의 Feature 학습은 매우 어렵다.</u> One-Shot Learning은 각 class별 1개의 data 만으로도 정확한 prediction이 가능하게 하는 것이다. 이 논문에서는 Siamese Neural Network를 통해서 입력 데이터들간의 Similarity 순위를 매기는 구조를 차용한다. 학습 후에는 새 데이터 뿐만 아니라 알지 못하는 분포의 새로운 class 전체에 대해서도 일반화 성능이 뛰어난 예측력을 지닌다.

<b>One-shot learning</b> is an object categorization problem in computer vision. Whereas most machine learning based object categorization algorithms <u>require training on hundreds or thousands of images and very large datasets</u>, <b>one-shot learning</b> <u>aims to learn information about object categories from one, or only a few, training images.</u> ([wikipedia](https://en.wikipedia.org/wiki/One-shot_learning))


## Approach
우리는 샴 네트워크(siamese Network)를 통해 supervised Metric 기반의 image representation을 학습하고, 학습된 네트워크를 별도의 재학습 없이 One-shot Learning에 사용한다.

One-shot classification model의 학습을 위해 image pair들의 class-identity를 구분하는 NN을 학습하는 것을 목표로 하였다.

![Siamese NN Strategy](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-10-07-siamese_nn/siamese_fig1_strategy.png?raw=true)

## Deep Siamese Networks for Image Verification

샴 네트워크는 입력이 구분되는 쌍둥이 네트워크로 구성되어 있으며, 상단에서 에너지 함수로써 네트워크가 연결된다. 이 함수는 고차원의 feature representation 간의 metric으로 계산된다. 쌍둥이 네트워크를 구성하는 파라미터는 공유된다. 네트워크의 파라미터가 공유되기 때문에 유사한 입력 쌍에 대한 두 네트워크의 기능이 동일하기 때문에 차원 공간에서 각각 다른 위치로의 맵핑할 수 없게 한다.

![Siamese NN Architecture1](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-10-07-siamese_nn/siamese_network_architecture_1.jpeg?raw=true)

LeCun이 제안한 방법(2005, [1])에서 저자는 같은 쌍에 대해서는 에너지를 감소하고, 다른 쌍에 대해서는 에너지를 증가시키는 contrastive energy function을 사용하였다.(아래 식4)

$$
Y = \{ \begin{matrix} 0,\quad if \space \vec{X_1} \space and \space \vec{X_2} \space are \space deemed\space similar \\ 1,\quad otherwise \end{matrix}
$$
일때,

$$
D_W(\vec{X_1}, \vec{X_2}) = \Vert G_W(\vec {X_1}) - G_W (\vec {X_2}) \Vert_2 \quad \quad \quad \quad \quad\quad \quad \quad\quad\quad\quad(1)
$$

$$ \mathcal{L}(W) = \sum_{i=1}^{P} {L(W,(Y,\vec{X_1},\vec{X_2})^i)} \quad \quad \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad(2)
$$

$$
L(W,(Y, \vec{X_1},\vec{X_2})^i) = (1 - Y) L_s (D^i_W) + Y L_D (D^i_W) \quad \quad\quad\quad\quad\quad\quad\quad(3)
$$

$$
L(W, Y, \vec {X_1}, \vec {X_2}) = (1-Y)\frac 1 2 (D_W)^2 + (Y) \frac 1 2 \{max(0, m - D_W)\}^2 \quad\quad (4)
$$

즉, 두 입력 쌍 X1, X2가 네트워크를 통해 임배딩된 feature representation같의 유클리디안 거리가 similar 쌍일 때에는 작아지고, dis-similar 쌍일 경우에는 두 입력간의 거리가 최소 m보다 커지게끔 학습이 된다. (m은 margin을 나타내며, user가 설정하는 파라미터임)

이 논문에서는 입력 쌍에 대한 embeding된 feature vector에 sigmoid activation을 적용한 h1, h2에 대해 weighted L1 distance를 이용한다.

계산된 weighted L1 distance는 sigmoid activation 함수를 통해 prediction P가 계산된다

##### prediction P with weighted L1 distance
$$
\mathbf{p} = \sigma (\sum{}_j \alpha_j \vert \mathbf{h}_{1, L-1}^{(j)} - \mathbf{h}_{2, L-1} ^{(j)} \vert)
$$

##### Loss Function
$$
\mathcal{L} (x_1^{(i)}, x_2^{(i)}) = \mathbf{y} (x_1^{(i)}, x_2^{(i)}) \log{\mathbf{p} (x_1^{(i)}, x_2^{(i)})} + (1 - \mathbf{y} (x_1^{(i)}, x_2^{(i)})) \log {(1 - \mathbf{p} (x_1^{(i)}, x_2^{(i)}))} + \lambda^T \vert \mathbf{w} \vert ^2
$$

![Siamese NN Architecture2](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-10-07-siamese_nn/siamese_network_architecture.jpeg?raw=true)



Email: [seongukzzz@gmail.com](mailto:seongukzzz@gmail.com)