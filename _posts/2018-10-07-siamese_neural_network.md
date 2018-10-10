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

$$$ y = 3 $$$
$$$ Y = \{ \begin{matrix} {0, if \quad X_1\quad and\quad X_2\quad are\quad deemed\quad similar} \\ {1, otherwise} \end{matrix}

$$$

y = \left\{ \begin{matrix} aa \\ bb \end{matrix} \right

$$$

Y = 0 if X1 and X2 are deemed similar,  else Y = 1
1
![lecun loss function_1](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-10-07-siamese_nn/siamese_lecun_loss_function_1.jpeg.jpeg?raw=true)

2
![lecun loss function_2](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-10-07-siamese_nn/siamese_lecun_loss_function_2.jpeg.jpeg?raw=true)



Email: [seongukzzz@gmail.com](mailto:seongukzzz@gmail.com)