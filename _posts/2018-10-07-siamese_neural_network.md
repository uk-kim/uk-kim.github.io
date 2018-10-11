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
<br/>
![Siamese NN Strategy](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-10-07-siamese_nn/siamese_fig1_strategy.png?raw=true)
<br/>
## Deep Siamese Networks for Image Verification

샴 네트워크는 입력이 구분되는 쌍둥이 네트워크로 구성되어 있으며, 상단에서 에너지 함수로써 네트워크가 연결된다. 이 함수는 고차원의 feature representation 간의 metric으로 계산된다. 쌍둥이 네트워크를 구성하는 파라미터는 공유된다. 네트워크의 파라미터가 공유되기 때문에 유사한 입력 쌍에 대한 두 네트워크의 기능이 동일하기 때문에 차원 공간에서 각각 다른 위치로의 맵핑할 수 없게 한다.
<br/>
![Siamese NN Architecture1](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-10-07-siamese_nn/siamese_network_architecture_1.jpeg?raw=true)
<br/>
LeCun이 제안한 방법(2005, [1])에서 저자는 같은 쌍에 대해서는 에너지를 감소하고, 다른 쌍에 대해서는 에너지를 증가시키는 contrastive energy function을 사용하였다.(아래 식4)
<br/>
$$
Y = \{ \begin{matrix} 0,\quad if \space \vec{X_1} \space and \space \vec{X_2} \space are \space deemed\space similar \\ 1,\quad otherwise \end{matrix}
$$
일때,
<br/>
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
<br/>

즉, 두 입력 쌍 X1, X2가 네트워크를 통해 임배딩된 feature representation같의 유클리디안 거리가 similar 쌍일 때에는 작아지고, dis-similar 쌍일 경우에는 두 입력간의 거리가 최소 m보다 커지게끔 학습이 된다. (m은 margin을 나타내며, user가 설정하는 파라미터임)

이 논문에서는 입력 쌍에 대한 embeding된 feature vector에 sigmoid activation을 적용한 h1, h2에 대해 weighted L1 distance를 이용한다.

계산된 weighted L1 distance는 sigmoid activation 함수를 통해 prediction P가 계산된다
<br/>
#### prediction P with weighted L1 distance
$$
\mathbf{p} = \sigma (\sum{}_j \alpha_j \vert \mathbf{h}_{1, L-1}^{(j)} - \mathbf{h}_{2, L-1} ^{(j)} \vert)
$$
<br/>
#### Loss Function
$$
\mathcal{L} (x_1^{(i)}, x_2^{(i)}) = \mathbf{y} (x_1^{(i)}, x_2^{(i)}) \log{\mathbf{p} (x_1^{(i)}, x_2^{(i)})} + (1 - \mathbf{y} (x_1^{(i)}, x_2^{(i)})) \log {(1 - \mathbf{p} (x_1^{(i)}, x_2^{(i)}))} + \lambda^T \vert \mathbf{w} \vert ^2
$$
<br/>
![Siamese NN Architecture2](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-10-07-siamese_nn/siamese_network_architecture.jpeg?raw=true)
<br/>

## Experiments

#### The Omniglot Datasets
![Omniglot Datasets](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-10-07-siamese_nn/omniglot_dataset_characters.jpeg?raw=true)

Omniglot 데이터셋은 handwritten character 인식 분야에서 적은 샘플로 학습을 시키기 위한 표준 벤치마크  데이터셋이다. Omniglot 셋은 라틴어, 한글과 같이 잘 정립된 국제 언어의 50개 알파벳으로 이루어져 있다. 또한 Aurek-Besh, Klingon과 같은 가상의 문제 셋도 존재한다. 각 알파벳을 구성하는 문자의 수는 15~40개로 이루어져 있다. 이 알파벳들의 모든 문자들은 20명에 의해 각각 쓰여졌고, 40개의 알파벳으로 이루어진 background set(for training or validation), 10개의 알파벳으로 이루어진 evaluation set(for measure one-shot classification performance)으로 나누어 진다.

#### Verification

우리의 검증 네트워크를 학습시키기 위해, 우리는 같거나 다른 30000, 90000, 15000개의 다른 크기의 3 데이터 셋을 합친다. 우리는 60%를 학습용 데이터로 두었으며, 이는 50개의 알파벳 중 30개의 알파벳으로, 20명의 작성자 중 12명으로 이루어져 있다.

#### One-shot Learning
학습된 Siamese Neural Network로 추출한 feature가 discriminative 문제에 대해 잠재력이 있다고 본다. 우리는 C개의 클래스 중의 하나의 클래스에 대응하는 테스트 이미지 x가 있다고 가정한다. 이제 우리는 테스트 이미지 x와 각 클래스의 이미지 x_c(c=1, …, C)를 네트워크에 입력한 다음, 최대 유사성에 해당하는 클래스를 예측한다.

$$
C^\ast = \arg\max{}_c \mathbf{p} ^{(c)}
$$
One-shot learning의 performance를 평가하기 위해 evaluation set에서 무작위로 일정하게 선택한 20개의 문자를 2명의 drawer에서 선택한다.

첫번째 drawer가 생성한 문자는 테스트용, 두번째 drawer의 문자는 검증용으로, 첫번째 drawer의 각 문자는 두번째 drawer의 모든 문자와 비교하여 해당하는 클래스를 예측하는데 사용된다. 이 과정은 두번째 drawer를 기준으로도 수행되며 이 전체 과정을 반복하여 최종적으로 성능을 평가한다.

![Omniglot Datasets](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-10-07-siamese_nn/siamese_omniglot_evaluation.png?raw=true)

#### MNIST One-shot Trial
Omniglot 데이타 셋은 클래스의 수가 각 클래스의 instance의 수보다 많기 때문에 “MNIST transpose”라고 불린다. 우리는 Omniglot 데이터셋으로 학습된 모델의 일반화 성능을 모니터링 하기 위해 10개의 알파벳으로 이루어진 MNIST 셋으로 평가하는 것을 흥미롭게 생각하였다. 10way One-shot classification 문제를 MNIST 데이터셋으로 수행한 결과는 아래와 같다.

![mnist oneshot result](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-10-07-siamese_nn/siamese_mnist_oneshot_result.png?raw=true)

성능이 매우 높지는 않지만,  10개 클래스에서 랜덤으로 선택했을 때의 정확도는 10%, 1-NN으로 했을 때의 정확도가 26.5%인것을 감안하면 MNIST 도메인으로 학습하지 않은 모델을 통한 70.3%는 유의한 결과라고 볼수 있음.

## Conclusions

우리는 기존의 다른 baselines에 비해 우수한 성능을 입증하였다. 이 네트워크의 강력한 성능은 인간 수준의 정확성을 우리의 metric 기반의 접근법으로 가능할 뿐만 아니라, 다른 one-shot learning 영역(특히, image classification)에까지 확장될 수 있다.

***

## 고찰

이 논문에서는 기존의 일반적인 classification 문제에서 각 클래스에 해당할 확률로써 표현하던 방법을 유사도로 표현하는 것으로 대체하여 해결한다. 입력 데이터와 이에 대한 대조군을 동시에 입력함으로써, 얼마나 유사한가?를 embedding된 feature의 거리로써 표현하는 것이다.

즉, 이 과정은 동일한 클래스에 해당하는 데이터는 high-level의 공간에서 가까운 곳에 embedding 되게끔, 다른 클래스는 먼 거리에 embedding 되게끔 하는 것을 목적으로 학습이 진행된다. 따라서 기존의 softmax 등을 활용하여 하는 학습들은 학습 데이터에 종속되어 overfitting이 발생하며, 타 도메인에 적용이 어려운 반면에, 위와같은 metric 기반의 방법은 비교 쌍 간의 거리를 조정하기 위한 형태로 학습을 하기 때문에 적은 데이터 만으로도 좋은 feature를 학습시키는 것이 가능하게 된다.

또한 이러한 classification 방법은 기존의 n-class classification과 달리 기존에 학습되지 않은 새로운 클래스에 대해서도 적용이 가능하다는 장점이 있다. 즉, 비교 대상 클래스 수에 제한이 없다는 것이다.

따라서 이러한 metric 기반의 방법은 얼굴인식과 같은 분야에 많이 적용되고 있다. 그리고 similarity를 계산한다는 점에서 object tracking 분야[2]에서도 적용이 되고 있다.
<br/>

Email: [seongukzzz@gmail.com](mailto:seongukzzz@gmail.com)