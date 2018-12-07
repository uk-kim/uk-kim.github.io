---
layout: post
title: Focal Loss for Dense Object Detection
tags: [Test, Markdown]
---

## Focal Loss for Dense Object Detection

![Focal Loss Paper](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-12-07-Focal-loss-for-dense-object-detection/focal_loss_paper.png?raw=true)

이 논문은 2017년 말에 공개된 논문으로써 기존의 Detection 알고리즘에서 분류기에서의 Loss Function으로 사용되던 CE(cross entropy)에 간단한 변형이 된 Focal Loss를 제안함으로써 분류 성능을 향상시는 방법을 소개합니다.

---

![Focal Loss](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-12-07-Focal-loss-for-dense-object-detection/ce_fl_compare_along_gamma.png?raw=true)

우리는 기존의 corss entropy에 $(1-p_t)^\gamma$ 텀을 추가한 <i>Focal Loss</i>라 명칭한 새로운 loss를 제안한다. $\gamma > .5$ 와 같이 설정하면 잘 분류가 된 $(p_t > .5)$ 샘플들에 대해서 상대적으로 loss 값을 감소시켜, 어렵고 잘못 분류된 샘플들에 대해서 모델이 더 집중할수 있도록 한다.

### Abstract
요즘 높은 정확도를 지닌 Object Detector들은 classifier가 sparse한 후보 영역을 생성하는 R-CNN 방식의 two-stage 기반의 방법으로 이루어진다. 반면에 one-stage detectors들은 dense detector라 불리며, 처리속도가 빠르지만 정확도가 낮은편이다.
우리는 dense detector 학습에서 전경과 배경이 극단적으로 불균형(imbalance)한 것이 주요 원인이라는 것을 알았다.
우리는 이러한 class imbalance한 문제를 보통의 cross entropy loss를 잘 분류된 샘플에 대해서 가중치를 낮추는 방식으로 변형하여 해결하는 방법을 제안한다.
Focal Loss는 적은 수의 hard examples이 주어진 상태에서 많은 easy negatives가 detector를 지배적으로 학습해버리는 상황을 예방하는 데에 중점을 두고자 한다.
우리는 Focal Loss의 효과를 평가하고자, RetinaNet이라 부르는 단순한 dense detector를 구축하여 학습하였다.


### 3. Focal loss
<i>Focal Loss</i>는 학습 중에서 전경과 배경이 극도로 imbalance(예, 1:1000)한 one-stage Object Detection 시나리오를 다루기 위해 설계되었다.

* cross entropy(CE) loss for binary classification
$$
CE(p, y)=\begin{cases}
-\log{(p)}, & \mathsf{if}\space y=1\\
-\log{(1-p)}, & \mathsf{otherwise.}
\end{cases}
$$

$y$는 Ground Truth(-1/1 또는 0/1)이고 $p$는 0~1 사이의 값을 지니며, $y=1$일 확률을 모델로부터 추정한 값이다.

$$
p_t=\begin{cases}
p, & \mathsf{if}\space y=1\\
1-p, & \mathsf{otherwise.}
\end{cases}
$$

$$CE(p,y)=CE(p_t)=-\log{(p_t)}$$

Fig.1을 보면, CE의 특징은 easily classified(잘 분류되는 샘플들, $p_t \gg .5$인 경우) 샘플들에 대해서도 Loss가 작지 않게 나타난다는 것이다. 즉, 잘 분류된 샘플들에 대해서도 loss가 꽤 크기 때문에 이들을 합치면 큰 값이 된다.
```
개인적인 소견
 확률이 높게 나타난 분류가 쉬운 샘플들도 loss가 크게 나타나기 때문에 이러한 경우에는 잘 분류된 샘플에 대해서도 에러를 줄이기 위한 학습이 많이 이루어 진다.
 즉 이 과정은 상대적으로 잘 분류된 샘플들 또한 모델이 학습하는데 주는 영향(가중치)가 크다고 생각할 수 있다.
 Focal Loss는 잘 분류된 샘플들에 대해서는(확률이 큰 샘플들) loss를 줄여 상대적으로 잘 분류가 안되는(또는 분류가 어려운) 샘플들에 대한 가중치를 상대적으로 높히는 형태로 디자인 되었다고 봄)
```

#### 3.1. Balanced Cross entropy
class imbalance 문제에 대한 일반적인 접근 방식은 class 1에 대해 가중치 $\alpha \in{[0, 1]}$를 적용하고 class -1에 대해 $1-\alpha$를 적용하는 것이다. $\alpha$ 결정은 반대 class의 빈도나 또는 corss validation을 통한 hyperparameter로써 결정한다. notation의 편의를 위해, $P_t$를 정의할 때 처럼 $\alpha_t$를 정의한다.
$\alpha$-balanced CE loss를 아래와 같이 작성한다.
$$
CE(p_t)=-\alpha_t \log{(P_t)}.
$$

#### 3.2. Focal Loss Definition
large class imbalance는 cross entropy loss가 dense detectors 학습 과정을 압도하면서 발생한다. 쉽게 분류된 negative 샘플들은 loss의 대부분을 이루고 있으며, gradient의 대부분을 지배하고 있다.

$\alpha$ 가 positive/negative 샘플의 중요도에 대해 균형을 잡기는 하지만, easy/hard 샘플에 대해서 구분하지는 않는다.
대신, 우리는 쉬운 샘플들에 대해서 가중치를 낮추는 형태로 loss function의 형태를 가공함으로써 어려운 샘플들에 대해서 집중하여 학습하는 방법을 제안한다.

우리는 tunable한 <i>focusing</i> 파라미터 $\gamma \geq 0$와 함께 조절 factor $(1-p_t)^\lambda$를 cross entropy loss에 추가하는 방법을 제안한다. focal loss는 다음과 같다.

$$
FL(p_t)=-(1-p_t)^\gamma \log{(p_t)}.
$$


focal loss의 두가지 속성은 다음과 같다.
<b>(1)</b> 어떤 샘플이 분류가 잘못되고, $p_t$가 작을 때, 조정 factor는 1에 가까운 값을 갖으며 losssms 거의 영향이 없다. 반면에 $p_t \rightarrow 1$일때에는 조정 factor가 0에 가까워 지고 잘 분류된 샘플에 대한 loss는 가중치가 작아지게 된다.
<b>(2)</b> focusing 파라미터인 $\gamma$는 easy 샘플들에 대한 가중치를 낮추는 정도를 somooth하게 조정한다. $\gamma = 0$일 때에는 FL은 CE와 같으며, $\gamma$가 증가하면 modulating factor의 효과가 증가하는 것과 같다.

직관적으로, 조정 factor는 easy 샘플들의 loss에 대한 영향을 줄이게하고, 샘플이 낮은 loss를 받는 구간을 확장시킨다. 예를 들어, $\gamma=2$이고 $p_t=0.9$로 분류된 샘플은 CE일때와 비교하여 100배 낮은 loss를 갖게되며, 그리고 $p_t \approx0.968$인 경우에는 100배 낮은 Loss를 갖게 된다. 이것은 잘못 분류된 샘플들에 대한 교정의 중요성을 증가시키게 한다($p_t \leq.5$, $\gamma=2$인 경우 loss는 최대 4배 까지 작아진다.).

### 4. RetinaNet Detector

RetinaNet은 <i>backbone</i> 네트워크와 두개의 task-specific <i>subnetworks</i>로 구성된 통합된 하나의 네트워크이다.
RetinaNet is a single, unified network composed of a backbone network and two task-specific subnetworks.

backbone 네트워크는 입력된 전체 이미지에 대해서 convolutional feature map을 계산하는 역할을 수행한다.
The backbone is responsible for computing a convolutional feature map over an entire input image.

첫번째 subnet은 backbone의 결과에서 convolutional 하게 object classification을 수행하는 단계이며, 두번째 subnet은 convolutional하게 bounding box를 추정하는 역할을 수행한다.

The first subnet performs convolutional object classification on the backbone's output; the second subnet performs convolutional bounding box regression.

![Focal Loss RetinaNet architecture](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-12-07-Focal-loss-for-dense-object-detection/retinanet_architecture.png?raw=true)

### 5. Experiments



![Focal Loss Experiments](https://github.com/uk-kim/uk-kim.github.io/blob/master/_posts/2018-12-07-Focal-loss-for-dense-object-detection/experiment_compair.png?raw=true)




a
a
a
a
a
a
a
a
a
