---
layout: post
title: Attention 메카니즘
tags: [Attention, bi-directional model, seq2seq, ratsgo blog]
---
### Attention 메카니즘


이번 글은 Attention 메카니즘에 대해서 간략하게 설명하는 내용을 담으려고 한다. 글의 내용은 [ratsgo blog-어텐션 매커니즘](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/10/06/attention/)을 보면서 공부한 내용을 다시 작성한 것으로, 해당 블로그의 글을 토대로 공부하면서 다시 정리하며 기록하였다.

참고 동영상 : https://www.youtube.com/watch?v=6aouXD8WMVQ

##### 개요
어텐션(attention) 매커니즘은 딥러닝 모델이 특정 벡터에 주목하게 만들어 모델의 성능을 높이는 기법이다. 기계번역을 위한 sequence-to-sequence(seq2seq) 모델에서 처음 도입되었다. seq2seq 모델은 아래 그림과 같이 연속된 시퀀스 데이터가 입력되어 벡터로 만들고(이 과정을 Encoder라고 함), 이 벡터로부터 시퀀스의 출력하는 과정(이 과정을 Decoder라고 함)으로 구성되어있다. 그런데 이러한 seq2seq 모델은 인코더, 디코더의 길이가 길어질수록 모델의 성능이 나빠지는 경향이 있다. 오래된 시점의 정보를 잊거나 또는 특정 단어 W를 예측할 때 A, B, C 모두에 대해 집중하여 보게 되면 정확도가 떨어질 수 있기 때문이다. 따라서 어텐션 메카니즘은 디코더의 어떤 단어를 예측할 때, 인코더의 정보에서 "현재 상태에서 중요한 부분만을 집중(attention)하게 만들자"가 핵심 아이디어이다.

![seq2seq model](https://i.imgur.com/6mbfPZR.png?raw=true)

예를 들어, 인코더의 입력으로 "Ich mochte ein bier"를 디코더에서 "I'd like a beer"로 번역하는 seq2seq 모델을 만든다고 하자. 모델이 인코더의 4번째 단어 "bier"에 대해 예측하고자 할 때, 모델은 인코더에서 주어진 시퀀스 데이터에 대한 embeded된 벡터에서 "bier"에 대해 주목하게 만들고자 하는 것이다. 즉, 어텐션 메카니즘에서의 가정은 <b>인코더가 'bier'를 받아서 벡터로 만든 결과(인코더 출력)는 디코더가 '맥주'를 예측할 때 쓰는 벡터(디코더 입력)와 유사할 것</b>이라는 것이다.

##### Encoder 계산 과정
아래 그림은 양방향(bi-directional) 모델을 사용한 seq2seq 모델의 인코더 영역이다. $$i$$번째 벡터 $$x_i$$가 입력되어 상태벡터(hidden state vecotr) $$h_i$$를 만든 후, $$h_i$$가 $$i$$번째 열벡터가 되도록 오른쪽과 같이 만들어 이를 행렬 $$F$$라고 정의한다.

![seq2seq encoder(bi-directional)](https://i.imgur.com/CbQjPWo.png?raw=true)

##### Decoder 계산 과정
어텐션 메커니즘을 통해 "bier"에 해당하는 인코더 출력 벡터와 "beer"에 해당하는 디코더의 입력 벡터의 유사도를 높게 하고자 한다. $$e_{ij}$$는 디코더가 $$i$$번째 단어를 예측할 때 쓰는 직전 스텝의 히든스테이트 벡터 $$s_{i-1}$$이 인코더의 $$j$$번째 열벡터 $$h_j$$와 얼마나 유사한지를 나타내는 스코어(스칼라)이다.
$$
e_{ij}=a(s_{i-1}, h_j)
$$
이 식에서 $$a$$는 원 논문에서 alignment model이라 소개된다. $$s_{i-1}$$과 $$h_j$$간의 유사도를 잘 나타낼수 있다면 다양한 변형이 가능하다.
$$e_{ij}$$에 Softmax 함수를 적용하여 합이 1이 되는 확률로 변환한다. 이때 $$T_x$$는 디코더 입력 단어의 수를 가리킨다.

$$ \alpha_{ij} = \frac {exp(e_{ij})} {\sum_{k=1}^{T_x} exp(e_{ik})} $$

디코더가 $$i$$번째 단어를 예측할 때 쓰이는 attention vector $$a_i$$는 다음과 같이 정의한다.
$$ \vec{\alpha_i} = [\alpha_{i1}, \alpha_{i2}, ... , {\alpha_{iT_x}}]$$

여기서 $$ \alpha_{ij} $$는 디코더의 $$i$$번째 단어를 예측할 때 인코더의 $$j$$번째 히든스테이트 벡터에 대해 얼마나 집중할 것인가를 나타낸 가중치로 해석가능하다. 디코더가 i번째 단어를 예측할 때 쓰이는 context vector $$c_i$$는 다음과 같이 정의되며, 인코더의 $$j$$번째 어텐션 확률값으로 가중합을 한 것이라고 볼 수 있다.
$$ \vec{c_i}=\sum_{j=1}^{T_x} {\alpha_{ij}h_j}=F\vec{\alpha_i} $$

##### Decoder의 계산 예시
디코더에서의 계산 과정은 아래 그림과 같다. alignment model $$a$$는 디코더가 2번째 단어 'like'를 예측할 때 쓰이는 첫번째 히든스테이트 벡터 $$s_i$$과 가장 유사한 인코더의 열벡터가 $$h_2$$라고 판단했다. 디코더가 2번째 단어를 예측할 때 쓰이는 attention vector $$\alpha_2$$를 보면 두번째 요소값이 가장 높기 때문이다.

![decoder calc example](https://i.imgur.com/4zdzDKL.png?raw=true)

디코더가 2번째 단어를 예측하는데 사용되는 context vector $$c_2$$는 인코더 출력벡터들로 구성된 행렬 $$F$$에 대해 $$\alpha_2$$를 내적하여 계산한다. 디코더 모델은 타겟 단어벡터(I'd)와 $$c_2$$를 concat하여 현시점의 히든스테이트 벡터 $$s_i$$를 만든다.

##### 요약
기존의 Seq2Seq 문제에서는 입력 Sequence가 인코더에 입력되어 시퀀스의 길이만큼 은닉 상태벡터 $$h_t$$가 생성된다. 그리고 마지막 은닉 상태벡터는 다시 디코더의 초기 상태 벡터로 사용되어 출력 시퀀스를 구하게 된다. 이 과정은 은닉 상태에 시퀀스 정보를 함축하여 그 정보로부터 변형된 다른 시퀀스를 출력하는 과정이라고 생각할 수 있다. 하지만 이러한 방식은 인코더, 디코더의 시퀀스 길이가 길어질수록 정보를 소실하는 현상이 발생할 수 있게 되기 때문에
