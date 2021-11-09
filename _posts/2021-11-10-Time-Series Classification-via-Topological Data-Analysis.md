---
title: [번역] Time Series Classification via Topological Data Analysis
tags: Mathematics
mathjax: true
---

# Time Series Classification via Topological Data Analysis

Author: Yuhei Umeda
Translator : 이성헌 (POSTECH MINDS)
Subject: ML, TDA
URL : https://www.jstage.jst.go.jp/article/imt/12/0/12_228/_pdf
# Summary

 이 논문은 변동성 시계열(volatile time series)에 대한 분류 문제에 초점을 맞춘다. 

시계열 분류에 대한 가장 인기 있는 접근 방식 중 하나는 Dynamic time warping(DTW) 및 feature-based 머신러닝 아키텍처이다. 

이전의 많은 연구에서 이러한 알고리즘은 다양한 데이터 세트에서 만족스럽게 수행되었다. 

그러나 측정된 값의 표면적 변화가 Chaotic한 시계열에 필수적인 것은 아니기 때문에 이러한 방법의 대부분은 Chaotic한 시계열에 적합하지 않다. 

일반적으로 대부분의 시계열 데이터 세트에는 chaotic한 시계열과 non-chaotic한 시계열 모두 포함되어 있으므로, 시계열의 더 본질적인 feature를 추출해야 한다. 

본 논문에서, 우리는 변동성 시계열 분류를 위한 새로운 접근법을 제안한다. 

우리의 접근 방식은 시계열의 전이 규칙(transition rule)을 나타내기 위해 Topological Data Analysis(TDA)를 사용하여 어트랙터(attractor)의 구조를 추출하여 새로운 feature를 생성한다. 

이 특징은 시계열 시스템의 본질적인 feature를 나타내므로, 우리의 접근 방식은 chaotic한 유형과 non-chaotic한 유형 모두에 효과적이다.

우리는 Convolutional Neural Network(CNN)에서 영감을 받은 학습 아키텍처를 이 기능에 적용했고 제안된 접근 방식이 기존 접근 방식에 비해 인간 활동 인식 문제의 성능을 18.5% 향상시킨다는 것을 발견했다.

# 1. Introduction

시계열 분류는 공학 및 의학과 같은 많은 분야에서 중요한 문제이다 [Aggarwal 14]. 

시계열 분류를 위한 몇 가지 알고리즘이 이미 제안되었으며 좋은 성능을 제공할 수 있었다.

가장 일반적인 접근방식은 `Dynamic time warping`(DTW)으로, 시간 축을 뒤틀림으로써 두 시계열 사이의 유사성을 측정한다 [Rakthanmanon 12].

이 접근법은 `k-Nearest Neighbor method`(k-NN)과 결합할 때 특히 효과적이다.

그러나 DTW 알고리즘은 측정 타이밍의 간극의 영향을 받으며 변수 사이의 상관관계에 대한 선험적 지식이 없으면 다채널 알고리즘으로 확장하기가 어렵다[Banko § 12].

이러한 맥락에서 feature-based 머신러닝 접근법이 활발하게 연구되었다.

이 접근법은 시계열에서 추출한 feature들의 경계에 fit 되도록 하는 모델을 사용한다.

`Support Vector Machine`(SVM) 및 `Artificial Neural Network`(ANN)와 같은 기존 머신 러닝 알고리즘은 이런 분류기를 생성하는 데 광범위하게 사용된다. 

표준적인 feature들에는 최대/최소값, 평균값, 분산 및 빈도 정보와 같은 통계적 feature들이 포함된다.

알툰 외 연구진과 바르샨 외 연구진. [Altun 10a, Barshan 14]는 인간 활동 인식을 위해 이러한 feature들을 사용하는 다양한 기계 학습 알고리즘을 조사했고 최고의 성능을 제공하는 분류기를 식별했다.

최근, 자연어 처리(Native Language Processing, NLP)의 기술에서 영감을 받은 `bag-of-words` 및 `bag-of-features`과 같은 local segment 정보를 사용하는 features의 새로운 프레임워크가 제안되었다[Baydogan 13, Baydogan 15, Wanga 13].

이러한 테크닉들은 많은 시계열 데이터 세트에서 DTW 접근 방식을 능가할 수 있다.

또한 딥 러닝은 데이터 세트의 feature들을 자동으로 추출하는 학습 모델군으로 부상했다.

딥러닝에서는 feature design을 자동화하기 위해 다중 레이어로 구성된 심층 아키텍처가 구축된다.

`Convolutional Neural Network`(CNN)은 이미지 인식 분야에서 특히 주목할 만한 성능을 달성했다.

양 외 연구진. [양 15]는 CNN 아키텍처를 시계열 분류에 직접 적용했고 다양한 시계열 데이터 세트에서 DTW 접근 방식에 비해 성능을 개선할 수 있었다.

왕 외 연구진. [Wang 15]는 CNN 아키텍처를 이미지 분류에 적용하기 위해 시계열을 이미지로 변환하는 방법을 제안했다.

이 알고리즘은 일부 시계열 데이터 세트에서도 우수한 성능을 제공했다.

위의 feature들은 시계열 값의 파형을 기반으로 한다.

즉, 동일한 클래스에 속하는 시계열들은 그 파형과 범위가 유사하다.

그러나 동일한 클래스에 속하지만 파형과 값 범위는 다른 성분을 포함하는 시계열 데이터 세트도 많다.

예를 들어, 동일한 시스템 내의 chaotic한 시계열은 초기 조건에 예민하다. 

이러한 상황에서 결정론적 비선형계(deterministic nonlinear system)의 한 상태의 작은 변화가 이후의 상태에 큰 차이를 초래할 수 있다.[Sprott 03].

이러한 민감도의 결과로 통계적 feature들도 민감도를 가진다.

 따라서 이러한 chaotic한 시계열에 대한 위의 feature들은 분류 문제에 적합하지 않다.

혼돈 시계열 분류의 경우 알리 외 [Ali 07]은 chaos theory에 기초한 feature들을 제안했으며, 동작 센서 신호의 분류 문제에서 통계적 feature와 비교하여 성능을 향상시켰다.

그러나 이러한 feature의 대부분은 확정적인 값들(determinate values)이기 때문에 non-chaotic 데이터 세트에는 유용하지 않다.

chaotic한 시계열은 다양한 시계열 데이터 세트에 포함된다.

예를 들어, 사회, 경제, 의학, 국가 간 관계 [Levy 94], 뇌파 [Korn 03], 인간 활동의 센서 신호 [Ali 07] 및 많은 영역의 시계열에는 chaotic한 시계열이 포함된다.

대부분의 분야에서 chaotic한 시계열과 non-chaotic한 시계열은 혼재해 있기 때문에 이 둘을 구분하기 어렵다.

이것이 chaotic한 시계열과 non-chaotic한 시계열 모두에 호환되는 분류 알고리즘을 생성해야 하는 이유다.

본 논문에서 우리는 chaotic한 시계열을 포함하고 있는 짧은 구간 내에서 상하 운동을 수행하는 시계열(volatile time series)의 분류문제를 고려한다.

특히 chaotic한 시계열과 non-chaotic 시계열 모두에 효율적인 새로운 feature를 제안한다.

우리의 개념은 transition rule (time evolution equation)의 기저에 의해 feature들이 생성된다는 아이디어를 기반으로 한다.

우리는 우리의 개념을 구현하기 위해 attractor를 사용한다. 

이는 동역학계에서 하나의 figure [Kantz 03, Sprott 03]에 의해 transition rule을 표현하는 공통된 접근 방식이다.

머신러닝 아키텍처를 attractor에 적용하려면 먼저 attractor를 아키텍처에 적합한 형태로 변환해야 한다.

분류 측면에서, attractor의 가장 중요한 요소는 figure의 위상적 정보이다.

따라서, 우리는 데이터 세트에서 위상적 정보를 추출하는 프레임워크를 사용하여 `Topological Data Analysis`(TDA)[Carlsson 09]을 기반으로 변환 방법을 개발했다.

우리는 주요 TDA 기법인 `persistent homology`[Edelsbrunner 08]을 사용하여 이 정보를 추출하고 `Betti sequence`라고 이름 붙인 시계열 분류를 위한 새로운 feature를 생성한다.

페레아 외 연구진[Perea 14, Perea 15]은 attractor의 유사성에 `persistent homology`를 적용하여 시계열의 주기성 정보를 추출했다.

`Betti sequence`는 시계열 분류 측면에서 표준 벡터 및 표준 시계열과 다른 몇 가지 특성이 있으므로 CNN을 기반으로 betti sequence에 적합한 학습 아키텍처를 소개한다.

우리의 기여는 attractor에 persistent homology를 적용하여 시계열의 feature를 추출하는 것과 CNN을 사용하여 그 feature들을 분류하는 것을 포함한다.

검증을 위해 인간 활동 인식 문제에 우리의 알고리즘을 적용하고 기존 알고리즘과 비교한다.

이 논문의 나머지 부분은 다음과 같이 구성되어 있다.

먼저, 우리는 2장에서 알고리즘의 개요를 제공한다.

다음으로, 우리는 알고리즘의 주요 도구, 즉 attractor 이론, topological data analysis 및 CNN 구조를 각각 3장, 4장 및 6장에 소개한다.

5장에서는 인공 데이터를 이용한 전처리의 효과를 확인한다.

실험 결과와 분석은 7장에 제시되어 있다.

우리는 8장에서 미래의 작업에 대한 간략한 요약과 언급으로 마무리한다.

# 2. Our Classification Algorithm

1. Preprocessing Part
1.1 Convert TSD to 'quasi-attractor'
1.2 Convert quasi-attractor to 'Betti sequence'
2. Learning Part
2.1 Construct a classifier based on a 1-D CNN using the Beti sequence dataset.

![Untitled](Time%20Series%20Classification%20via%20Topological%20Data%20An%20be717522bb804f75a2a6e33909ac615b/Untitled.png)

이 장에서는 그림 1과 같이 시계열 분류 알고리즘의 개요를 개략적으로 설명한다.

우리의 분류 알고리즘은 전처리 및 학습 부분으로 구성된다.

시계열 데이터 세트를 기계 학습 알고리즘에 적용하기에 적합한 새로운 데이터 세트로 변환하는 사전 처리 부분은 두 단계로 구성된다.

먼저, 시계열을 해당 시스템의 transition rule을 나타내는 quasi-attractor로 변환한다. (3장 참조)

다음으로 우리는 quasi-attractor의 위상적 정보를 추출하여 생성된 betti sequence라는 새로운 형태로 준 어트랙터를 변환한다. (4장 참조)

학습 파트에서는 betti sequence 데이터 세트를 사용하여 1차원 CNN을 기반으로 분류기를 구성한다(6장 참조).

# 3. Analysis of Dynamical Systems.

이전의 많은 연구에서 시계열 분류를 위한 다양한 feature들을 제안했다.

가장 일반적인 기능은 최대/최소값, 평균값,빈도 정보와 같은 통계량이다([Altun 10a, Barshan 14] 참조).

local segment 정보를 사용하는 feature의 일부 프레임워크가 최근에 제안되었다 [Baydogan 13, Wanga 13].

이러한 기능은 특정 패턴을 반복하는 시계열, 특히 non-chaotic한 시계열에 효과적이지만 무작위 패턴이 있는 chaotic 시계열에 대한 이러한 기능의 효과는 미미하다.

chaotic한 시계열 분류의 경우, 예를 들어 Lyapunov exponenet 같은 chaos 특유의 일부 feature들이 성공적으로 사용되었다[Basharat 09].

그러나 이러한 feature들은 non-chaotic한 시계열에 대해서는 차별점을 생성하지 않는다.

 기존의 통계적 feature를 활용하는 방법들 ([Altun], [Barshan], [Baydogan], [Wanga])는 특정한 패턴이 반복되는 TSD에 효과적임. 특히 Non-chaotic한 데이터에 효과적이었음. 그러나 Chaotic한 문제들에는 별루였음.

일반적으로 chaotic과 non-chaotic을 구별하는 것은 어렵고 관측된 시계열 데이터 세트는 chaotic/non-chaotic 시계열이 혼재한다.

따라서 chaotic과 non-chaotic을 구별할 필요성을 없애는 것이 바람직하다.

## 3.1 Difference Equation and Attractor

본 논문에서 우리는 관찰된 시계열 $\left\{x_1,\ldots,x_t\right\}$ ($x_i \in \R$)은 다음과 같이 정의되는 difference function을 가진다고 가정한다.

$$x_{k+1} = f(x_k,\ldots,x_1)\quad \quad (1)$$

일반적으로 chaotic한 상황뿐 아니라 자연에서 관측된 많은 non-chaotic 시계열의 전이 규칙을 방정식 (1) 로 나타낼 수 있다. [Basharat 09]

우리의 접근 방식의 원래 아이디어는 이 방정식을 기반으로 시계열 데이터 세트를 분류하는 것이다.

그러나 관측된 시계열 데이터에서 이 방정식을 구성하는 것은 매우 어렵다.

따라서 우리는 비선형 동역학계의 분석에 사용되는 `attractor`를 활용한다. [Basharat 09]

`Attractor`는 $d$-차원 공간에 내장된 시스템의 다양한 초기 조건에 대해 (1)에 의해 설명되는 시스템이 변화하는 경향이 있는 수치 값들의 집합이다[Kantz 03].

일반적으로 많은 시계열 데이터 세트에는 같은 전환 규칙을 가질 것임에도 다른 파형을 가지는 몇몇 시계열들이 포함된다.

대조적으로, 유사한 전환 규칙을 가진 시계열의 매력들은 서로 닮았다.

`Attractor`의 개념은 근본적인 chaotic 시스템이 모델링되는 기초를 형성한다.

그러나 `Attractor`는 difference equation (1)의 모양을 형성하기 때문에 difference equation을 갖는 chaotic 시스템뿐만 아니라 non-chaotic system의 특징을 나타낸다.

따라서 `Attractor`는 전환 규칙을 사용하는 시계열 분류에 적합하다.

## 3.2 Quasi-attractor

`Attractor`는 시스템 값들의 전이가 수렴하는 궤도이기 때문에 일반적으로 무한개의 점을 가진다.

관측된 시계열 데이터로부터 길이가 유한한 `Attractor`를 생성하는 것은 불가능하다.

더군다나 `Attractor`가 매장된 공간의 차원을 찾기도 어렵다.

관측된 시계열로부터 `Attractor`의 정보를 얻는 가장 유명한 방법은 `Quasi-attractor`를 사용하는 방법이다.

시계열 관측 $\left\{x_0,x_1,\ldots,x_t\right\}$를 `delay embedding`을 통해 phase space vector $\bf{Z} = \left\{z_0,z_1,\ldots,z_{t'}\right\}$ ($t' = t-(p-1)\tau +1)$로 변환하는 것이다.

delay vector는 다음으로 정의된 시계열의 local information으로 생성된 벡터다.

$$\textbf{z}_{k} = [x_k,x_{k+\tau},\ldots,x_{k+(p-1)\tau}] \in \R^p \quad\quad (2)$$

여기서 $\tau$는 sampling lag 이고, $p$는 embedding할 차원이다.

`Quasi-attractor`는 delay vector의 집합을 나타낸다.

delay vector로부터 얻은 `Quasi-attractor`의 예는 그림 2에 제시되어 있다.

적절한 embedding dimension은 시계열 데이터에 따라 다르다.

그러나 시계열 분류의 경우 다른 설정을 사용하여 정보를 변환하는 데는 적합하지 않다.

따라서 본 논문에서는 매개 변수를 $\tau = 1$ 및 $p=3$으로 상수 값을 할당한다.

![Untitled](Time%20Series%20Classification%20via%20Topological%20Data%20An%20be717522bb804f75a2a6e33909ac615b/Untitled%201.png)

`Quasi-attractor`는 point cloud 데이터로 얻어진다.

`Quasi-attractor`의 핵심 정보는 점들의 배열이다.

# 4. Topological Data Analysis

quasi-attractor에는 두 가지 주요 특성이 있다.
첫째, quasi-attractor는 point cloud 형태로 생성되므로 희소 데이터로 구성된다.

둘째, 점 구름의 좌표 값은 아무런 의미가 없으므로 점 구름의 배열의 중요성이 강조된다.
따라서 phase space $\bf{Z}$의 포멧은 머신러닝 아키텍처의 입력으로 사용하기에 적합하지 않다.

quasi-attractor의 분류는 point cloud 배열의 위상적 정보 추출을 필요로 한다.

이를 위해 TDA의 주요 기술인 persistent homology를 사용하는 결합 방법을 제안한다.

TDA는 지난 10년간 빠르게 성장하고 있는 응용 수학의 비교적 새로운 분야이다. 

TDA는 데이터 셋에서 위상적 정보를 추출하는 프레임 워크를 제공하며, 이는 특정 거리(metric)에 민감하지 않으며, 차원 감소 및 소음에 대한 견고성을 제공한다는 점에서 성공을 거뒀다.

## 4.1 Persistent Homology

`Persistent homology`는 포인트 클라우드 데이터의 위상적 정보를 구성하기 위한 전략을 제공한다. 

간단히 말해서, homology는 point cloud 구조에서 "구멍"을 발견한다.

`Persistent homology`의 기본 아이디어는 데이터 세트 $\bf{Z} = \left\{z_i\right\}_{i=1}^m$의 각 점을 중심으로 하는 반지름이 $\epsilon >0$인 closed ball $B(z_i,\epsilon)$의 확장이  배치되는 공간 $\mathbb{X}_\epsilon = \cup_{i=1}^m B(z_i,\epsilon)$의 homology의 족을 구성하는 것이다.

반지름 매개변수 $\epsilon$가 고정되어 있을 때, 우리는 0차원 구멍(= 점/연결 요소에 homeomorphic), 1차원 구멍(= 원에 homeomorphic), 2차원 구멍(= 구에 homeomorphic)과 더 높은 차원의 구멍의 조합으로 $\mathbb{X}_\epsilon$의 구조를 얻을 수 있다.

`Persistent homology`는 반지름 $\epsilon$에 따라 각 차원 구멍의 탄생과 사망에 대한 정보를 제공한다.

여기서 두 topological sapce $X$와 $Y$가 `homeomorphic`하다는 것은 다음 세 조건을 만족시키는 $X$와 $Y$ 사이의 함수가 존재할 때를 말한다. (1) 전단사함수 (2) 연속함수 (3) 역함수도 연속

그림 3은 포인트 클라우드의 공간을 커버하는 지속성의 예를 보여준다.

![Untitled](Time%20Series%20Classification%20via%20Topological%20Data%20An%20be717522bb804f75a2a6e33909ac615b/Untitled%202.png)

`Persistent homology`는 반지름 매개변수의 변화에 따른 구멍 수의 변화를 추적하여 point cloud 모양의 특징을 추출하는 방법이다[Carlsson 09].

이는 `Persistent homology`를 사용하여 attractor의 특징을 추출함으로써 시계열의 역학 정보를 추출할 수 있게 해준다.

`Persistent homology`의 수학적 정의는 [Carlsson 09]를 참조하라.

지속적인 호몰로지는 최근 센서 네트워크 및 단백질 분류와 같은 광범위한 영역에 적용되고 있다.

시계열 영역에서 Perea 등은 [Perea 14, Perea 15]에서 시계열의 주기를 추출하기 위해 적용하였다.

본 논문에서, 우리는 시계열의 transition rule을 나타내는 attractor 모양의 특징을 추출하기 위해 `Persistent homology`를 사용한다.

## 4.2 Feature of Time Series from Persistent Homology

`Persistent homology`의 고전적인 출력은 `Persistent Diagram`과 `Persistent Barcode`이다.

`Persistent diagram`은 평면에 점의 모임을 그려서 작성할 수 있다.

두 개의 좌표를 가진 점으로서 구멍이 탄생하는 반지름과 구멍이 사라지는 반지름의 쌍으로 구성된 확장 평면 $(\R\cup \left\{\infty\right\})^2$을 생각해 보자.

클래스 중 일부는 절대 죽지 않을 수 있으며 무한대에 위치한 점으로 표현된다.

`Persistent barcode`는 본질적으로 $[a,b)$ 꼴의 interval들의 multiset이며, 여기서 $a$와 $b$는 각각 탄생하고 및 사라질 때의 반지름이다.

하나의  point cloud에 대하여, 각 차원의 구멍의 `Persistent diagram`들과 `Persistent barcode`가 생성된다.

그림 4는 point cloud에서 나오는 1차원 구멍의 `Persistent diagram` 및 `Persistent barcode`의 예를 보여준다.

![Untitled](Time%20Series%20Classification%20via%20Topological%20Data%20An%20be717522bb804f75a2a6e33909ac615b/Untitled%203.png)

`Persistent diagram`과 `Persistent barcode`는 중요한 정보를 제공합니다.

그러나 `Persistent diagram`은 이미지 데이터로 사용하기에는 너무 sparse하고 `Persistent barcode`의 구성 요소 수는 일정하지 않기 때문에 대부분의 머신러닝 테크닉의 입력 데이터로는 적합하지 않다.

따라서 우리는 머신러닝 입력에 대한 새로운 형태의 `Persistent homology`의 출력을 제안한다.

`Persistent homology`의 중요한 점은 반지름 파라미터의 변화에 따른 구멍 수의 변화를 따르는 것이다.

따라서 본 논문에서는 데이터에 대해 통합된 설정으로 반지름 매개 변수를 보존하는 벡터화 방법을 채택한다.

이 새로운 형태는 확장된 closed ball의 반지름과 해당 반지름에 있는 구조의 구멍 수(`Betti number`)에 대응한다.

$\mathbb{X}_\epsilon$의 $d$-차원 구멍들의 수 ($d$-dimensional Betti number)를 $BN_d(r)$이라 표기하자.

유한한 길이의 벡터를 생성하기 위해 우리는 반지름 매개 변수를 $0<r<E$로 제한한다. 

여기서 $E$는 유한한 값을 가지는 하이퍼파라미터다.

이 데이터는 벡터들 $BS_0, BS_1, \ldots ,BS_n$을 오름차순으로 연결하여 생성된다.

여기서 $n$은 사용하고자 하는 homology dimension의 최댓값으로 주어진다.

각 벡터들 $BS_d$의 $i$번째 성분은 $\mathbb{X}_{i*E/m_d}$의 $d$-dimensional Betti number 수이다.

다시 말해, $BS_d(i) = BN_d(i*E/m_d)$이다.

여기서 $m_d$는 $BS_d$의 이산화 mesh size (vector size)로 주어진다.

각 $d$에 대하여 $\left\{m_d\right\}_{d=0}^n$이 모두 같을 필요는 없다.

이 논문에서는 모든 $d$에 대하여 $m_d=300$의 공통된 값을 설정한다.

그림 5는 `Persistent barcode`에서 이 논문에서 `Betti sequence`라고 부르는 형태의 입력 데이터로의 변환을 보여준다.

`Betti sequence`의 벡터 길이는 $M=m_0+\cdots +m_n$이다.

![Untitled](Time%20Series%20Classification%20via%20Topological%20Data%20An%20be717522bb804f75a2a6e33909ac615b/Untitled%204.png)

# 5. Synthetic Data

## 5.1 Synthetic Data

이 장에서는 다음 인공 데이터를 사용하여 제안된 전처리 알고리즘의 효과를 확인한다.

$$\textbf{Sin-I} \begin{cases}x_{k+1} = -0.56x_k - x_{k-1}, \\ x_0 = 0.5, x_1 = 0.9998,\end{cases}\\

\textbf{Sin-II}\begin{cases}x_{k+1} = -0.56x_k - x_{k-1}, \\ x_0 = 0.5, x_1 = 0.6999,\end{cases}\\

\textbf{Sin-III}\begin{cases}x_{k+1} = -0.56x_k - x_{k-1}, \\ x_0 = 0.5, x_1 = 0.966,\end{cases}\\

\textbf{Chaos-I}\begin{cases}y_{k+1} = 3.97y_k(1 - y_{k}),y_0=0.5 \\ x_k = 2.2y_k - 1.1,\end{cases}\\

\textbf{Chaos-II}\begin{cases}y_{k+1} = 3.97y_k(1 - y_{k}),y_0=0.2 \\ x_k = 2.2y_k - 1.1.\end{cases}\\$$

Sin-I, Sin-II 및 Sin-III 데이터는 주기적인 파동의 진폭과 주기에 대한 영향을 확인하기 위해 준비했다.

이 데이터들은 다음 함수들의 difference equation이다:

$$\textbf{Sin-I} \quad x_k =0.5\sin(1.6k)+0.5,\\

\textbf{Sin-II} \quad x_k =0.2\sin(1.6k)+0.5,\\

\textbf{Sin-III} \quad x_k =0.5\sin(1.2k)+0.5.\\$$

이 difference equation들은 대응하는 미분방정식의 이차도함수를 근사하여 구한다.

예를 들어, $\textbf{Sin-I}$의 경우 stride $\Delta k = 1$인 central difference를 사용하면 $d^2x_k/dk^2=x_{k+1} -2x_k+x_{k-1}$을 얻고, 이로부터  $d^2x_k/dk^2 = -1.6^2x_k$ 임을 얻는다.

$\textbf{Sin-I}$와 $\textbf{Sin-III}$를 비교하면 주기의 영향을 알 수 있는 반면, $\textbf{Sin-I}$와 $\textbf{Sin-II}$를 비교하면 진폭의 영향을 확인할 수 있다.

데이터 $\textbf{Chaos-I}$  및 $\textbf{Chaos-II}$는 가장 잘 알려진 chaotic 시계열 중 하나인 로지스틱 맵으로부터 생성된다.

이 두 시계열은 초기 조건의 민감도로 인해 파형이 다르지만 time evolution equation에 기초한 분류 측면에서 동일한 클래스에 포함되어야 한다.

## 5.2 Preprocessing Synthetic Data

![Untitled](Time%20Series%20Classification%20via%20Topological%20Data%20An%20be717522bb804f75a2a6e33909ac615b/Untitled%205.png)

그림 6은 인공 데이터의 파형 (a), Quasi-attractor (b) 및 Betti sequence (c)를 보여준다.

그림 6(a)에서 왼쪽 그림은 진폭과 주기의 파형 차이를 보여준다.

이러한 차이는 local segment 정보로는 구별하기 어렵다.

오른쪽 그림은 chaotic 시계열의 파형을 보여준다.

이러한 파형은 상당히 다르므로 local segment 정보와 통계 정보를 사용해도  $\textbf{Chaos-I}$ 과  $\textbf{Chaos-II}$ 를 동일한 클래스로 분류하기는 어렵다.

그림 6(b)는 인공 데이터의 Quasi-attractor를 보여준다.

그림 6(b)의 $\textbf{Sin-I}$, $\textbf{Sin-II}$, 그리고 $\textbf{Sin-III}$로부터 우리는 시계열의 진폭과 빈도의 차이가 고리의 반지름과 각도의 차이로 표현된다는 것을 관찰할 수 있다.

게다가, Chaotic 데이터의 모양은 거의 비슷하다.

 따라서 우리는 모양을 기준으로 이 데이터를 분류할 수 있다.

그림 6(c)는 인공 데이터의 Betti sequence를 보여준다.

$\textbf{Sin-I}$, $\textbf{Sin-II}$, 그리고 $\textbf{Sin-III}$의 Betti sequence를 통해 Betti sequence가 시계열의 진폭과 주기의 차이를 나타낼 수 있음을 확인할 수 있다.

또한 동일한 evolution equation과 다른 파형을 가진 시계열의 Betti sequence가 그림 6(c)의 $\textbf{Chaos-I}$ 과  $\textbf{Chaos-II}$ 와 유사한 형태를 갖는 것을 관찰할 수 있다.

# 6. Learning Architecture

앞 장에서는 시계열 분류 문제를 다음과 같은 두 가지 특성을 가진 베티 시퀀스에 대한 분류 문제로 변환하는 것에 대해 논의하였다.

첫 번째 특성은 `Betti sequence`의 반지름 파라미터(벡터의 셀 번호)가 시계열 측정이 시작된 지점에 의존하지 않는다는 것이다.

반면에, 시계열의 시간 파라미터는 일반적으로 측정 시작점에 따라 달라지기 때문에 `Betti sequence`는 일반 시계열과는 다르다.

예를 들어, 측정 시작 시간이 다른 동일한 시계열로부터 생성된 벡터는 서로 다르다.

이런 성질은 `Betti sequence`로의 변환을 수행하기로 한 우리의 결정을 입증하는 이유 중 하나이다.

두 번째 특성은 진폭이 다른 시계열의 `Betti sequence`의 차이가 반지름 파라미터 방향의 간격을 나타낸다는 것이다.

- ***Proposition 6-1)***
    
    다른 진폭을 가지는 두 시계열 데이터 $x_t$와 $\tilde{x}_t = ax_t$ ($a>0$), 그리고 $d=0,1,\ldots,n$에 대하여, $x_t$로부터 얻은 closed ball space $\mathbb{X}_\epsilon$의 $d$-dimensional Betti numbers $BN_d(r)$과 $\tilde{x}_t$로부터 얻은 closed ball space $\tilde{\mathbb{X}}_\epsilon$의 Betti numbers $\tilde{BN}_d(r)$은 $\tilde{BN}_d(r) = BN_d(ar)$의 관계를 가진다.
    
    - ***Proof)***
        
        $$\textbf{z}_{k} = [x_k,x_{k+\tau},\ldots,x_{k+(p-1)\tau}] \in \R^p \quad\quad (2)$$
        
        위 방정식 (2)로부터 $x_t$의 phase space vector $\bf{z_k}$와 $\tilde{x}_t$ phase space vector  $\tilde{\bf{z_k}}$를 얻으면 $\tilde{\bf{z}_k} = a\bf{z}_k$ 임을 알 수 있다.
        
        그러면 $\tilde{x}_t$의 attractor는 $x_t$의 attractor의 scaling image가 되고, 이 경우 $\tilde{x}_t$의 closed ball space $\tilde{\mathbb{X}}_r$과 $x_t$의 $\mathbb{X}_{ar}$도 scaling image가 된다.
        
        그러므로 $\tilde{\mathbb{X}}_r$과 $\mathbb{X}_{ar}$의 Betti number는 같다.
        
          
        

Proposition 6-1로부터, $x_t$의 Betti sequence $BS$와 $\tilde{x}_t$의 Betti sequence $\tilde{BS}$는 cell shift relation을 가진다.

즉, $\tilde{BS}_d(i) = BS_d(\lceil ai \rceil) + \epsilon$ 이다. (여기서 $\epsilon$은 이산화 오차)

이러한 시계열의 스케일에 대한 민감도는 매우 중요한 성질이다.

왜냐하면 분류 문제에 대해서 시계열의 스케일은 매우 중요한 정보이기 때문이다.

다른 한편으로 cell shift relation은 일반적인 벡터 데이터의 분류 문제와의 차이점이다.

만약 $a\approx 1$이라면, 두 시계열이 같은 그룹으로 분류될 가능성은 매우 높다.

그러나 $BS$와 $\tilde{BS}$의 Euclid distance는 상대적으로 크다.

그러므로, SVM이나 ANN같은 간단한 vector들에 대한 머신러닝 알고리즘들은 충분한 성능을 낼 수 없다.

이미지 분류 분야에서도 피사체의 위치 이동과 유사한 문제가 있다.

따라서 우리는 `Betti sequence`에 이미지 분류에 매우 효과적인 방법인 CNN을 적용하는 것을 고려한다.

## 6.1 One-dimensional Convolutional Neural Network

k-NN 및 DTW의 간단한 조합은 시간 매개변수 방향의 격차에 저항할 수 있는 이 알고리즘의 능력 때문에 대부분의 영역에서 우수한 분류 성능을 제공할 수 있다[Rakthanmanon 12].

그러나 이 방법을 Betti sequence에 응용하려고 한다면, $x_t$와 $ax_t$ ($a >>1$) 사이의 DTW distance 또한 작다.

이렇게 scaling 계수 $a$가 매우 큰 경우에, 두 시계열이 다른 그룹에 분류될 확률이 매우 높다.

그러므로 DTW는 Betti sequence 분류 문제에 적합하지 않다.

CNN은 하나 이상의 컨볼루션 레이어(종종 서브샘플링 단계 포함)와 표준적인 다층 신경 네트워크에서처럼 하나 이상의 완전히 연결된 레이어로 구성된다.

CNN의 아키텍처는 위치 정보를 보존하고 객체 크기의 차이를 막는 것도 중요한 이미지 분류에 대해 최고의 성능을 발휘할 수 있다.

우리는 이러한 특성이 베티 시퀀스의 반경 매개 변수 간격과 유사하므로 CNN이 베티 시퀀스의 분류에 적합하다고 생각한다.

1차원 CNN(1-CNN) 아키텍처는 다수의 컨볼루션 및 서브샘플링 레이어와 선택적으로 완전히 연결된 레이어로 구성된다.

컨볼루션 레이어의 입력은 $M$개의 값들이며, 여기서 $M$은 베티 수열의 길이이다.

컨볼루션 레이어는 크기  $n$ 의 $k$개 필터(또는 커널)를 가질 것이며, 여기서 $n$은 베티 시퀀스의 길이보다 작다.

이러한 필터는 Betti sequence의 로컬로 연결된 구조를 강조하여 크기가 $M-n+1$인 $k$개의 out-feature map을 생성한다.

그런 다음 각 맵은 일반적으로 $q$개의 인접 셀에 대한 평균 또는 최대 풀링을 사용하여 서브샘플링되며, 여기서 $q$는 하이퍼 파라미터로 하위 샘플링 단위 크기가 주어진다.

서브샘플링 레이어 전후에 각 feature map에 추가적인 bias와 시그모이드 비선형성이 적용된다. 

컨볼루션 레이어 뒤에는 완전 연결 레이어(fully connected layer)가 얼마든지 있을 수 있다.

Densely connected layer는 표준 다층 신경망의 계층과 동일하다.

그림 7은 CNN의 컨볼루션 및 하위 샘플링 하위 계층으로 구성된 전체 계층을 보여준다.

![Untitled](Time%20Series%20Classification%20via%20Topological%20Data%20An%20be717522bb804f75a2a6e33909ac615b/Untitled%206.png)

## 6.2 Parallel One-dimensional CNN

시계열 분류를 수행할 때, 우리는 또한 손과 다리에 부착된 센서와 같은 다채널 데이터를 사용할 수 있다.

각 채널의 시계열 데이터를 베티 시퀀스로 변환하여 다채널 시계열 데이터에서 다채널 베티 시퀀스를 구성할 수 있다.

다채널 베티 시퀀스 분류를 위해서는 1-CNN 아키텍처를 수정해야 한다.

단순한 확장에는 다중 채널 베티 시퀀스를 단위 시계열에 연결하는 것이 포함된다.

이 경우 입력 레이어는 $M_1+M_2+\cdots +M_s$ 단위를 갖는다.

여기서 $s$는 채널 수이고 $M_i$는 $i$번째 채널의 베티 수열 길이이다.

정 외 연구진 [Zheng 14]는 다채널 심층 CNN을 제안했다.

우리의 유사한 개념을 가진 아키텍처는 다변량 베티 시퀀스를 일변량 시퀀스로 분리하고 각 일변량 시리즈에 대해 개별적으로 feature learning을 수행한다.

이 아키텍처는 다채널 베티 시퀀스의 각 특징을 추출한 다음 해당 특징을 결합하여 각 레이블에 대한 점수를 계산한다.

그림 8은 병렬 1차원 신경망 아키텍처를 보여준다.

![Untitled](Time%20Series%20Classification%20via%20Topological%20Data%20An%20be717522bb804f75a2a6e33909ac615b/Untitled%207.png)

위의 두 아키텍처 사이의 차이점은 다채널 베티 시퀀스에 대한 필터가 같거나 혹은 다르다는 것이다.

분명히, 병렬 아키텍처는 더 상세한 모델을 나타낼 수 있다.

## 6.3 Learning Algorithm

1-CNN 및 parallel 1-CNN에 대한 학습 알고리즘은 기본적으로 이미지에 대한 CNN의 학습 알고리즘과 유사하다[Simard 03].

본 논문에서, 우리는 미니 배치가 있는 back propagation 알고리즘을 채택했다.

parallel 1- CNN의 경우, 우리는 첫 번째 완전 연결 계층의 오차를 각각의 채널 오차로 나눈 다음 각 채널에 대해 독립적으로 back propagation을 수행했다.