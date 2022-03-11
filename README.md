[CVPR 2016 Open Access Repository](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

# 코드 구현

[https://github.com/Soohyeon-Bae/ResNet](https://github.com/Soohyeon-Bae/ResNet)

# 논문 리뷰

# Introduction

- 신경망을 깊게 하면 무조건 성능이 좋아질까? (VGG)

→ 아니다.

Vanishing/exploding gradients 문제

층이 깊어지면 정확도가 포화되어 성능 저하가 발생, 오버피팅과는 다른 기울기 소실 문제

> with the network depth increasing, accuracy gets saturated and then degrades rapidly.
> 
- 그렇다면 어떤 방법으로 신경망을 쌓아야 깊이에 대한 효과를 볼 수 있는가?

→ 잔차를 학습하자!

# Deep Residual Learning

### Residual Learning

- 기존의 신경망은 입력 $x$를 타겟 $y$로 매핑하는 함수 $H(x)$를 얻는 것이 목적이었다면 **ResNet은 $F(x) + x$의 최소화를 목적으로 함**
- $x$는 변하지 않는 값이므로 $F(x)$를 0에 가까운 값으로 만드는 것이다. $F(x) = H(x) - x$이므로 $F(x)$를 최소로 하면 $H(x) - x$의 최솟값과 동일한 의미를 가짐
- 여기서 $H(x) - x$를 **잔차(residual)**라고 하며, $H(x) = x$이 최적의 목표값이 된다.
- 네트워크 학습 시 $F(x)+x$의 미분에서,  $x$의 미분값은 1이므로 layer가 아무리 깊어져도 최소 gradient로 1이상의 값을 갖게 하여  vanishing gradient 문제 해결
- **Weight layer를 통해 나온 결과와 그 전의 결과를 더한 후 ReLU 연산 수행**
- x가 그대로 skip connection되기 때문에 연산의 증가 없이 성능 향상 가능

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/09b24b39-7971-41af-a001-3345ba8d09ee/Untitled.png)

### Identity Mapping by Shortcuts

- I**dentity block**은 네트워크의 output F(x)에 x를 그대로 더함
    
    ![Identity block](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/272681bb-88fe-4180-8387-f0b128b9f2e0/Untitled.png)
    
    Identity block
    
    - 입력과 출력의 차원이 동일해야 함
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/372e1402-f3d0-49a4-8522-6700eb1af5c2/Untitled.png)
        
    - 입력과 출력의 차원을 맞추는 과정이 필요한 경우에는 linear projection $Ws$를 사용함
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/24b4d497-fadd-4bd8-be9e-7c4fa1d3a601/Untitled.png)
        
- 본 논문에서는 layer 2~3개마다 적용했지만, 유연하게 조정할 수 있음. 그러나 layer 하나에 대한 identity mapping은 linear layer와 비슷하기 때문에 효과가 없음
- **Convolution block**은 1x1 convolution연산을 거친 후 F(x)에 더함
    
    ![Convolution block](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5678f609-d9a0-42ea-8926-5a44844d3525/Untitled.png)
    
    Convolution block
    

### Network Architectures

- 기존 VGG-19가 더 깊어진 34-layer plain network
    - 3x3 conv layer, stride=2
    - **연산량 보존**을 위해 출력 맵의 사이즈가 절반으로 줄어들면, filter의 개수(depth)를 두 배 증가
    - conv layer 2개마다 skip connection 연결
- Skip connection이 추가된 **34-layer residual network**
    - 실선 : 입력과 출력의 차원이 동일한 경우, identity shortcut 수행
    - 점선 : 입력과 출력의 차원이 다른 경우
        - zero padding으로 차원을 맞춰준 후 identity shortcut 수행
        - projection shortcut 시행

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3064f807-7ee1-4c4b-8ec3-7efa9d1e7e3f/Untitled.png)

### Implementation

- Input size : 224x224
- Batch Normalization
- Initialize weights
- SGD, mini-batch : 256
- Learning rate : 0.1
- Iteration : 60x10^4
- Weight decay : 0.0001
- Momentum : 0.9
- Dropout : no use

# Experiments

- **Plain Network**
    - 순방향 전파 신호가 non-zero variances를 갖는 Batch Normalization을 사용하여 훈련됨
        
        → 층이 깊어질때 **훈련 성능이 떨어지는 원인은 기울기 소실 문제가 아님**
        
- **Residual Network**
    - Identitiy shortcuts 적용
    - 차원 증가가 필요한 경우, **zero-padding** 사용
        
        → plain network와 비교하여 **추가 파라미터 없음**
        

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/87fe7a82-c343-4cb3-86dd-9c9785e12f37/Untitled.png)

1. 18-layer ResNet 보다 34-layer ResNet가 더 낮은 training error를 가지며, 성능이 더 좋음
    
    **잔차를 학습하면 층이 깊어져도 degradation problem 을 잘 조절함**을 알 수 있음
    
2. Plain network와 비교했을 때, 성공적으로 training error 를 줄였다.
3. 18-layer끼리 비교했을 때 accruracy는 비슷하지만, ResNet이 더 빠르게 수렴함(converges faster)
→ ResNet에서 SGD를 사용한 optimization이 더 쉬움

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e27d8654-e658-4f67-b992-c43d0a15bf85/Untitled.png)

(A) using identity mapping for all shortcuts and zero-padding for increasing dimensions

(B) using projection shortcuts for increasing dimensions

(C) only using projection shortcuts 

- (A)보다 (B)의 성능이 높음 → (A)의 zero-padding 부분에서는 잔차 학습이 이뤄지지 않기 때문
- (B)보다 (C)의 성능이 높음 → projection shortcuts 과정에서 추가 파라미터가 생성, 학습됨
- 그러나 projection shortcuts를 사용하지 않더라도, plain network보다 확실히 높은 성능을 보이기 때문에, 굳이 파라미터를 늘려 메모리와 시간을 낭비할 필요가 없음

- **Bottleneck Architecture**
    - 처음 1x1 conv를 통해 차원을 줄이고, 3x3 conv으로 연산을 수행한 후, 뒷부분에 있는  1x1 conv를 통해 다시 차원 확대
    - 이는 3x3 conv를 2개 연결시킨 구조에 비해 **연산량 절감** 효과가 있음
    - 1x1 Convolution은 연산량이 작기 때문에 Feature Map(Output Channel)을 줄이거나 키울 때 자주 사용됨 (Inception)
    - **Convolution Parameters = Kernel Size x Kernel Size x Input Channel x Output Channel**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5359cb0c-17ea-4108-b533-8ca5ab48ad05/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9ada9b58-782e-4bd2-926d-b2bf14841d55/Untitled.png)

- 참고
    
    [[CNN 알고리즘들] ResNet의 구조](https://bskyvision.com/644)
    
    [8. CNN 구조 3 - VGGNet, ResNet](https://m.blog.naver.com/laonple/221259295035)
    
    [(논문리뷰) ResNet 설명 및 정리](https://ganghee-lee.tistory.com/41)
    
    [(ResNet)Deep Residual Learning for Image Recognition 논문 리뷰](https://tobigs.gitbook.io/tobigs/deep-learning/computer-vision/resnet-deep-residual-learning-for-image-recognition)
    
    [[Part Ⅴ. Best CNN Architecture] 8. ResNet [2] - 라온피플 아카데미 -](https://m.blog.naver.com/laonple/220764986252)
