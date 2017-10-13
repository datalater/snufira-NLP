
ⓒ JMC 2017

http://ling.snu.ac.kr/class/AI_Agent/

---

## 02 Finite State Automata

### 정의

+ 유한한 수의 상태와 transition을 가진 오토마타(기계, 회로, 알고리즘)

### 유한하다는 것

+ 정규언어(regular expression)는 반복되는 형태가 있기 때문에 적은(유한한) 개수의 state로도 표현할 수 있다.

### p.27 Formally (slide title)

+ language의 structure를 구현하는 flow
+ delta function (가장 중요)
  + q0에서 b를 input으로 받으면 q1으로 가라
  + q3에서 a를 보면 반복을 해라
  + q3에서 !를 보면 q4로 넘어가라
  + q5가 되면 멈춘다.
+ FSA를 만드는 이유는 이대로 컴퓨터의 회로로 구현할 수 있기 때문이다.


### p.29 Recognition

+ FSA를 recognize algorithm으로 사용할 수 있다.

### deterministic vs. non-deterministic

+ qn에서 특정 input이 들어왔을 때 어느 상태로 가면 되는지 결정되어 있으므로 deterministic하다고 말한다.
+ 그러나 실제 자연어에서는 non-deterministic한 경우가 많다.

### non-deterministic

+ 같은 symbol 인풋에 대해서 다른 선택을 할 수도 있다


### Problems of non-deterministic

+ choice point에서 어떤 결정을 해야 할지 문제가 생긴다.
  + 선택의 분기점마다 표시를 해두고 특정 길로 갔다가 백업하여 다른 길로 하나 하나씩 다른 path를 모두 찾아간다.
  + 미리 앞의 state를 살펴보고 결정한다.

### Recognition as search

+ recognition 과정은 결국 가능한 combination을 모두 펼쳐 놓고 일일이 체크한다.
+ 이러한 recognition은 인공지능에서 search 문제로 볼 수 있다.

### p.39

+ NFA : 입실론 무브먼트 포함하는 오토마타

### Sum

+ 지금까지 한 게 formal representation이다.
+ 가장 간단한 방법이 Regular Expression이고 이것을 구현한 것이 FSA이다.
+ FSA는 그대로 컴퓨터 회로에 구현할 수 있다.

---

## 01 Regular Expression

### formal representation의 도구

+ 인간의 언어를 컴퓨터에게 이해시키는 formal representation의 한 가지 방법으로 Regular Expression을 사용할 수 있다.

> **Note**: 촘스키의 언어 표현 단위에 따르면 가장 작은 단위인 regular expression을 포함한 4가지가 있다.

### Regular Expression

+ 어떤 기호를 가지고 패턴을 만들면 원하는 search string을 찾을 수 있다.
+ 이러한 패턴을 regular expression이라 한다.

### Regular Expression :: Example (1) 한 글자

| RE | 매칭 예시 | 기능 |
| --- | --- | --- |
| `/[aA]/` | a 또는 A | chracter disjunction  |
| `/[A-Za-z]/` | ABCDEFG...Zabcdefg...z에 해당하는 모든 single character | range |
| `/[^A-Z]/` | A-Z가 아닌 모든 single character | negation |
| `/[e][^f]/` | e로 시작하고 뒤에는 f가 아닌 two strings | some included, others excluded |

> **Note**: range를 쓸 때는 from A와 to B를 명확히 이해할 필요가 있다. 가령, `/[A-z]/`의 경우 특수문자까지 매칭된다. 아스키코드를 보면 A-Z와 a-z 사이에 특수문자가 포함되기 때문이다.

### Regular Expression :: Example (2) 반복

| RE | 매칭 예시 | 기능 |
| --- | --- | --- |
| `/[b]a+!/` | baa! or baaa! baaaa! | repetition |

### Regular Expression에서 알아야 할 것들

+ Hierarchy 매칭 우선순위가 존재한다
+ 불필요한 내용까지 greedy하게 검색되지 않도록 세밀하게 패턴을 만들어야 한다.

**끝.**

---

## 00 Introduction (2)

### NLP가 크게 각광을 받게 된 이유

+ IBM Watson이 인간을 꺾고 Jeopardy 우승

### Sentiment Analysis

감정 분석 / 의견 분석
- 영화평, 상품평이 negative한지 positive한지 분류
- 온라인상에 엄청난 리뷰들이 올라오는데 이러한 의견을 자동으로 분석해보자.
- 기업의 경우 sentiment analysis가 매우 중요하다. 유저들의 반응을 실시간으로 살펴가며 흐름을 아는 게 고객관리 차원에서 매우 중요하기 때문에.
- 이슈들이 어떻게 변화되어 가는지 sentiment analysis로 알 수 있다.

### QA가 어려운 이유
- domain이 제한되어 있지 않으므로 질문을 이해하거나 답변을 찾아야 할 범위가 너무 넓음
- 질의 자체를 이해하는 게 쉽지 않음
- 왓슨의 경우 사회자의 말을 듣고 Answering을 한 게 아니라 질문지를 미리 전달 받았음

---

## 00 인간 언어의 특징

### Ambiguity

+ 인간 언어에는 우리가 생각하는 것 이상으로 ambiguity 문제가 존재한다.
+ 인간의 언어는 중의적인 의미를 가진다.
+ ambiguity 문제 때문에 semantic processing에서 의미를 어떻게 결정해야 할지 문제가 생긴다.
+ 이를 '**ambiguity resolution**'이라 한다.

### 한국어와 영어

+ 한국어는 refraction(굴절)이 굉장히 많다.
+ refraction이 많다는 것은 하나의 단어가 여러 형태를 가진다는 것.
+ language model을 사용할 때 한국어처럼 언어의 꼴이 달라지게 되면 빈도도 떨어지게 되고 모델링이 복잡해진다.
+ 따라서 언어의 특성에 맞게 알고리즘을 개발해야 한다.

### NLP 세 가지 단계

+ human language knowledge
+ formal representation
+ efficient algorithm

---

## 00 Introduction (1)

### What is NLP

+ 인간이 이해할 수 있는 언어를 컴퓨터에게 이해시키는 것.
+ 컴퓨터에게 이해시키는 이유? 컴퓨터에게 지시해서 우리가 필요한 것을 얻기 위해.
+ 그래서 인간의 언어가 어떻게 되어있는지 knowledge가 있어야 하고 그 knowledge를 컴퓨터가 이해할 수 있는 형태로 transform 해야 하고, 그러한 과정을 efficent한 algorithm으로 표현할 수 있어야 한다.

### 우리가 배울 것

+ 인간의 언어에 대한 knowledge는 있다고 가정하고 (morphology, refraction 등)
+ formal representation과
+ efficient한 algorithm으로 표현하는 방법을 배운다.

### NLP의 역사

+ Machine Translation, 기계번역의 역사가 곧 NLP의 역사
+ 제2차 세계대전에서 정보를 전달하기 위해 본격적으로 시작됨

---
---
