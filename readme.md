
ⓒ JMC 2017

http://ling.snu.ac.kr/class/AI_Agent/

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

> **Note**: `/[A-z]/` : A..Z특수문자a..z 매칭한다. 아스키코드를 보면 A-Z와 a-z 사이에 특수문자가 포함되기 때문이다.

### Regular Expression :: Example (2) 반복

| RE | 매칭 예시 | 기능 |
| --- | --- | --- |
| `/[b]a+!/` | baa! or baaa! baaaa! | repetition |

**1주차 끝.**

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
