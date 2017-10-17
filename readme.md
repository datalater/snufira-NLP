
ⓒ JMC 2017

http://ling.snu.ac.kr/class/AI_Agent/

---

## Summary

+ Regular Expression
+ Finite State Automata
+ Edit Distance : 두 string의 유사성 판단
+ Minimum Edit Distance : 두 string의


---
---
---

## 2주차 :: 03 Word Prediction

단어 예측하는 알고리즘으로 N-gram이 있다.
N-gram은 앞에 단어를 토대로 뒤에 나올 단어를 예측하는 알고리즘이다.
N은 token의 개수, 즉 개별적인 단어의 개수를 뜻한다.
1개의 단어만 보고 뒤에 단어를 예측하면 1-gram(uni-gram)이라고 한다.
연속된 2개의 단어만 보고 뒤에 단어를 예측하면, 2-gram(bi-gram)이라고 한다.

N-gram 알고리즘은 그 다음 단어가 나올 확률을 알아내기 위해서 빈도를 계산한다.
다음과 같이 conditional probability를 사용한다.
$P(w_i | w_{i-1})$

N-gram은 기본적으로 확률을 구하는 모델링이기 때문에 next word를 예측할 때 유용한 알고리즘이다.
Language Modeling에서도 앞의 singnal을 보고 뒤의 signal을 예측하는 것인데 원리가 같다.

언어의 확률값을 가장 정확하게 구하는 방법은 counting이다.
그래서 counting이 중요하다.
그런데 word tokenization이나 sentence tokenization 모두 counting이 쉽지 않다.

2-gram으로 예시를 들어보자.
"He stepped out into the hall, was dleighted to encounter a water brother."
He가 주어이고 brother가 문장 마지막인지 알기 위해 눈에 보이지 않지만 start, end 표시가 있다고 생각한다.
따라서, (start, He), (He, stepped) .... (water, brother), (brother, end)로 14쌍이 나온다.

counting할 때는 token과 type을 counting한다.
실제로 나타난 개별적인 단어는 token이라고 하고, unique한 token의 종류를 type이라고 한다.

wordform에 따라 count의 기준이 달라질 수 있다.
cat과 cats는 따로 세야 할까?
Lemma는 cat과 cats이 cat이라는 lemma로 묶이므로 따로 세지 않는다.
Wordform은 cat과 cats를 별도의 단어로 센다.
대부분의 경우는 wordform 단위로 작업을 한다.
왜냐하면 Lemmatization하는 것이 어렵기 때문이다.
가령 goose와 geese가 같은 lemma인지 알기 어렵다.

> **Note**: Lemma : stem, major part of speech, rough word sense가 같은 단어

우리가 하려는 것은 단어의 확률을 구하는 것이다.
단어의 확률을 구하는 이유는 다음 단어를 예측하기 위해서이다.
확률을 구할 때는 실제로 특정 단어 뒤에 어떤 단어가 몇 번 나오는지 counting을 해서 계산한다.
확률은 어느 자료에서 구할까?
Corpora(Corpus의 복수형)에서 구한다.

Language Modeling이란 무엇일까.
'새빨간 ____'를 맞추는 것처럼, 앞 단어 sequence를 가지고 다음 단어를 predict하는 확률적인 모델을 language modeling이라 한다.
`<--휴식-->`

$P(w_i | w_{i-1})$를 사용해서 실제로 확률을 구해보면 잘 안된다.
긴 연쇄를 corpus에서 찾는 것은 매우 드물고 있더라도 biase되어 있을 확률이 높다.
biase되어 있다는 것은 한 문서에 쓰인 문장이 다른 곳에서 등장하는 경우 여러 곳에서 자주 쓰인다고 볼 수 없는 경우를 뜻한다.
실제 corpus에서 우리가 긴 연쇄를 찾으면 거의 없다보니 0이 될 확률이 높다.
다른 방법을 사용해야 한다.
'신효필은 천재다'를 찾는 것보다 '신효필', '천재'를 break해서 찾는 것이 낫다.
그러니 chain rule을 쓰자.
$P(A | B) = P(A, B)/ P(B)$

chain rule을 사용하면 다음과 같이 정리된다.
"P(its water was so transparent) = P(its) * P(water | its) * P(was | its water) * P(so | its water was) * P(transparent | its water was so)"
그런데 여기에도 문제점이 있다.
쪼깨진 P중에 하나가 0이 되면 전체 확률이 0이 되어버리거나, 결국 가장 뒤에 있는 conditional probabilty에서는 긴 연쇄가 등장할 수밖에 없기 때문에 긴 연쇄를 찾는 어려움이 또 등장하게 된다.

그래서 independent assumption을 사용한다.
이는 앞에 나온 단어는 뒤에 나온 단어와 independent하다는 가정을 말한다.
이를 Markov assumption이라고 한다.
Markov 가정에서는 보통 bi-gram으로 사용한다.

$$P(w_i | w_{i-1}) = \frac{count(w_{i-1}, w_i)}{count(w_{i-1})}$$
예를 들면, "P(거짓말 | 새빨간) = P(새빨간, 거짓말) / P(새빨간 uni-gram)"

실제 counting에 바탕을 두었기 때문에 참고한 corpus에서 최대가 되는 값을 찾았으므로 Maximum Likelihood Estimates가 된다.  

bi-gram estimation 끝.

bi-gram을 보면 조악해보이지만 직관적으로 일치하는 경우가 많다.
`슬라이드 p.30`을 보면 N-gram이 world knowledge나 language knowledge를 반영하고 있다고 볼 수 있다.

`슬라이드 p.33` N이 높아질수록 불확실성 entropy가 줄어든다.

`슬라이드 p.34` bi-gram 확률 중 중간에 0이 나타나면 smoothing을 해야 한다. `<--휴식-->`

우리가 어떤 corpus에서 학습을 해서 language modelingm을 만들고 나면 모델을 evaluation을 해야 한다.
evalulation은 information theory의 entropy 또는 cross-entroy를 통해 수행한다.
corpus 데이터에서 training data와 test data를 8:2로 나눴을 때,
training data에 없는 단어가 test data에 나오면 `<UNK>`(unknown)로 표시한다.

### Zero-Count Smoothing

`@p.34` 대부분의 bi-gram이 존재하지 않는 경우가 있다.
어떤 경우에 대한 count가 0일 때 어떻게 대응해야 할까?
Zipf의 법칙에 따르면, 소수의 단어가 빈도수가 매우 높다.
대다수의 단어는 빈도수가 낮다.
또 대다수의 단어는 쓰이기까지 시간이 꽤 걸릴 수도 있다.
그래서 0이 나타나는 현상을 보조, 즉 smoothing 해야 한다.

Laplace Smoothing.
모든 type에 count를 +1을 더한다.
$P_{Laplace}(w_i)=\frac{c_{i}+1}{N+V}$.
Laplace Smoothing의 의미는 counting 개수가 많은 것에서 빼앗아서 count 개수가 없는 것에 나눠준 것과 같다.
실제로 계산해보면 `@p.45` 참조.
type수가 많을수록 V값이 커지고 discount ratio가 커져서 original count에 통계적 왜곡이 크게 발생한다.

`@p.51` Good-Turing.
count가 없다가 새롭게 등장한 type에 +1을 더한다.
한 번도 안 나타난 type은 1로 초기화한다.
문제는 실제로 1번 잡힌 물고기는 원래대로라면 1/18인데
확률의 총합은 1이고 count 없는 것에 확률을 할당했으니, 1/18보다 작아져야 한다.
...
Laplace Smoothing에 비하면 Good-Turing은 adjusted ratio의 비율이 훨씬 더 합리적으로 조절된다.

Backoff.
bi-gram 없으면 uni-gram으로 backoff 해라.
우리가 원하는 것은 '이 수업이 재미있다'인데, '이 수업'이라는 bigram이 없으면 '수업'이라는 uni-gram을 사용하면 의미가 완전히 달라질 수 있다.
왜냐하면 '저 수업' 또는 '그 수업'이 될 수도 있기 때문이다.

`@p.66` Practical Issues.
대부분의 bi-gram은 소수점이 6개 정도 등장할 정도로 확률 값이 매우 낮다.
그 수를 계속 곱셈하면 0의 개수가 너무 늘어나서 overflow로 계산을 못하게 된다.
그래서 실제로 계산할 때는 log를 취해서 덧셈으로 바꾼다.
마지막 값에 exponentiate하면 원래 값을 구할 수 있다.

대부분의 과제 작업에서는 add +0.5를 하면 된다.

**수업끝.**

---

## 1주차 :: 02 Finite State Automata

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

## 1주차 :: 01 Regular Expression

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
