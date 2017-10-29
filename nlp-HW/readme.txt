=============================
#--CrossEntropy Assignment--#
=============================
1. 실행방법
 1.1 hangulJamoDecoder.py 실행
     : sejong.nov.train.txt, sejong.nov.test.txt, hani.test.txt의 자모 분리 파일 생성
 
 1.2 hangulSyllablesDecoder.py 실행
     : sejong.nov.train.txt, sejong.nov.test.txt, hani.test.txt의 음절 분리 파일 생성

 1.3 run.py 실행
     : unigram과 bigram 시 entropy와 cross_entropy 계산
     *주의 :bigram 계산 시 시간이 오래 걸림



2. 문제점
 2-1. bigram 문제 
      : 현재 코드에서는 자모, 음절 순서대로 bigram을 만듦(ex.('ㄱ', 'ㄲ'), ('ㄲ', 'ㄴ'), ('ㄴ', 'ㄷ'))
	문제는 주어진 데이터에는 순서대로 생성되지 않은 bigram이 많다는 것임(ex.ㅅ,ㅣ,ㄴ,ㅇ,ㅐ,none,ㄴ,ㅜ)
	이로 인해 bigram 데이터를 사용하여 entropy와 cross_entropy를 계산할 때에 구하는 확률 p(bigram) 값의 합이 1이 안 나옴
	
	문제를 코드로 확인해보고 싶다면, 
	run.py에서 vec_entropy, vec_cross_entropy에서 주석처리된 print문들을 주석해제하고 실행하면 됨


 2-2. 속도 문제
      : 속도 향상을 위해, for문을 vector 계산으로 모두 바꾸려는 시도를 하였음
        그러나 cross_entropy에서 확률을 계산할 때, type_list에서 bigram의 위치를 기억할 필요성를 발견하고,
	for문 하나를 추가하였는데, 계산 시간이 상당히 증가함

	vector 계산의 속도 향상을 확인하고 싶다면,
	## non_vectorization Vs. vectorization ## 가 포함된 주석을 해제하고 실행하면 됨
	
	개선이 요구
	

 2-3. 지나치게 많은 print문
      : run.py 뒷 부분을 보면, print문과 = 문이 너무 많아 보기 어려움
 
	깔끔하게 개선이 요구



3. 결과물
   : outcome.txt 파일 참고

