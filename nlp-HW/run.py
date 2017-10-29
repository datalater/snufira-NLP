import re
import numpy as np
from nltk.util import ngrams
from nltk.probability import FreqDist
def loadTxtFile(path):
    contain = []
    with open(path, 'r', encoding='utf-8') as f :
        data = f.readlines()
    for line in data:
        contain += line.split(",")
    return contain

def prob(element, input_list):
    ans = input_list.count(element) / len(input_list)
    return ans

def entropy(input_list, type_list):
    tmp = 0
    for val in type_list:
        tmp += - prob(val, input_list) * math.log2(prob(val, input_list))
    return tmp

def cross_entropy(model_list, base_list, type_list):
    tmp = 0
    for val in type_list:
        tmp += - prob(val, base_list) * math.log2(prob(val, model_list))
    return tmp


def vec_entropy(input_list, type_list):
    len_input_list = len(input_list)  # base_list 길이 
    fdist = FreqDist(input_list)      # FreqDist로 base_list의 # token을 구함

    init_zero = [0.0000000001] * len(type_list)
    type_dict = dict(zip(type_list, init_zero)) # type_list dictionary 초기화

                                                ## bigram 계산 시   
    for key_val, counts_val in fdist.items() :  # base_list의 bigram 종류가 type_list의 bigram보다 많기 때문에
        if key_val in type_list:                # base_list에 속하는 bigram만 저장
            type_dict[key_val] = counts_val

    input_dict_values = np.array(list(type_dict.values())).reshape(1,-1) # np.array로 변환
    prob_li = np.divide(input_dict_values, len_input_list).reshape(1,-1) # token의 확률을 구함

    # print("type_list_len",len(type_list))
    # print("type_dict.keys()_len",len(type_dict.keys()))
    # print("len_input_list",len_input_list)                ## bigram 계산 시  
    # print("sum of prob_li", np.sum(prob_li))              # model_list에 속하는 bigram의 확률만 구했기 때문에
    # print("prob_li", prob_li)                             # 모든 확률을 더한 값이 1이 안 나옴
    outcome = - np.multiply(prob_li, np.log2(prob_li))  
    return np.sum(outcome)


def vec_cross_entropy(model_list, base_list, type_list):
    len_base_list = len(base_list) # base_list 길이 
    b_fdist = FreqDist(base_list)  # FreqDist로 base_list의 # token을 구함

    init_zero = [0.0000000001] * len(type_list) 
    b_type_dict = dict(zip(type_list, init_zero))   # type_list dictionary 초기화
    
                                                    ## bigram 계산 시    
    for key_val, counts_val in b_fdist.items() :    # base_list의 bigram 종류가 type_list의 bigram보다 많기 때문에
        if key_val in type_list:                    # base_list에 속하는 bigram만 저장
            b_type_dict[key_val] = counts_val


    base_dict_values = np.array(list(b_type_dict.values())).reshape(1,-1)   # np.array로 변환
    base_prob_li = np.divide(base_dict_values, len_base_list).reshape(1,-1) # token의 확률을 구함

    len_model_list = len(model_list) # model_list 길이
    m_fdist = FreqDist(model_list)   # FreqDist로 model_list의 # token을 구함

    m_type_dict = dict(zip(type_list, init_zero))   # type_list dictionary 초기화

                                                    ## bigram 계산 시  
    for key_val, counts_val in m_fdist.items() :    # model_list의 bigram 종류가 type_list의 bigram보다 많기 때문에
        if key_val in type_list:                    # model_list에 속하는 bigram만 저장
            m_type_dict[key_val] = counts_val       

    model_dict_values = np.array(list(m_type_dict.values())).reshape(1,-1)
    model_prob_li = np.divide(model_dict_values, len_model_list).reshape(1,-1) 
    # print("type_list_len",len(type_list))
    # print("b_type_dict.keys()_len",len(b_type_dict.keys()))
    # print("type_list_len",len(type_list))
    # print("m_type_dict.keys()_len",len(m_type_dict.keys()))
    # print("len_model_list",len_model_list)                    ## bigram 계산 시  
    # print("sum of model_prob_li", np.sum(model_prob_li))      # model_list에 속하는 bigram의 확률만 구했기 때문에
    # print("model_prob_li", model_prob_li)                     # 모든 확률을 더한 값이 1이 안 나옴
    outcome = - np.multiply(base_prob_li, np.log2(model_prob_li))
    return np.sum(outcome)


if __name__ == '__main__':
    raw_train_jamo = loadTxtFile("./raw_train_jamo.txt")[:-1]
    raw_test_jamo = loadTxtFile("./raw_test_jamo.txt")[:-1]
    raw_hani_jamo = loadTxtFile("./raw_hani_jamo.txt")[:-1]

    raw_train_sil = loadTxtFile("./raw_train_sil.txt")[:-1]
    raw_test_sil = loadTxtFile("./raw_test_sil.txt")[:-1]
    raw_hani_sil = loadTxtFile("./raw_hani_sil.txt")[:-1]

    # print(raw_train_jamo[-1]); print(raw_train_sil[-1])
    # print(raw_test_jamo[-1]); print(raw_test_sil[-1])
    # print(raw_hani_jamo[-1]); print(raw_hani_sil[-1])

    from hangulJamoDecoder import *
    cor_FINAL_JAMO = [val for val in list(FINAL_JAMO.values()) if not val in list(INITIAL_JAMO.values())]
    jamo = list(INITIAL_JAMO.values()) + list(MID_JAMO.values()) + cor_FINAL_JAMO
    syllable = [chr(val) for val in range(0xAC00, 0xD7A4)]

    """
    print(## non_vectorization Vs. vectorization ##)
    import time
    start_time = time.clock()
    j_vec_uni_test_entropy = vec_entropy(raw_test_jamo,jamo)
    print("j_vec_uni_test_entropy: {0:.4f}".format(j_vec_uni_test_entropy))
    print("{0:.4f} seconds".format(time.clock() - start_time))

    start_time = time.clock()
    j_non_vec_uni_test_entropy = entropy(raw_test_jamo, jamo)
    print("j_non_vec_uni_test_entropy: {0:.4f}".format(j_non_vec_uni_test_entropy))
    print("{0:.4f} seconds".format(time.clock() - start_time))


    start_time = time.clock()
    j_vec_uni_test_cross_entropy =  vec_cross_entropy(raw_test_jamo, raw_train_jamo, jamo)
    print("j_vec_uni_test_cross_entropy: {0:.4f}".format(j_vec_uni_test_cross_entropy))
    print("{0:.4f} seconds".format(time.clock() - start_time))

    start_time = time.clock()
    j_non_vec_uni_test_cross_ent = cross_entropy(raw_test_jamo, raw_train_jamo, jamo)
    print("j_non_vec_uni_test_cross_entropy: {0:.4f}".format(j_non_vec_uni_test_cross_ent))
    print("{0:.4f} seconds".format(time.clock() - start_time))

    """

    import math
    print("######*------unigram------*######")
    j_uni_train_entropy = vec_entropy(raw_train_jamo, jamo)
    j_uni_test_entropy = vec_entropy(raw_test_jamo, jamo)
    j_uni_hani_entropy = vec_entropy(raw_hani_jamo, jamo)
    j_uni_train_cross_ent = vec_cross_entropy(raw_train_jamo, raw_train_jamo, jamo)
    j_uni_test_cross_ent = vec_cross_entropy(raw_test_jamo, raw_train_jamo, jamo)
    j_uni_hani_cross_ent = vec_cross_entropy(raw_hani_jamo, raw_train_jamo, jamo)

    print("j_uni_train_entropy: {0:.4f}".format(j_uni_train_entropy))
    print("j_uni_test_entropy: {0:.4f}".format(j_uni_test_entropy))
    print("j_uni_hani_entropy: {0:.4f}".format(j_uni_hani_entropy))

    print("j_uni_train_cross_entropy: {0:.4f}".format(j_uni_train_cross_ent))
    print("j_uni_test_cross_entropy: {0:.4f}".format(j_uni_test_cross_ent))
    print("j_uni_hani_cross_entropy: {0:.4f}".format(j_uni_hani_cross_ent))
    print("")

    s_uni_train_entropy = vec_entropy(raw_train_sil, syllable)
    s_uni_test_entropy = vec_entropy(raw_test_sil, syllable)
    s_uni_hani_entropy = vec_entropy(raw_hani_sil, syllable)
    s_uni_train_cross_ent = vec_cross_entropy(raw_train_sil, raw_train_sil, syllable)
    s_uni_test_cross_ent = vec_cross_entropy(raw_test_sil, raw_train_sil, syllable)
    s_uni_hani_cross_ent = vec_cross_entropy(raw_hani_sil, raw_train_sil, syllable)

    print("s_uni_train_entropy: {0:.4f}".format(s_uni_train_entropy))
    print("s_uni_test_entropy: {0:.4f}".format(s_uni_test_entropy))
    print("s_uni_hani_entropy: {0:.4f}".format(s_uni_hani_entropy))

    print("s_uni_train_cross_entropy: {0:.4f}".format(s_uni_train_cross_ent))
    print("s_uni_test_cross_entropy: {0:.4f}".format(s_uni_test_cross_ent))
    print("s_uni_hani_cross_entropy: {0:.4f}".format(s_uni_hani_cross_ent))
    print("")

    

    bi_jamo, bi_syllable = list(ngrams(jamo,2)), list(ngrams(syllable,2))
    bi_raw_train_jamo, bi_raw_test_jamo, bi_raw_hani_jamo = list(ngrams(raw_train_jamo,2)), list(ngrams(raw_test_jamo,2)), list(ngrams(raw_hani_jamo,2))
    bi_raw_train_sil, bi_raw_test_sil, bi_raw_hani_sil = list(ngrams(raw_train_sil,2)), list(ngrams(raw_test_sil,2)), list(ngrams(raw_hani_sil,2))

    print("######*------bigram------*######")
    j_bi_train_entropy = vec_entropy(bi_raw_train_jamo, bi_jamo)
    j_bi_test_entropy = vec_entropy(bi_raw_test_jamo, bi_jamo)
    j_bi_hani_entropy = vec_entropy(bi_raw_hani_jamo, bi_jamo)
    j_bi_train_cross_ent = vec_cross_entropy(bi_raw_train_jamo, bi_raw_train_jamo, bi_jamo)
    j_bi_test_cross_ent = vec_cross_entropy(bi_raw_test_jamo, bi_raw_train_jamo, bi_jamo)
    j_bi_hani_cross_ent = vec_cross_entropy(bi_raw_hani_jamo, bi_raw_train_jamo, bi_jamo)

    print("j_bi_train_entropy: {0:.4f}".format(j_bi_train_entropy))
    print("j_bi_test_entropy: {0:.4f}".format(j_bi_test_entropy))
    print("j_bi_hani_entropy: {0:.4f}".format(j_bi_hani_entropy))

    print("j_bi_train_cross_entropy: {0:.4f}".format(j_bi_train_cross_ent))
    print("j_bi_test_cross_entropy: {0:.4f}".format(j_bi_test_cross_ent))
    print("j_bi_hani_cross_entropy: {0:.4f}".format(j_bi_hani_cross_ent))
    print("")
 
    s_bi_train_entropy = vec_entropy(bi_raw_train_sil, bi_syllable)
    s_bi_test_entropy = vec_entropy(bi_raw_test_sil, bi_syllable)
    s_bi_hani_entropy = vec_entropy(bi_raw_hani_sil, bi_syllable)
    s_bi_train_cross_ent = vec_cross_entropy(bi_raw_train_sil, bi_raw_train_sil, bi_syllable)
    s_bi_test_cross_ent = vec_cross_entropy(bi_raw_test_sil, bi_raw_train_sil, bi_syllable)
    s_bi_hani_cross_ent = vec_cross_entropy(bi_raw_hani_sil, bi_raw_train_sil, bi_syllable)

    print("s_bi_train_entropy: {0:.4f}".format(s_bi_train_entropy))
    print("s_bi_test_entropy: {0:.4f}".format(s_bi_test_entropy))
    print("s_bi_hani_entropy: {0:.4f}".format(s_bi_hani_entropy))

    print("s_bi_train_cross_entropy: {0:.4f}".format(s_bi_train_cross_ent))
    print("s_bi_test_cross_entropy: {0:.4f}".format(s_bi_test_cross_ent))
    print("s_bi_hani_cross_entropy: {0:.4f}".format(s_bi_hani_cross_ent))
    print("")
