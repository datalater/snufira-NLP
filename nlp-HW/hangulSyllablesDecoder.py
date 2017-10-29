import re
def saveRawSil(path, name):
    with open(path, "r", encoding="utf-8") as f:
        data = f.readlines()

    with open(name, "w", encoding="utf-8") as f:
        for hangulText in data:
            for ch in hangulText:
                if (re.search('^[가-힣]+$', ch)):
                    f.write(ch + ',')



if __name__ == '__main__':
    import os.path

    if os.path.isfile("./raw_train_sil.txt"):
        print("There are already existed 'raw_train_sil.txt'")
    else:
        saveRawSil("./sejong.nov.train.txt", "./raw_train_sil.txt")


    if os.path.isfile("./raw_test_sil.txt"):
        print("There are already existed 'raw_test_sil.txt'")
    else:
        saveRawSil("./sejong.nov.test.txt", "./raw_test_sil.txt")


    if os.path.isfile("./raw_hani_sil.txt"):
        print("There are already existed 'raw_hani_sil.txt'")
    else:
        saveRawSil("./hani.test.txt", "./raw_hani_sil.txt")
