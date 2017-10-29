#coding=utf-8
'''
Created on 2013-04-28

@author: Andrew


This is a library for processing characters form Unicode's Hangul
Syllables block. For more information, please see these links:

http://www.unicode.org/charts/PDF/UAC00.pdf
http://en.wikipedia.org/wiki/Korean_language_and_computers
'''

import math

'''--------------- Constants ---------------'''

'''Map offsets to the Jamo they represent'''
#http://en.wikipedia.org/wiki/Korean_language_and_computers#Initial_Jamo
INITIAL_JAMO = {0 : u"ㄱ",
                1 : u"ㄲ",
                2 : u"ㄴ",
                3 : u"ㄷ",
                4 : u"ㄸ",
                5 : u"ㄹ",
                6 : u"ㅁ",
                7 : u"ㅂ",
                8 : u"ㅃ",
                9 : u"ㅅ",
                10: u"ㅆ",
                11: u"ㅇ",
                12: u"ㅈ",
                13: u"ㅉ",
                14: u"ㅊ",
                15: u"ㅋ",
                16: u"ㅌ",
                17: u"ㅍ",
                18: u"ㅎ"}

#http://en.wikipedia.org/wiki/Korean_language_and_computers#Medial_Jamo
MID_JAMO     = {0 : u"ㅏ",
                1 : u"ㅐ",
                2 : u"ㅑ",
                3 : u"ㅒ",
                4 : u"ㅓ",
                5 : u"ㅔ",
                6 : u"ㅕ",
                7 : u"ㅖ",
                8 : u"ㅗ",
                9 : u"ㅘ",
                10: u"ㅙ",
                11: u"ㅚ",
                12: u"ㅛ",
                13: u"ㅜ",
                14: u"ㅝ",
                15: u"ㅞ",
                16: u"ㅟ",
                17: u"ㅠ",
                18: u"ㅡ",
                19: u"ㅢ",
                20: u"ㅣ"}

#http://en.wikipedia.org/wiki/Korean_language_and_computers#Final_Jamo
FINAL_JAMO   = {0 : u"none", #note that this is the string "none", different from the value None
                1 : u"ㄱ",
                2 : u"ㄲ",
                3 : u"ㄳ",
                4 : u"ㄴ",
                5 : u"ㄵ",
                6 : u"ㄶ",
                7 : u"ㄷ",
                8 : u"ㄹ",
                9 : u"ㄺ",
                10: u"ㄻ",
                11: u"ㄼ",
                12: u"ㄽ",
                13: u"ㄾ",
                14: u"ㄿ",
                15: u"ㅀ",
                16: u"ㅁ",
                17: u"ㅂ",
                18: u"ㅄ",
                19: u"ㅅ",
                20: u"ㅆ",
                21: u"ㅇ",
                22: u"ㅈ",
                23: u"ㅊ",
                24: u"ㅋ",
                25: u"ㅌ",
                26: u"ㅍ",
                27: u"ㅎ"}

'''
Constants for defining the Hangul Syllables Block
   http://www.unicode.org/charts/PDF/UAC00.pdf
'''
HAN_SYL_START = 0xAC00 #the address of the first character
                       #in the Hangul Syllables Block.
HAN_SYL_END   = 0xD7AF #the address of the final character
                       #in the Hangul Syllables Block

'''
Constants for Hangul Syllable Decoding formula
   http://en.wikipedia.org/wiki/Korean_language_and_computers#Hangul_Syllables_Area
'''
INITIAL_COEFF = 0x24C #588 in hex
MID_COEFF     = 0x1C  #24 in hex






'''--------------- Hangul Syllables Functions ---------------'''

def isHangulSyllable(ch):
    """Takes a character and returns true if it is a member of
    the Hangul Syllables Block in the Unicode standard:
        http://www.unicode.org/charts/PDF/UAC00.pdf

    @var ch : the character to be checked
    @type ch : str
    @return : True if the character is from the Hangul Syllables Block, False if it is from a different block, None otherwise
    @rtype : bool
    """
    #Java's Regex had an easy way to do this. No Python equivalent.
    #Had to invent this

    if len(ch) != 1: #if it isn't 1 character long
        return None #it's not true nor false

    #convert to an ordinal number
    chOrd = ord(ch)

    if (chOrd < HAN_SYL_START) or (chOrd > HAN_SYL_END):
        #if the character is outside the range for the
        #Hangul Syllables Block as defined here:
        #    http://www.unicode.org/charts/PDF/UAC00.pdf
        return False
    else:
        #It must be inside the block
        return True

def getInitialJamo(syl):
    """Takes a character form the Unicode Hangul Syllables Block and
    calculates its initial jamo using the following formula:
        sylOrd = initialOffset*INITIAL_COEFF + midOffset*MID_COEFF + finalOffset + HAN_SYL_START
        http://en.wikipedia.org/wiki/Korean_language_and_computers#Hangul_Syllables_Area

    @var syl : the hangul syllable to be decoded
    @type syl : str
    @return : initial jamo
    @rtype : str
    """
    if isHangulSyllable(syl):
        #convert to a number
        sylOrd = float(ord(syl))
        initialOffset = math.floor((sylOrd - HAN_SYL_START) / INITIAL_COEFF)

        return INITIAL_JAMO[initialOffset]
    else:
        return None

def getMidJamo(syl):
    """Takes a character form the Unicode Hangul Syllables Block and
    calculates its middle jamo using the following formula:
        sylOrd = initialOffset*INITIAL_COEFF + midOffset*MID_COEFF + finalOffset + HAN_SYL_START
        http://en.wikipedia.org/wiki/Korean_language_and_computers#Hangul_Syllables_Area

    @var syl : the hangul syllable to be decoded
    @type syl : str
    @return : middle jamo
    @rtype : str
    """
    if isHangulSyllable(syl):
        #convert to a number
        sylOrd = float(ord(syl))
        #% is the modulo operator. It gets the remainder.
        midOffset = math.floor(((sylOrd - HAN_SYL_START) % INITIAL_COEFF) / MID_COEFF)

        return MID_JAMO[midOffset]
    else:
        return None

def getFinalJamo(syl):
    """Takes a character form the Unicode Hangul Syllables Block and
    calculates its final jamo using the following formula:
        sylOrd = initialOffset*INITIAL_COEFF + midOffset*MID_COEFF + finalOffset + HAN_SYL_START
        http://en.wikipedia.org/wiki/Korean_language_and_computers#Hangul_Syllables_Area

    @var syl : the hangul syllable to be decoded
    @type syl : str
    @return : final jamo
    @rtype : str
    """
    if isHangulSyllable(syl):
        #convert to a number
        sylOrd = float(ord(syl))
        #% is the modulo operator. It gets the remainder.
        finalOffset = ((sylOrd - HAN_SYL_START) % INITIAL_COEFF) % MID_COEFF

        return FINAL_JAMO[finalOffset]
    else:
        return None #note that None is different from the FINAL_JAMO[0] value of "none"

def decodeSyllable(syl):
    """Takes a character form the Unicode Hangul Syllables Block and
    calculates its constituent jamo using the following formula:
        sylOrd = initialOffset*INITIAL_COEFF + midOffset*MID_COEFF + finalOffset + HAN_SYL_START
        http://en.wikipedia.org/wiki/Korean_language_and_computers#Hangul_Syllables_Area

    @var syl : the hangul syllable to be decoded
    @type syl : str
    @return : (initial, mid, final) if syl is a member of Hangul Syllabes else None
    @rtype : (str, str, str)
    """
    initialJamo = getInitialJamo(syl)
    midJamo = getMidJamo(syl)
    finalJamo = getFinalJamo(syl)

    return (initialJamo, midJamo, finalJamo)

def saveRawJamo(path, name):
    with open(path, "r", encoding = "utf-8") as f:
        data = f.readlines()

    with open(name, "w", encoding = "utf-8") as f:
        for hangulText in data:
            for ch in hangulText:
                initialJamo, midJamo, finalJamo = decodeSyllable(ch)
                if initialJamo:
                    f.write(initialJamo + ',' + midJamo + ',' + finalJamo + ',')

                    
                

if __name__ == '__main__':
    import os.path

    if os.path.isfile("./raw_train_jamo.txt"):
        print("There are already existed 'raw_train_jamo.txt'")
    else:
        saveRawJamo("./sejong.nov.train.txt", "./raw_train_jamo.txt")


    if os.path.isfile("./raw_test_jamo.txt"):
        print("There are already existed 'raw_test_jamo.txt'")
    else:
        saveRawJamo("./sejong.nov.test.txt", "./raw_test_jamo.txt")


    if os.path.isfile("./raw_hani_jamo.txt"):
        print("There are already existed 'raw_hani_jamo.txt'")
    else:
        saveRawJamo("./hani.test.txt", "./raw_hani_jamo.txt")

