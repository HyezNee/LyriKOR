from transformers import PreTrainedTokenizerFast
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline
import re
from tqdm import tqdm
import nltk
from nltk.corpus import cmudict
import csv
import argparse
import pandas as pd


def write_list_to_file(lst, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        # 한번에 쓰기
        f.seek(0)
        f.writelines(lst)

# 1. 한 -> 영 -> 한
def translate_lyrics(df, kor_eng, eng_kor):

    punct = "/-?!.,#$%\'\"()*+-/:;<=>@[\\]^_`{|}~" + '“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    original = []
    res = []

    for i, song in enumerate(tqdm(df['lyrics'])):
        for line in str(song).split("[SEP]"):
            if line == "":
                continue
            translated = ""
            to_translate = ""
            for word in line.split():
                if bool(re.match(f"^[A-Za-z{punct}]+$", word)): # English -> 번역X
                    # 일단 쌓인 한국어 (to_translate)를 번역
                    # print(to_translate)
                    if to_translate != "":
                        ke = kor_eng(to_translate.rstrip())[0]['translation_text']
                        ek = eng_kor(ke)[0]['translation_text']
                        # to_translate 초기화, 영어는 그대로 붙이기
                        to_translate = ""
                        translated += ek + " " + word + " "
                    else:
                        translated += word + " "
                elif bool(re.match(f"^[가-힣{punct}]+$", word)):  # Korean -> 번역O
                    to_translate += word + " "
                elif bool(re.match(f"^[A-Za-z가-힣0-9{punct}]+$", word)): # 영어+한글 (cool한) 의 경우 영어만 사용 (한국어는 거의 조사일 꺼임)
                    translated += re.sub(r'[^a-zA-Z0-9]', '', word) + " "
                else:    # 나머지 언어는 pass
                    print("아예 무시된 단어: ", word)
                    break   # 무시 단어 + 한/영 조합의 line일 경우 한/영 부분만 번역되는 불상사를 방지하기 위해 break
            # to_translate에 남아 있는 한국어 마저 번역
            if to_translate != "":
                ke = kor_eng(to_translate.rstrip())[0]['translation_text']
                ek = eng_kor(ke)[0]['translation_text']
                translated += ek
            # 최종 결과에 추가
            if translated != "":
                res.append(translated.rstrip()+'\n')  # 개행 추가해줘야 파일 쓸 때 반영됨!!!
            # 원본 가사도 추가 (영어, 한국어만)
            if bool(re.match(f"^[A-Za-z가-힣{punct}\s]+$", line)):
                original.append(line+'\n')

        # For back-up
        # if (i+1) % 30 == 0:
        #   filename = f"/content/drive/MyDrive/result/X_{(i+1) // 30}.txt"
        #   write_list_to_file(res, filename)
        #   res = []

        #   filename2 = f"/content/drive/MyDrive/result/GT_{(i+1) // 30}.txt"
        #   write_list_to_file(original, filename2)
        #   original = []

    return res, original


# 2. 음절 세서 붙이기
def is_korean_word(word):
    korean_pattern = re.compile(r"[가-힣]+")  # Unicode range for Korean characters
    return bool(re.match(korean_pattern, word))

def is_number(word):
    number_pattern = re.compile(r"^[0-9]+$")
    return bool(re.match(number_pattern, word))

def count_syllables(d, word):
    if is_korean_word(word) or is_number(word):
        return len(word)
    if word.lower() in d:
        return len([syl for syl in d[word.lower()][0] if any(char.isdigit() for char in syl)])
    else:
        return len(re.findall(r'[aeiouy]+', word.lower()))

def sentence_syllables(d, sent):
    s = 0
    for word in sent.split():
        word = word.lower().strip(".:;?!-,'()\"")
        s += count_syllables(d, word)
    return s

def process_files(original, res, save_path):    # file1_path, file2_path, save_path):
    
    custom_vocab_map = {'[START]': '<s>', '[LEN1]': '<unused0>', '[LEN2]': '<unused1>',
        '[LEN3]': '<unused2>', '[LEN4]': '<unused3>', '[LEN5]': '<unused4>',
        '[LEN6]': '<unused5>', '[LEN7]': '<unused6>', '[LEN8]': '<unused7>',
        '[LEN9]': '<unused8>', '[LEN10]': '<unused9>', '[LEN11]': '<unused10>',
        '[LEN12]': '<unused11>', '[LEN13]': '<unused12>', '[LEN14]': '<unused13>',
        '[LEN15]': '<unused14>', '[LEN16]': '<unused15>', '[LEN17]': '<unused16>',
        '[LEN18]': '<unused17>', '[LEN19]': '<unused18>', '[LEN20]': '<unused19>',
        '[LEN21]': '<unused20>', '[LEN22]': '<unused21>', '[LEN23]': '<unused22>',
        '[LEN24]': '<unused23>', '[LEN25]': '<unused24>', '[LEN26]': '<unused25>',
        '[LEN27]': '<unused26>', '[LEN28]': '<unused27>', '[LEN29]': '<unused28>',
        '[LEN30]': '<unused29>', '[LEN31]': '<unused30>', '[LEN32]': '<unused31>',
        '[LEN33]': '<unused32>', '[LEN34]': '<unused33>', '[LEN35]': '<unused34>',
        '[LEN36]': '<unused35>', '[LEN37]': '<unused36>', '[LEN38]': '<unused37>',
        '[LEN39]': '<unused38>', '[LEN40]': '<unused39>', '[LEN41]': '<unused40>',
        '[LEN42]': '<unused41>', '[LEN43]': '<unused42>', '[LEN44]': '<unused43>',
        '[LEN45]': '<unused44>', '[LEN46]': '<unused45>', '[LEN47]': '<unused46>',
        '[LEN48]': '<unused47>', '[LEN49]': '<unused48>', '[LEN50]': '<unused49>',
        '[LEN51]': '<unused50>', '[LEN52]': '<unused51>', '[LEN53]': '<unused52>',
        '[LEN54]': '<unused53>', '[LEN55]': '<unused54>', '[LEN56]': '<unused55>',
        '[LEN57]': '<unused56>', '[LEN58]': '<unused57>', '[LEN59]': '<unused58>',
        '[LEN60]': '<unused59>', '[LEN0]': '<unused60>', '[END]': '</s>'}

    nltk.download('cmudict')
    d = cmudict.dict()  # word - pronunciation

    # For back-up
    # with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r+', encoding='utf-8') as f2:
    #     # old_lines1 = f1.readlines()
    #     lines1 = f1.readlines()
    #     lines2 = f2.readlines()
    #     lines1 = list(map(lambda s: s.strip(), lines1)) # GT
    #     lines2 = list(map(lambda s: s.strip(), lines2)) # X
        
    lines1 = original
    lines2 = res

    # back-up mode의 경우 들여쓰기 존재
    for i, line1 in tqdm(enumerate(lines1)):
        syllable_count = sentence_syllables(d, line1)
        if syllable_count < 61:
            len_token = custom_vocab_map['[LEN'+str(syllable_count)+']']
        else:
            len_token = custom_vocab_map['[LEN60]']

        if i < len(lines2):
            line2 = lines2[i].strip()
            lines2[i] = f"<s>{len_token}{line2}</s>\n"
        else:
            lines2.append(f"<s>{len_token}</s>\n")

    # open the file in the write mode
    with open(save_path, 'w', encoding='utf-8') as f:
      # create the csv writer
      writer = csv.writer(f)
        
      header = ['GT', 'X_w_tokens']
        
      # write the header
      writer.writerow(header)

      new_lines1 = list(map(lambda s: '<s>'+s+'</s>', lines1))  
      # write a row to the csv file
      writer.writerows([[new_lines1[i], lines2[i]] for i in range(len(lines2))])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Make dataset for syllabic adjustment model')
    parser.add_argument('--lyrics_dataset_path', type=str)
    parser.add_argument('--save_dataset_path', type=str)
    args = parser.parse_args()

    # KOR -> ENG (1)
    checkpoint = "circulus/kobart-trans-ko-en-v2"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    # ENG -> KOR (2)
    checkpoint2 = "circulus/kobart-trans-en-ko-v2"
    tokenizer2 = PreTrainedTokenizerFast.from_pretrained(checkpoint2)
    model2 = AutoModelForSeq2SeqLM.from_pretrained(checkpoint2)

    # Load data => column name: 'lyrics'
    df = pd.read_csv(args.lyrics_dataset_path, encoding='cp949')

    # step 1 : 영->한->영
    kor_eng = pipeline("translation", tokenizer=tokenizer, model=model, max_length=50, device=0)
    eng_kor = pipeline("translation", tokenizer=tokenizer2, model=model2, max_length=50, device=0)

    res, original = translate_lyrics(df, kor_eng, eng_kor)

    # file1_path = f"GT.txt"
    # file2_path = f"X_wo_tokens.txt"

    # step 2 : 음절 구해서 token 붙이기
    process_files(original, res, args.save_dataset_path)