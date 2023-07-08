import torch
from transformers import PreTrainedTokenizerFast
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline
import pandas as pd
import os
import re
import nltk
from nltk.corpus import cmudict
import argparse

from Syllabic_adjustment.Syllabic_adjustment_model import CustomKoBART
from kobart import get_kobart_tokenizer


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

def process_files(inputs, eng_kor):
    
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
        
    # Count syllable of inputs
    syllable_count = sentence_syllables(d, inputs)
    if syllable_count < 61:
        len_token = custom_vocab_map['[LEN'+str(syllable_count)+']']
    else:
        len_token = custom_vocab_map['[LEN60]']

    translated = eng_kor(inputs, max_length=50)[0]['translation_text']

    return f"<s>{len_token}{translated}</s>"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='inference')
    # parser.add_argument('--lyrics_csv_file', type=str)
    parser.add_argument('--input', type=str)
    # parser.add_argument('--save_dataset_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ENG -> KOR (2)
    checkpoint2 = "circulus/kobart-trans-en-ko-v2"
    tokenizer2 = PreTrainedTokenizerFast.from_pretrained(checkpoint2)
    model2 = AutoModelForSeq2SeqLM.from_pretrained(checkpoint2)
    eng_kor = pipeline("translation", tokenizer=tokenizer2, model=model2, device=0)

    # TODO: csv file 단위로 한번에 inference 처리 => .ipynb 파일을 py 파일로 옮기기
    # filename = 'eng_lyrics_syllables.csv'   # column name: 'lyrics'
    # df = pd.read_csv(os.path.join('dataset', args.lyrics_csv_file))

    kobart_tokenizer = get_kobart_tokenizer()
    model_input = process_files(args.input, eng_kor)
    model = CustomKoBART()
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device(device)))
    model.to(device)

    model_input = kobart_tokenizer([model_input], padding="max_length", max_length=30, truncation=True, return_tensors='pt')
    res_ids = model.generate(model_input["input_ids"].to(device), num_beams=4, min_length=0, max_length=50)
    result = kobart_tokenizer.batch_decode(res_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print(result)