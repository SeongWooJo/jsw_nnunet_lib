import unicodedata
import re

def parse_endoscope_pattern(filename):
    # EXPORT_ 뒤의 숫자와 마지막 _ 뒤의 숫자만 추출
    pattern = r"EXPORT_(\d+)_(.*)_(\d+)_(\d+)\.jpg"
    # 정규식으로 매칭
    match = re.match(pattern, filename)    
    if match:
        # 첫 번째 숫자 (예: 9866)
        num1 = match.group(1)
        name = match.group(2)
        time = match.group(3)
        num2 = match.group(4)
        #new_filename = f"Endo_{num1}-{str(num2).zfill(4)}"
    else:
        print(f"{filename} 파일명이 예상한 패턴과 일치하지 않습니다.")
        return 0, 0, 0, 0, False
    return num1, name, time, num2, True

# 문자열 정규화 함수
def normalize_text(text):
    return unicodedata.normalize('NFC', text)

# 중복 검사 함수수
def check_duplicate(df, keys):
    duplicate_rows = df[df.duplicated(keys, keep=False)]
    return duplicate_rows

def get_case(filename):
    num1, _, _, _, _ = parse_endoscope_pattern(filename)
    return num1

def get_name(filename):
    _, name, _, _, _ = parse_endoscope_pattern(filename)
    name = normalize_text(name)
    return name

def get_time(filename):
    _, _, time, _, _ = parse_endoscope_pattern(filename)
    return time

def get_detail(filename):
    _, _, _, detail, _ = parse_endoscope_pattern(filename)
    return detail
