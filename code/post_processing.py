import re

def delete_korean(s):
    korean = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    no_korean = re.sub(korean, '', s)
    # ?, ", $, [, \, ], and +.

    return no_korean


def split_str(s):
    if ':' in s:
        lists = s.split(':')
        return ''.join(lists[1::])
    else:
        return s


def receipt_cleansing(s):
    # 00000000-00-XXXXXX 42번 113번: ID, PW로 인식되는 것 제거
    p = re.compile('\d{8}-\d{2}-\w{6}')
    m = p.match(s)
    if m:
       return ''
    else:
        return s
    

def post_process(id_lists, pw_lists):
    # out {'id':[[text1, points], [text2, points]],
    #      'pw':[[text1, points], [text2, points]]}
    new_out = {'id':[], 'pw':[]}

    for out_id in id_lists:
        out_id = split_str(out_id)
        if len(out_id) != 1:        # 아이디의 맨 앞부분 :가 인식되는 경우
            new_out['id'].append(out_id)

    for out_pw in pw_lists:
        out_pw = receipt_cleansing(out_pw)
        out_pw = delete_korean(out_pw)
        out_pw = split_str(out_pw)

        if len(out_pw) >= 8: # 8자리 미만 비밀번호 제외
            p = re.compile('\w\s') # 'L ', 'ㅣ ' 등의 형태
            m = p.match(out_pw[0:2])
            if m:
                out_pw = out_pw[2::]

            new_out['pw'].append(out_pw)

    return new_out