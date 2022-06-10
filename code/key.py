# id_list = ['ID', '아이디', 'NETWORK', '네트워크', 'IP', 'WIFI', "WIFIID"]
# seperater = [':','.','_']
# for s in seperater:
#     id_list += list(map(lambda x:x + s, id_list))
# pw_list = ['PW', '비밀번호','PASSCODE', 'PASSWORD', '패스워드', 'PIN', 'P.W', '비번', "WIFIPW"]
# for s in seperater:
#     pw_list += list(map(lambda x:x + s, pw_list))
# wifi_list = ['WIFI', 'WI-FI', '와이파이', ':', '/']
# key_list = id_list + pw_list + wifi_list

# ----------------------------------------------------------------

key_list3_general = ["와이파이명", "1층", "2층", "3층", "4층", "(WI-FI)", "패스워드", "WIFI", "ID", "PW", "1·2층", 
"FREE", "WI-FI", "네트워크:", "패스워드:", "WIFI", "PW:", "WI-FI:", "PW:", "ZONE", "무선",
"인터넷", "WITI", "AP:", "PASSWORD", "와이파이", "비밀번호", "ID:", "WIF", "PASSWOR", "PA",
"SSWORD", "아이디", "비번", "DW", "WF", "WITI", "WI", "★PASSWORD", "★PASSWORD★",
"WI-FI", "WIFI", "I-FI", "PASSWO", "WIFI(와이", "WIFI(와이파이)", "PASSWORE", "네트워크", "WI", "PW)",
"WI-FIZONE", "SSID", "W:", "D:", "PASS:", "1F", "2F", "3F", "4F", "SSID",
"검색명칭", "패스워트", "P/W", "WI-FI(ID)", "I'D:", "PN:", "P.W", "WI-", "WI-FIZONE", "무선랜",
"5G전용", "POSSWORD", "POSS", "WORD", "무선랜명:", "FI", "KEY", "계정", "WIFT", "WTFI",
"WTFT", "P.W.:", "ID:", "PASS:", "FW", "IP", "와이", "파이", "이파이", "와이파",
"ID_", "PW_", "PIN", "P/W", "WI-F", "I_", "FI_", "IFI_", "NAME", "PASS", "WORD", "WIFIID", "NETWORK"]

key_list3_receipt = ["[WIF", "PASSWORD]", "SWORD]", "PAS", "WIFI아이디는", "비밀번호는", "무선인터넷은",
"무선인터넷", "WIFI아이디", "WIFI비밀번호", "WI", "-FI", "WI-", "PW-", "WIP", "WIPI",
"W:", "IF", "와이파이P/W", "P/Ø", "|비밀번호", "FI비밀번호", "IFI비밀번호", "W1TI", "W1T1", "WIT1",
"W1", "T1", "Wㅣ", "Fㅣ", "PASSWARD", "ACCESS", "CODE", "I-FI:", "-FI:", "FI:", "PASS", "PASS:",
"1D", "이름", "WIFIPASSWORD", "[와이파", "OI] ", "[와이파이]", "I아이디", "PIII", "1/D", "무선", "무선랜", "P.W",
"P.\\", "IFI", "\\IFI", "FI비밀번호", "WI-F1", "W1F1", "ID-", "PW-", "B1층", "1층,2층", "P.", "<WI-FI>", "ID:KT", "RD", "무료",
"W:", "III", "WIII", "DW", "PWI", "이비번", "와이파이비번", "파이비번",
"SSWOR", "WIFI비밀번호", "*WIFI", "F/W:", "ACCESS", "CODE", "Passw", "rd:", "ID/PASSWORD", "P/W", "1D",
"1층WIFI", "2층WIFI", "3층WIFI", "WI+I", "네트워크명", "-WI-FI-", "WI-I", "I비번", "와이", "파이", "와이파이비번",
"PASSWO", "RD", "WIT1", "D)", "ID)", "PW)", "W)", "[WI-FI", "PW]", "IDI", "ID]", "WIFI-PASSWORD", "F/W:", "WI-F",
"P.", "P.W", "IFI", "비밀번", "P/W", "W1-F1", "W1-F1", "무선와이파이", "/PW", "이디", "II)", "와이", "파이",
"비일번호", "번호", "WIF", "PA", "-비번", "SWORD:", "PASSWO", "RD", "WIT", "와이파이비번:", "PB:", "PIPI", "PASSWO",
"PASSWOR", "WIFI명", "WIFI명:", "·FI", "WI·FI", "비밀번", "비밀번오", "WIFE", "I-", "PIII", "Wiri"]

key_list = key_list3_general + key_list3_receipt
seperater = [':','.','_',"-","<",">","ㅣ","L","I","/","(",")", "'", ""]

new_list = []
for key in key_list:
    for s in seperater:
        keytmp = key
        key = key + s
        new_list.append(key)
        key= keytmp
        key = s + key
        new_list.append(key)
        key = keytmp

new_list = list(set(new_list)) +  [':', '/']
    
general_cleansing_list_v2 = [0, 11, 12, 14, 23, 26, 27, 28, 30, 32, 33, 45, 47, 48, 52, 64, 69, 70, 72, 74, 75, 77, 84, 87, 91, 93, 95, 111,
146, 149, 156, 157, 194, 198, 201, 202, 207, 209, 230 ]

receipt_cleansing_list_v2 = [16, 24, 28, 45, 47, 53, 55, 56, 60, 61, 63, 66, 70, 71, 78, 87, 96, 101, 104, 107, 109, 111, 119, 122, 130, 135,
140, 143, 144, 147, 148, 149, 150, 152, 157, 158, 168, 171, 174, 178, 179, 182, 184, 192, 208, 211, 214, 216,
222, 224, 227, 240, 243, 249, 251]