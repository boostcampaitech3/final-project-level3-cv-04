general_cleansing_list = [14, 28, 32, 48, 52, 64, 70, 74, 75, 77, 84, 93, 202, 209]
general_cleansing_list2 = [0, 149, 157, 193, 194, 198, 201, 207]


id_list = ['ID', '아이디', 'NETWORK', '네트워크', 'IP', 'WIFI', "WIFIID"]
seperater = [':','.','_']
for s in seperater:
    id_list += list(map(lambda x:x + s, id_list))
pw_list = ['PW', '비밀번호','PASSCODE', 'PASSWORD', '패스워드', 'PIN', 'P.W', '비번', "WIFIPW"]
for s in seperater:
    pw_list += list(map(lambda x:x + s, pw_list))
wifi_list = ['WIFI', 'WI-FI', '와이파이', ':', '/']
key_list = id_list + pw_list + wifi_list


key_list2 = ["와이파이명", "1층", "2층", "3층", "4층", "(WI-FI)", "패스워드", "WIFI", "ID", "PW", "1·2층", 
"FREE", "WI-FI", "네트워크:", "패스워드:", "WIFI", "PW:", "WI-FI:", "PW:", "ZONE", "무선",
"인터넷", "WITI", "AP:", "PASSWORD", "와이파이", "비밀번호", "ID:", "WIF", "PASSWOR", "PA",
"SSWORD", "아이디", "비번", "DW", "**", "WF", "WITI", "WI", "★PASSWORD", "★PASSWORD★",
"WI-FI", "WIFI", "I-FI", "PASSWO", "WIFI(와이", "WIFI(와이파이)", "PASSWORE", "네트워크", "WI", "PW)",
"WI-FIZONE", "SSID", "W:", "D:", "PASS:", "1F", "2F", "3F", "4F", "SSID",
"검색명칭", "패스워트", "P/W", "WI-FI(ID)", "I'D:", "PN:", "P.W", "WI-", "WI-FIZONE", "무선랜",
"5G전용", "POSSWORD", "POSS", "WORD", "무선랜명:", "FI", "KEY", "계정", "WIFT", "WTFI",
"WTFT", "P.W.:", "ID:", "PASS:", "FW", "IP", "와이", "파이", "이파이", "와이파",
"ID_", "PW_", "PIN", "P/W", "WI-F", "I_", "FI_", "IFI_", "NAME", "PASS", "WORD"]