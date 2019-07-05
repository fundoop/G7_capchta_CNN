# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:01:19 2019

@author: fz
"""


import urllib.request as ur
import http.cookiejar as hc
import time
from urllib.parse import urlencode
from random import random

headdic={
'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
'Accept-Encoding': 'gzip, deflate, br',
'Accept-Language': 'zh-CN,zh;q=0.9',
'Connection': 'keep-alive',
'Host': 'hosturl',
'Upgrade-Insecure-Requests': '1',
'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}
url='url'
data=[]
cookie=hc.MozillaCookieJar()
handler=ur.HTTPCookieProcessor(cookie)
opener=ur.build_opener(handler)
req=ur.Request(url,headers=headdic)
res=opener.open(req)
i=5000

while True:
    res=opener.open('url/captcha.jpg?'+str(random()))
    with open(r'./tmp_captcha/'+str(i)+'.jpg','wb') as f:
        f.write(res.read())
    i+=1
    print(i)
    time.sleep(1)
