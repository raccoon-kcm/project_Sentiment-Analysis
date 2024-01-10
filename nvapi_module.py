# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:36:14 2021

@author: A
"""
import urllib.request
import datetime
import json

#본인이 발급받은 Client-Id,Client-Secret
client_id = '0LHQM4VX_MQM6JfkXofa'
client_secret = 'OcPgqpswCg'


#[CODE 1]
#요청에 Client-Id,Client-Secret 인증정보를
#추가하고 응답 response를 받아서 리턴
def getRequestUrl(url):    
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", client_id)
    req.add_header("X-Naver-Client-Secret", client_secret)
    
    try: 
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            print ("[%s] Url Request Success" % datetime.datetime.now())
            #time.sleep(1)
            return response.read().decode('utf-8')
    except Exception as e:
        print(e)
        print("[%s] Error for URL : %s" % (datetime.datetime.now(), url))
        return None
         

#[CODE 2]
#네이버 오픈(웹) api(요청 URL주소)를 구성하여
#getRequestUrl(url) 호출  
def getNaverSearch(node, srcText, start, display):    
    base = "https://openapi.naver.com/v1/search"
    node = "/%s.json" % node #크롤링 할 대상 (news)
    #parameters =  검색어,검색 시작 위치,검색 결과 출력 건수,정렬 
    parameters = "?query=%s&start=%s&display=%s&sort=sim" % (urllib.parse.quote(srcText), start, display)
    
    url = base + node + parameters    
    responseDecode = getRequestUrl(url)   #[CODE 1]
    
    if (responseDecode == None):
        return None
    else:
        return json.loads(responseDecode)

#[CODE 3]
#하나의 뉴스건을 제목,요약내용,링크등으로 분석
#날짜는 하나의 형식화된 문장열로 변환시키고 
#분석된 정보들을 사전으로 취합후 
#jsonResult 리스트에 추가  
def getPostData(post, jsonResult, cnt):    
    title = post['title']
    description = post['description']
    org_link = post['originallink']
    link = post['link']
    
    pDate = datetime.datetime.strptime(post['pubDate'],  '%a, %d %b %Y %H:%M:%S +0900')
    pDate = pDate.strftime('%Y-%m-%d %H:%M:%S')
    
    jsonResult.append({'cnt':cnt, 'title':title, 'description': description, 
    'org_link':org_link,   'link': link,   'pDate':pDate})
    return    

#[CODE 0]
#검색어 입력화면 제공
#100 * 10 = 1000 검색된 응답을 문자열로 인코딩하여
#json 파일에 저장
def webcrawler():
    node = 'news'   # 크롤링 할 대상
    srcText = input('검색어를 입력하세요: ')
    cnt = 0
    jsonResult = []

    jsonResponse = getNaverSearch(node, srcText, 1, 100)  #[CODE 2]
    total = jsonResponse['total']
 
    while ((jsonResponse != None) and (jsonResponse['display'] != 0)):         
        for post in jsonResponse['items']:
            cnt += 1
            getPostData(post, jsonResult, cnt)  #[CODE 3]       
        
        start = jsonResponse['start'] + jsonResponse['display']
        jsonResponse = getNaverSearch(node, srcText, start, 100)  #[CODE 2]
       
    print('전체 검색 : %d 건' %total)
    #파일명 : 검색어s_naver_node('news').json
    with open('%s_naver_%s.json' % (srcText, node), 'w', encoding='utf8') as outfile:
        #jsonResult 문자열로 인코딩
        jsonFile = json.dumps(jsonResult,  indent=4, sort_keys=True,  ensure_ascii=False)
        #jsonResult 문자열을 파일에 저장                 
        outfile.write(jsonFile)
    
    file_name = str(srcText)+'_naver_'+str(node)+'.json'
    print("가져온 데이터 : %d 건" %(cnt))
    print('%s_naver_%s.json SAVED' % (srcText, node))
    
    return file_name

