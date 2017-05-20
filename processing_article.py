# -*- coding: utf-8 -*-

import codecs
import glob
from konlpy.tag import Twitter
import math
from collections import namedtuple
from gensim import models
import time
import re
import networkx
import itertools

_twitter = Twitter()
_stopwords = ["아", "휴", "아이구", "아이쿠", "아이고", "어", "나", "우리", "저희", "따라", "의해", "을", "를", "에", "의", "가", "으로", "로", "에게", "뿐이다", "의거하여", "근거하여", "입각하여", "기준으로", "예하면", "예를", "들면", "들자면", "저", "소인", "소생", "저희", "지말고", "하지마", "하지마라", "다른", "물론", "또한", "그리고", "비길수", "없다", "해서는", "안된다", "뿐만", "아니라", "만이", "아니다", "만은", "아니다", "막론하고", "관계없이", "그치지", "않다", "그러나", "그런데", "하지만", "든간에", "논하지", "않다", "따지지", "않다", "설사", "비록", "더라도", "아니면", "만", "못하다", "하는", "편이", "낫다", "불문하고", "향하여", "향해서", "향하다", "쪽으로", "틈타", "이용하여", "타다", "오르다", "제외하고", "이", "외에", "이", "밖에", "하여야", "비로소", "한다면", "몰라도", "외에도", "이곳", "여기", "부터", "기점으로", "따라서", "할", "생각이다", "하려고하다", "이리하여", "그리하여", "그렇게", "함으로써", "하지만", "일때", "할때", "앞에서", "중에서", "보는데서", "으로써", "로써", "까지", "해야한다", "일것이다", "반드시", "할줄알다", "할수있다", "할수있어", "임에", "틀림없다", "한다면", "등", "등등", "제", "겨우", "단지", "다만", "할뿐", "딩동", "댕그", "대해서", "대하여", "대하면", "훨씬", "얼마나", "얼마만큼", "얼마큼", "남짓", "여", "얼마간", "약간", "다소", "좀", "조금", "다수", "몇", "얼마", "지만", "하물며", "또한", "그러나", "그렇지만", "하지만", "이외에도", "대해", "말하자면", "뿐이다", "다음에", "반대로", "반대로", "말하자면", "이와", "반대로", "바꾸어서", "말하면", "바꾸어서", "한다면", "만약", "그렇지않으면", "까악", "툭", "딱", "삐걱거리다", "보드득", "비걱거리다", "꽈당", "응당", "해야한다", "에", "가서", "각", "각각", "여러분", "각종", "각자", "제각기", "하도록하다", "와", "과", "그러므로", "그래서", "고로", "한", "까닭에", "하기", "때문에", "거니와", "이지만", "대하여", "관하여", "관한", "과연", "실로", "아니나다를가", "생각한대로", "진짜로", "한적이있다", "하곤하였다", "하", "하하", "허허", "아하", "거바", "와", "오", "왜", "어째서", "무엇때문에", "어찌", "하겠는가", "무슨", "어디", "어느곳", "더군다나", "하물며", "더욱이는", "어느때", "언제", "야", "이봐", "어이", "여보시오", "흐흐", "흥", "휴", "헉헉", "헐떡헐떡", "영차", "여차", "어기여차", "끙끙", "아야", "앗", "아야", "콸콸", "졸졸", "좍좍", "뚝뚝", "주룩주룩", "솨", "우르르", "그래도", "또", "그리고", "바꾸어말하면", "바꾸어말하자면", "혹은", "혹시", "답다", "및", "그에", "따르는", "때가", "되어", "즉", "지든지", "설령", "가령", "하더라도", "할지라도", "일지라도", "지든지", "몇", "거의", "하마터면", "인젠", "이젠", "된바에야", "된이상", "만큼 어찌됏든", "그위에", "게다가", "점에서", "보아", "비추어", "보아", "고려하면", "하게될것이다", "일것이다", "비교적", "좀", "보다더", "비하면", "시키다", "하게하다", "할만하다", "의해서", "연이서", "이어서", "잇따라", "뒤따라", "뒤이어", "결국", "의지하여", "기대여", "통하여", "자마자", "더욱더", "불구하고", "얼마든지", "마음대로", "주저하지", "않고", "곧", "즉시", "바로", "당장", "하자마자", "밖에", "안된다", "하면된다", "그래", "그렇지", "요컨대", "다시", "말하자면", "바꿔", "말하면", "즉", "구체적으로", "말하자면", "시작하여", "시초에", "이상", "허", "헉", "허걱", "바와같이", "해도좋다", "해도된다", "게다가", "더구나", "하물며", "와르르", "팍", "퍽", "펄렁", "동안", "이래", "하고있었다", "이었다", "에서", "로부터", "까지", "예하면", "했어요", "해요", "함께", "같이", "더불어", "마저", "마저도", "양자", "모두", "습니다", "가까스로", "하려고하다", "즈음하여", "다른", "다른", "방면으로", "해봐요", "습니까", "했어요", "말할것도", "없고", "무릎쓰고", "개의치않고", "하는것만", "못하다", "하는것이", "낫다", "매", "매번", "들", "모", "어느것", "어느", "로써", "갖고말하자면", "어디", "어느쪽", "어느것", "어느해", "어느", "년도", "라", "해도", "언젠가", "어떤것", "어느것", "저기", "저쪽", "저것", "그때", "그럼", "그러면", "요만한걸", "그래", "그때", "저것만큼", "그저", "이르기까지", "할", "줄", "안다", "할", "힘이", "있다", "너", "너희", "당신", "어찌", "설마", "차라리", "할지언정", "할지라도", "할망정", "할지언정", "구토하다", "게우다", "토하다", "메쓰겁다", "옆사람", "퉤", "쳇", "의거하여", "근거하여", "의해", "따라", "힘입어", "그", "다음", "버금", "두번째로", "기타", "첫번째로", "나머지는", "그중에서", "견지에서", "형식으로", "쓰여", "입장에서", "위해서", "단지", "의해되다", "하도록시키다", "뿐만아니라", "반대로", "전후", "전자", "앞의것", "잠시", "잠깐", "하면서", "그렇지만", "다음에", "그러한즉", "그런즉", "남들", "아무거나", "어찌하든지", "같다", "비슷하다", "예컨대", "이럴정도로", "어떻게", "만약", "만일", "위에서", "서술한바와같이", "인", "듯하다", "하지", "않는다면", "만약에", "무엇", "무슨", "어느", "어떤", "아래윗", "조차", "한데", "그럼에도", "불구하고", "여전히", "심지어", "까지도", "조차도", "하지", "않도록", "않기", "위하여", "때", "시각", "무렵", "시간", "동안", "어때", "어떠한", "하여금", "네", "예", "우선", "누구", "누가", "알겠는가", "아무도", "줄은모른다", "줄은", "몰랏다", "하는", "김에", "겸사겸사", "하는바", "그런", "까닭에", "한", "이유는", "그러니", "그러니까", "때문에", "그", "너희", "그들", "너희들", "타인", "것", "것들", "너", "위하여", "공동으로", "동시에", "하기", "위하여", "어찌하여", "무엇때문에", "붕붕", "윙윙", "나", "우리", "엉엉", "휘익", "윙윙", "오호", "아하", "어쨋든", "만", "못하다    하기보다는", "차라리", "하는", "편이", "낫다", "흐흐", "놀라다", "상대적으로", "말하자면", "마치", "아니라면", "쉿", "그렇지", "않으면", "그렇지", "않다면", "안", "그러면", "아니었다면", "하든지", "아니면", "이라면", "좋아", "알았어", "하는것도", "그만이다", "어쩔수", "없다", "하나", "일", "일반적으로", "일단", "한켠으로는", "오자마자", "이렇게되면", "이와같다면", "전부", "한마디", "한항목", "근거로", "하기에", "아울러", "하지", "않도록", "않기", "위해서", "이르기까지", "이", "되다", "로", "인하여", "까닭으로", "이유만으로", "이로", "인하여", "그래서", "이", "때문에", "그러므로", "그런", "까닭에", "알", "수", "있다", "결론을", "낼", "수", "있다", "으로", "인하여", "있다", "어떤것", "관계가", "있다", "관련이", "있다", "연관되다", "어떤것들", "에", "대해", "이리하여", "그리하여", "여부", "하기보다는", "하느니", "하면", "할수록", "운운", "이러이러하다", "하구나", "하도다", "다시말하면", "다음으로", "에", "있다", "에", "달려", "있다", "우리", "우리들", "오히려", "하기는한데", "어떻게", "어떻해", "어찌됏어", "어때", "어째서", "본대로", "자", "이", "이쪽", "여기", "이것", "이번", "이렇게말하자면", "이런", "이러한", "이와", "같은", "요만큼", "요만한", "것", "얼마", "안", "되는", "것", "이만큼", "이", "정도의", "이렇게", "많은", "것", "이와", "같다", "이때", "이렇구나", "것과", "같이", "끼익", "삐걱", "따위", "와", "같은", "사람들", "부류의", "사람들", "왜냐하면", "중의하나", "오직", "오로지", "에", "한하다", "하기만", "하면", "도착하다", "까지", "미치다", "도달하다", "정도에", "이르다", "할", "지경이다", "결과에", "이르다", "관해서는", "여러분", "하고", "있다", "한", "후", "혼자", "자기", "자기집", "자신", "우에", "종합한것과같이", "총적으로", "보면", "총적으로", "말하면", "총적으로", "대로", "하다", "으로서", "참", "그만이다", "할", "따름이다", "쿵", "탕탕", "쾅쾅", "둥둥", "봐", "봐라", "아이야", "아니", "와아", "응", "아이", "참나", "년", "월", "일", "령", "영", "일", "이", "삼", "사", "오", "육", "륙", "칠", "팔", "구", "이천육", "이천칠", "이천팔", "이천구", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉", "령", "영"]

#---------------------------------------------------유사도함수---------------------------------------------------
def Cosine(vec1, vec2) : # 1-완전일치, 0-완전수직
    result = InnerProduct(vec1,vec2) / (VectorSize(vec1) * VectorSize(vec2))
    return result

def VectorSize(vec) :
    return math.sqrt(sum(math.pow(v,2) for v in vec))

def InnerProduct(vec1, vec2) :
    return sum(v1*v2 for v1,v2 in zip(vec1,vec2))

def Euclidean(vec1, vec2) : # 0-일치, 커질수록 불일치
    return math.sqrt(sum(math.pow((v1-v2),2) for v1,v2 in zip(vec1, vec2)))

def Magnitude_Difference(vec1, vec2) :
    return abs(VectorSize(vec1) - VectorSize(vec2))

def Triangle(vec1, vec2) :
    theta = math.radians(Theta(vec1,vec2))
    return (VectorSize(vec1) * VectorSize(vec2) * math.sin(theta)) / 2

def Theta(vec1, vec2) :
    return math.acos(Cosine(vec1,vec2)) + 10

def Sector(vec1, vec2) :
    ED = Euclidean(vec1, vec2)
    MD = Magnitude_Difference(vec1, vec2)
    theta = Theta(vec1, vec2)
    return math.pi * math.pow((ED+MD),2) * theta/360

def TS_SS(vec1, vec2) :
    return Triangle(vec1, vec2) * Sector(vec1, vec2)
#---------------------------------------------------유사도함수---------------------------------------------------

def make_article_graph(article_list_all):
    graph = networkx.Graph()
    graph.add_nodes_from(article_list_all)
    pairs = list(itertools.combinations(article_list_all, 2))
    for pair in pairs:
        weight = TS_SS(pair[0].vector, pair[1].vector)
        if weight:
            graph.add_edge(pair[0], pair[1], weight=weight)
    return graph

def find_similar_article(articles): # 같은 카테고리 내에서 뉴스끼리 유사도 분석(sim_map 변수)하고 가장 유사한뉴스 5개 인덱스(sim_index) 저장
    MIN = -1
    pairs = list(itertools.combinations(articles, 2)) # pair순서: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

    articles[0].sim_map.append(MIN)
    for pair in pairs:
        similarity = TS_SS(pair[0].vector, pair[1].vector)
        if(articles.index(pair[0]) == len(articles[articles.index(pair[0])].sim_map)): # sim_map에서 같은 기사의 인덱스일 경우
            articles[articles.index(pair[0])].sim_map.append(MIN) # 같은 기사는 유사도-1로 저장
            articles[articles.index(pair[1])].sim_map.append(similarity) # 다른 기사와의 유사도 저장
        else:
            articles[articles.index(pair[0])].sim_map.append(similarity)
            articles[articles.index(pair[1])].sim_map.append(similarity) # 같은 기사는 이미 MIN으로 한번 저장했으므로 다른기사의 sim_map에만 저장
    articles[-1].sim_map.append(MIN)

    for article in articles:
        article.sim_index = get_similar_index(article.sim_map)

def get_similar_index(sim_map): # 같은 카테고리 내에서 가장 유사한 뉴스5개의 인덱스 반환하는 함수
    sim_top5 = []
    for epoch in range(5):
        max = -100
        max_idx = -1
        for idx, val in enumerate(sim_map):
            if (max < val and idx not in sim_top5):
                max = val
                max_idx = idx
        sim_top5.append(max_idx)
    return sim_top5

def make_article_instance(dir_news):
    article_list_all=[]
    list_pol = []; list_soc = []; list_eco = []; list_it = []; list_wor = []; list_spo = []; list_cul = []; list_ent = []

    for dir_each_news in dir_news:
        newsfile_list = glob.glob(dir_each_news + '/*.txt') # 신문사폴더 안의 모든 텍스트파일이름을 리스트에 저장

        print('--------------- [' + dir_each_news[46:] + ' reading] ---------------')
        for idx, fname in enumerate(newsfile_list): # 신문사폴더 안의 텍스트파일 하나씩 처리
            print(' ' + fname + ' processing')
            f = codecs.open(fname, 'r', 'utf-8')

            category=''; list_tem = []
            for idx, l in enumerate(f):
                l_split = l.split('div\t')
                list_tem.append(Article(l_split[0], l_split[1], l_split[2], l_split[3], l_split[4], l_split[5]))
                # 크롤링데이터를 읽어와서 모든 기사를 각 인스턴스로 생성
                if idx==1:
                    category=l_split[1]
            # 카테고리별로 분류
            if(category=='정치'): list_pol = list_pol+list_tem
            elif(category=='사회'): list_soc = list_soc+list_tem
            elif (re.search('경제', category)): list_eco = list_eco+list_tem
            elif (re.search('IT', category)): list_it = list_it+list_tem
            elif (category == '세계' or category == '국제'): list_wor = list_wor+list_tem
            elif (category == '스포츠'): list_spo = list_spo+list_tem
            elif (category == '문화'): list_cul = list_cul+list_tem
            else: list_ent = list_ent+list_tem
    article_list_all = list_pol + list_soc + list_eco + list_it + list_wor + list_spo + list_cul + list_ent

    ''' 기사 2160개(7.27MB)로 모델 트레이닝
    doc_to_vector(article_list_all)'''
    #doc2vectorizer.save('trained.model')
    #doc2vectorizer.wv.save_word2vec_format('trained.word2vec')

    doc2vectorizer = models.Doc2Vec.load('trained.model') # 트레이닝된 모델 로드

    for idx, article in enumerate(article_list_all):
        article.vector = doc2vectorizer.infer_vector(doc_words=article.keywords) # 각 list(list_pol, list_soc등).vector 로도 확인할 수 있다.

    for category_list in article_list_all:
        find_similar_article(category_list)

    ''' 시뮬레이션용 (정치카테고리)
    find_similar_article(list_pol)
    print('\n-----------------------원문-----------------------\n' + list_pol[0].title)
    print('\n--------------------유사한 기사--------------------\n' + list_pol[list_pol[0].sim_index[0]].title)
    print(list_pol[list_pol[0].sim_index[1]].title)
    print(list_pol[list_pol[0].sim_index[2]].title)
    print(list_pol[list_pol[0].sim_index[3]].title)
    print(list_pol[list_pol[0].sim_index[4]].title)'''

    ############################# 데이터베이스에 저장하는 코드 들어갈 곳#############################
    # 유사한 뉴스 인덱스는 같은 카테고리안에서 해야함 (list_pol, list_soc, ... 등등 따로따로)


    ############################# 데이터베이스에 저장하는 코드 들어갈 곳#############################

def doc_to_vector(article_list):
    # nltk.Text().concordance('단어') # 문서중 해당단어가 포함된 문장 추출
    # nltk.Text().similar('단어') # 문서 내 가장 비슷한 단어하나
    # model = gensim.models.Doc2Vec.load("/home/wiki/stock/model/wiki_pos_tokenizer_without_taginfo.model") # 모델을 불러오는 코드
    # corpora.Dictionary('텍스트') # 단어들의 id를 생성하고 사전을 구성
    # dorpus = [dictionary.doc2bow(text) fo text in texts] # vector로 변환한 bag of words(corpus) 결과물을 얻을 수 있다.
    docs = []
    TaggedDocument = namedtuple('TaggedDocument', 'words tags')  # AnalyzeDocument라는 이름을 가지고 words와 tags를 인자로 가진 클래스 인스턴스
    for i, article in enumerate(article_list):
        tags = [i]
        docs.append(TaggedDocument(article.keywords, tags))
    model = models.doc2vec.Doc2Vec(docs, size=300, alpha=0.025, min_alpha=0.025, min_count=2, workers=4)

    for epoch in range(10):
        model.train(docs, total_examples=model.corpus_count,epochs=model.iter)
        model.alpha -=0.002
        model.min_alpha = model.alpha

    model.save('trained.model')
    model.wv.save_word2vec_format('trained.word2vec')

def get_keywords(text): # text에서 키워드 구하는 함수
    #keywords = [noun for noun in _twitter.nouns(text) if noun not in _stopwords]
    keywords = text.split()
    keywords_ = remove_duplicate(keywords)
    return keywords_

def remove_duplicate(n_list): # 리스트 내의 항목 중복제거하는 함수
    new_list = []
    for i in n_list:
        if i not in new_list:
            new_list.append(i)
    return new_list

class Article: # 뉴스 기사를 하나의 클래스
    def __init__(self, source, category, title, summary, url, body_main):
        self.source = source # 신문사
        self.category = category # 카테고리(정치, 사회, 연예 등등)
        self.title = title # 기사 제목
        self.summary = summary # 기사 요약
        self.url = url # 기사 링크
        self.body_main = body_main # 기사 본문
        self.keywords = get_keywords(body_main) # 기사의 키워드 리스트
        self.vector = [] # 기사의 벡터
        self.sim_map = [] # 다른 기사들과의 유사도를 저장한 리스트
        self.sim_index = [] # sim_map에서 유사도가 가장 높은 기사 5개의 인덱스를 저장한 리스트

if __name__ == "__main__":
    t = time.time()
    dir_nocut = 'C:/Users/sanghak/PycharmProjects/crawled_data/nocut'
    dir_nocut170517 = 'C:/Users/sanghak/PycharmProjects/crawled_data/nocut170517'
    dir_yonhap = 'C:/Users/sanghak/PycharmProjects/crawled_data/yonhap'
    dir_yonhap170517 = 'C:/Users/sanghak/PycharmProjects/crawled_data/yonhap170517'
    dir_newsis = 'C:/Users/sanghak/PycharmProjects/crawled_data/newsis'
    dir_newsis170517 = 'C:/Users/sanghak/PycharmProjects/crawled_data/newsis170517'

    # 트레이닝 셋(기사 2160개)
    #dir_news = [dir_nocut, dir_nocut170517, dir_yonhap, dir_yonhap170517,  dir_newsis, dir_newsis170517]

    # 실제 처리할 데이터(기사 1080개)
    dir_news = [dir_nocut, dir_yonhap, dir_newsis]

    make_article_instance(dir_news)

    print('실행시간: %.02f초' % (time.time() - t)) # 45.75초 / 기사 1080개
