# utils/function_calling_system.py

import os
import json
import sqlite3
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from utils.prompts.recall_prompts import RecallPrompts

load_dotenv()

# 전역 변수 - 시스템 컴포넌트들
_sqlite_conn = None
_vectorstore = None  
_logical_processor = None
_db_initialized = False

def initialize_sqlite_db(db_path="./data/fda_recalls.db"):
    """SQLite 데이터베이스 연결 초기화 (스레드 안전)"""
    try:
        if not os.path.exists(db_path):
            print(f"❌ SQLite 데이터베이스가 존재하지 않습니다: {db_path}")
            return None
        
        # 🔧 스레드 안전 설정
        conn = sqlite3.connect(
            db_path, 
            check_same_thread=False,  # 스레드 안전성 해제
            timeout=30.0  # 타임아웃 설정
        )
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM recalls")
        total_records = cursor.fetchone()['count']
        print(f"✅ SQLite 연결 성공: {total_records}개 레코드")
        
        return conn
        
    except Exception as e:
        print(f"❌ SQLite 연결 실패: {e}")
        return None

def initialize_recall_vectorstore():
    """ChromaDB 벡터스토어 초기화"""
    persist_dir = "./data/chroma_db_recall"
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            print("기존 리콜 벡터스토어를 로드합니다...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name="FDA_recalls"
            )
            
            collection = vectorstore._collection
            doc_count = collection.count()
            print(f"✅ 리콜 벡터스토어 로드 완료 ({doc_count}개 문서)")
            return vectorstore
                
        except Exception as e:
            print(f"⚠️ 벡터스토어 로드 실패: {e}")
            return None
    else:
        print("⚠️ 벡터스토어 폴더가 존재하지 않습니다")
        return None
    
def parse_relative_dates(period_text: str) -> str:
    """상대적 날짜 표현을 절대 연도로 변환 (2025년 기준)"""
    import datetime
    
    current_year = datetime.datetime.now().year  # 2025
    
    # 🔧 올바른 한국어 표현 매핑
    korean_mappings = {
        "올해": str(current_year),           # 2025 ✅
        "작년": str(current_year - 1),       # 2024 ✅  
        "재작년": str(current_year - 2),     # 2023 ✅
        "이번년": str(current_year),         # 2025
        "현재": str(current_year),           # 2025
        "지난해": str(current_year - 1),     # 2024 ✅
        "전년": str(current_year - 1),       # 2024 ✅
        "금년": str(current_year),           # 2025
        "작년도": str(current_year - 1),     # 2024
        "올해년도": str(current_year),       # 2025
    }
    
    period_lower = period_text.lower().strip()
    
    # 한국어 매핑 확인
    for korean, year in korean_mappings.items():
        if korean in period_lower:
            print(f"🔧 날짜 매핑: '{period_text}' → {year}년 (현재: {current_year})")
            return year
    
    # 숫자인 경우 그대로 반환
    if period_text.isdigit() and len(period_text) == 4:
        return period_text
    
    # 인식하지 못한 경우 현재 연도 반환
    print(f"⚠️ 날짜 인식 실패: '{period_text}' → 기본값 {current_year}년 사용")
    return str(current_year)

def translate_to_english(korean_text: str) -> str:
    """한국어 텍스트를 영어로 번역하는 함수"""
    from langchain_openai import ChatOpenAI
    
    try:
        translator = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        translation_prompt = f"""
다음 한국어 텍스트를 영어로 정확히 번역해주세요. 
식품, 리콜, 알레르겐 관련 전문 용어는 FDA 표준 용어를 사용하세요.

한국어: {korean_text}
영어:"""
        
        response = translator.invoke([{"role": "user", "content": translation_prompt}])
        english_text = response.content.strip()
        
        print(f"🔄 번역: '{korean_text}' → '{english_text}'")
        return english_text
        
    except Exception as e:
        print(f"번역 오류: {e}")
        # 번역 실패 시 기존 키워드 매핑 사용
        return korean_text

def get_recall_vectorstore():
    """tab_recall.py 호환용 함수"""
    return initialize_recall_vectorstore()

def _get_system_components():
    global _sqlite_conn, _vectorstore, _db_initialized
    
    if not _db_initialized:
        _sqlite_conn = initialize_sqlite_db()
        _vectorstore = initialize_recall_vectorstore()
        _db_initialized = True
    
    return _sqlite_conn, _vectorstore, None  

# ======================
# Function Calling 도구들
# ======================

@tool
def count_recalls(company: Optional[str] = None,
                 food_type: Optional[str] = None, 
                 allergen: Optional[str] = None,
                 contaminant: Optional[str] = None,
                 year: Optional[str] = None,
                 recall_reason: Optional[str] = None,
                 keyword: Optional[str] = None) -> Dict[str, Any]:  # 🆕 통합 키워드 추가
    """리콜 건수를 세는 함수 (SQLite 기반) - 다중 필드 통합 검색 지원"""
    
    sqlite_conn, _, _ = _get_system_components()
    
    if not sqlite_conn:
        return {"error": "SQLite 데이터베이스 연결 실패"}
    
    try:
        sql = "SELECT COUNT(*) as count FROM recalls WHERE 1=1"
        params = []
        
        # 🆕 통합 키워드 검색 (여러 필드에서 동시 검색)
        if keyword:
            # 🆕 한국어 키워드를 영어로 번역
            english_keyword = translate_to_english(keyword)
            search_terms = [keyword, english_keyword]  # 원본과 번역본 모두 사용
            
            print(f"🔍 검색어: {search_terms}")
            
            search_conditions = []
            for term in search_terms:
                search_conditions.append("""(
                    LOWER(ont_food_type) LIKE LOWER(?) OR 
                    LOWER(ont_food) LIKE LOWER(?) OR
                    LOWER(ont_allergen) LIKE LOWER(?) OR
                    LOWER(ont_contaminant) LIKE LOWER(?) OR
                    LOWER(ont_recall_reason) LIKE LOWER(?) OR
                    LOWER(ont_company) LIKE LOWER(?)
                )""")
                params.extend([f"%{term}%"] * 6)
            
            sql += f" AND ({' OR '.join(search_conditions)})"
        
        # 기존 개별 필터들
        if company:
            english_company = translate_to_english(company)
            company_terms = [company, english_company] if english_company != company else [company]
            company_conditions = []
            for term in company_terms:
                company_conditions.append("LOWER(ont_company) LIKE LOWER(?)")  # company → ont_company
                params.append(f"%{term}%")
            sql += f" AND ({' OR '.join(company_conditions)})"
            
        if food_type:
            english_food_type = translate_to_english(food_type)
            food_type_terms = [food_type, english_food_type] if english_food_type != food_type else [food_type]
            food_type_conditions = []
            for term in food_type_terms:
                food_type_conditions.append("LOWER(ont_food_type) LIKE LOWER(?)")  # food_type → ont_food_type
                params.append(f"%{term}%")
            sql += f" AND ({' OR '.join(food_type_conditions)})"
            
        if allergen:
            english_allergen = translate_to_english(allergen)
            allergen_terms = [allergen, english_allergen] if english_allergen != allergen else [allergen]
            allergen_conditions = []
            for term in allergen_terms:
                allergen_conditions.append("LOWER(ont_allergen) LIKE LOWER(?)")  # allergen → ont_allergen
                params.append(f"%{term}%")
            sql += f" AND ({' OR '.join(allergen_conditions)})"

        if contaminant:
            english_contaminant = translate_to_english(contaminant)
            contaminant_terms = [contaminant, english_contaminant] if english_contaminant != contaminant else [contaminant]
            contaminant_conditions = []
            for term in contaminant_terms:
                contaminant_conditions.append("LOWER(ont_contaminant) LIKE LOWER(?)")  # contaminant → ont_contaminant
                params.append(f"%{term}%")
            sql += f" AND ({' OR '.join(contaminant_conditions)})"
            
        if recall_reason:
            english_recall_reason = translate_to_english(recall_reason)
            recall_reason_terms = [recall_reason, english_recall_reason] if english_recall_reason != recall_reason else [recall_reason]
            recall_reason_conditions = []
            for term in recall_reason_terms:
                recall_reason_conditions.append("LOWER(ont_recall_reason) LIKE LOWER(?)")  # recall_reason → ont_recall_reason
                params.append(f"%{term}%")
            sql += f" AND ({' OR '.join(recall_reason_conditions)})"
        if year:
            sql += " AND strftime('%Y', effective_date) = ?"
            params.append(year)
        
        print(f"🔧 통합 검색 SQL: {sql}")
        print(f"🔧 파라미터: {params}")
        
        cursor = sqlite_conn.cursor()
        cursor.execute(sql, params)
        result = cursor.fetchone()
        
        return {
            "count": result["count"],
            "filters": {
                "company": company, "food_type": food_type,
                "allergen": allergen, "contaminant": contaminant, 
                "year": year, "recall_reason": recall_reason,
                "keyword": keyword  # 🆕 추가
            },
            "search_fields": "multiple" if keyword else "specific",  # 🆕 검색 방식 표시
            "query_type": "unified_count"
        }
        
    except Exception as e:
        return {"error": f"SQL 카운팅 오류: {e}"}

@tool
def rank_by_field(field: str, limit: int = 10, 
                 company: Optional[str] = None,
                 food_type: Optional[str] = None,
                 year: Optional[str] = None) -> Dict[str, Any]:
    """필드별 순위 (다층 필드 검색 지원)"""
    
    sqlite_conn, _, _ = _get_system_components()
    
    if not sqlite_conn:
        return {"error": "SQLite 데이터베이스 연결 실패"}
    
    try:
        cursor = sqlite_conn.cursor()
        
        # 🆕 다층 필드 매핑
        field_mapping = {
        "company": "ont_company",
        "food_type": "ont_food_type",           # 단일 필드
        "food": "ont_food",             
        "recall_reason": "ont_recall_reason",
        "allergen": "ont_allergen",
        "contaminant": "ont_contaminant"
    }
        
        db_fields = field_mapping.get(field.lower(), ["company"])
        if isinstance(db_fields, str):
            db_fields = [db_fields]
        
        # 🆕 다층 검색 SQL 구성
        if len(db_fields) > 1:
            # COALESCE로 다층 필드 통합
            select_field = f"COALESCE({', '.join(db_fields)}) as name"
            where_conditions = []
            for db_field in db_fields:
                where_conditions.append(f"{db_field} IS NOT NULL AND {db_field} != '' AND {db_field} != 'N/A'")
            where_clause = "(" + " OR ".join(where_conditions) + ")"
        else:
            select_field = f"{db_fields[0]} as name"
            where_clause = f"{db_fields[0]} IS NOT NULL AND {db_fields[0]} != '' AND {db_fields[0]} != 'N/A'"
        
        sql = f"""
            SELECT {select_field}, COUNT(*) as count 
            FROM recalls 
            WHERE {where_clause}
        """
        params = []
        
        # 기존 필터 조건들
        if company and db_fields[0] != "ont_company":
            sql += " AND LOWER(ont_company) LIKE LOWER(?)"
            params.append(f"%{company}%")
            
        if food_type and db_fields[0] != "ont_food_type":
            sql += " AND LOWER(ont_food_type) LIKE LOWER(?)"
            params.append(f"%{food_type}%")
            
        if year:
            sql += " AND strftime('%Y', effective_date) = ?"
            params.append(year)

        sql += f" GROUP BY name ORDER BY count DESC LIMIT ?"
        params.append(limit)
        
        print(f"🔧 다층 검색 SQL: {sql}")
        print(f"🔧 파라미터: {params}")
        
        cursor.execute(sql, params)
        results = [{"name": row["name"], "count": row["count"]} for row in cursor.fetchall()]
        
        return {
            "ranking": results,
            "field": field,
            "search_fields": db_fields,  # 검색한 필드들 표시
            "total_items": len(results),
            "query_type": "multilayer_ranking"
        }
        
    except Exception as e:
        return {"error": f"다층 순위 조회 오류: {e}"}
    
@tool 
def get_monthly_trend(months: int = 12,
                     food_type: Optional[str] = None,
                     company: Optional[str] = None) -> Dict[str, Any]:
    """월별 리콜 트렌드를 가져오는 함수 (SQLite 기반)"""
    sqlite_conn, _, _ = _get_system_components()
    
    if not sqlite_conn:
        return {"error": "SQLite 데이터베이스 연결 실패"}
    
    try:
        sql = """
            SELECT strftime('%Y-%m', effective_date) as month, COUNT(*) as count
            FROM recalls 
            WHERE effective_date IS NOT NULL
        """
        params = []
        
        if food_type:
            sql += " AND LOWER(ont_food_type) LIKE LOWER(?)"
            params.append(f"%{food_type}%")
        if company:
            sql += " AND LOWER(ont_company) LIKE LOWER(?)"
            params.append(f"%{company}%")
        
        sql += " GROUP BY month ORDER BY month DESC LIMIT ?"
        params.append(months)
        
        cursor = sqlite_conn.cursor()
        cursor.execute(sql, params)
        results = [{"month": row["month"], "count": row["count"]} for row in cursor.fetchall()]
        
        return {
            "trend": results, 
            "months": months,
            "query_type": "trend"
        }
        
    except Exception as e:
        return {"error": f"트렌드 조회 오류: {e}"}

@tool
def compare_periods(period1: str, period2: str, 
                   metric: str = "count",
                   include_reasons: bool = False) -> Dict[str, Any]:  # 🆕 매개변수 추가
    """기간별 비교 분석 함수 (자연어 날짜 지원 + 사유별 분석)"""
    
    sqlite_conn, _, _ = _get_system_components()
    
    if not sqlite_conn:
        return {"error": "SQLite 데이터베이스 연결 실패"}
    
    try:
        # 🆕 상대적 날짜 표현을 절대 연도로 변환
        actual_period1 = parse_relative_dates(period1)
        actual_period2 = parse_relative_dates(period2)
        
        print(f"🔧 날짜 변환: '{period1}' → {actual_period1}, '{period2}' → {actual_period2}")
        
        cursor = sqlite_conn.cursor()
        
        def get_period_data(period: str):
            if len(period) == 4:  # 연도
                date_filter = "strftime('%Y', effective_date) = ?"
            elif len(period) == 7:  # 연월
                date_filter = "strftime('%Y-%m', effective_date) = ?"
            else:
                return None
            
            result_data = {}
            
            # 기본 카운트
            if metric == "count":
                sql = f"SELECT COUNT(*) as value FROM recalls WHERE {date_filter}"
            elif metric == "companies":
                sql = f"SELECT COUNT(DISTINCT ont_company) as value FROM recalls WHERE {date_filter} AND company IS NOT NULL"
            elif metric == "food_types":
                sql = f"SELECT COUNT(DISTINCT ont_food_type) as value FROM recalls WHERE {date_filter} AND food_type IS NOT NULL"
            
            cursor.execute(sql, [period])
            result = cursor.fetchone()
            result_data["total"] = result["value"] if result else 0
            
            # 🆕 리콜 사유별 분석 추가 (리콜 원인 비교 질문용)
            if include_reasons or "원인" in str(period) or "사유" in str(period):
                cursor.execute(f"""
                    SELECT recall_reason, COUNT(*) as count 
                    FROM recalls 
                    WHERE {date_filter} AND recall_reason IS NOT NULL
                    GROUP BY recall_reason 
                    ORDER BY count DESC 
                    LIMIT 5
                """, [period])
                reasons = [{"reason": row["recall_reason"], "count": row["count"]} for row in cursor.fetchall()]
                result_data["top_reasons"] = reasons
            
            return result_data
        
        data1 = get_period_data(actual_period1)
        data2 = get_period_data(actual_period2)
        
        if data1 is None or data2 is None:
            return {"error": "잘못된 기간 형식입니다. YYYY 또는 YYYY-MM 형식을 사용하세요."}
        
        # 변화율 계산 (총 건수 기준)
        value1 = data1.get("total", 0)
        value2 = data2.get("total", 0)
        change = value2 - value1
        change_percent = (change / value1 * 100) if value1 > 0 else 0
        
        return {
            "period1": {"period": f"{period1}({actual_period1})", "data": data1},  # 🆕 구조 변경
            "period2": {"period": f"{period2}({actual_period2})", "data": data2},  # 🆕 구조 변경
            "change": change,
            "change_percent": round(change_percent, 1),
            "metric": metric,
            "query_type": "enhanced_period_comparison",
            "actual_periods": [actual_period1, actual_period2]
        }
        
    except Exception as e:
        return {"error": f"기간 비교 오류: {e}"}

@tool
def search_recall_cases(query: str, limit: int = 5) -> Dict[str, Any]:
    """ChromaDB 기반 의미적 검색 (한영 번역 지원)"""
    

    _, vectorstore, _ = _get_system_components()
    
    if not vectorstore:
        return {"error": "ChromaDB 벡터스토어 연결 실패"}
    
    try:
        search_queries = []
        search_queries.append(query)  # 원본

        # 🆕 전체 쿼리 번역
        english_query = translate_to_english(query)
        if english_query != query:
            search_queries.append(english_query)

        print(f"🔍 검색어 확장: {search_queries}")
        
        # 핵심 키워드 매핑
        translations = {
            "소스를 포함한 복합식품": "sauce processed food",
            "복합 가공식품": "processed foods",
            "살모넬라": "Salmonella",
            "대장균": "E.coli",
            "리스테리아": "Listeria",
            "알레르겐": "allergen",
            "오염": "contamination",
            "리콜 사례": "recall cases"
        }
        
        # 전체 문장 번역
        english_query = query
        for ko_term, en_term in translations.items():
            if ko_term in english_query:
                english_query = english_query.replace(ko_term, en_term)
        
        if english_query != query:
            search_queries.append(english_query)
        
        # 개별 키워드 추출 및 번역
        for ko_term, en_term in translations.items():
            if ko_term in query:
                search_queries.append(en_term)
        
        print(f"🔍 검색어 확장: {search_queries}")
        
        all_docs = []
        seen_urls = set()
        
        # 각 검색어로 검색 실행
        for search_query in search_queries:
            docs = vectorstore.similarity_search(
                search_query, 
                k=limit * 2,
                filter={"document_type": "recall"}
            )
            
            for doc in docs:
                url = doc.metadata.get("url", "")
                if url not in seen_urls:
                    all_docs.append(doc)
                    seen_urls.add(url)
        
        # 상위 결과 선택
        selected_docs = all_docs[:limit]
        

        # 결과 포맷팅
        cases = []
        for doc in selected_docs:
            cases.append({
                "title": doc.metadata.get("title", ""),
                "company": doc.metadata.get("ont_company", "Unknown"),
                "food": doc.metadata.get("ont_food", ""),
                "reason": doc.metadata.get("ont_recall_reason", ""),
                "allergen": doc.metadata.get("ont_allergen", ""),
                "date": doc.metadata.get("effective_date", ""),
                "url": doc.metadata.get("url", ""),
                "content_preview": doc.page_content[:200] + "..."
            })
        
        return {
            "cases": cases,
            "total_found": len(cases),
            "original_query": query,
            "search_queries": search_queries,
            "search_method": "multilingual_search"
        }
        
    except Exception as e:
        return {"error": f"검색 오류: {e}"}

@tool
def filter_exclude_conditions(exclude_terms: List[str],
                             include_terms: Optional[List[str]] = None) -> Dict[str, Any]:
    """특정 조건을 제외한 데이터를 필터링하는 함수 (ChromaDB 기반)
    
    Args:
        exclude_terms: 제외할 조건들 (예: ["유제품", "Dairy"])
        include_terms: 포함할 조건들 (선택사항)
    """
    _, _, logical_processor = _get_system_components()
    
    if not logical_processor:
        return {"error": "논리 프로세서 초기화 실패 (ChromaDB 필요)"}
    
    try:
        question = f"리콜 사례에서 {', '.join(exclude_terms)}를 제외한 데이터"
        if include_terms:
            question = f"{', '.join(include_terms)} 중에서 {', '.join(exclude_terms)}를 제외한 데이터"
        
        result = logical_processor.process_logical_query(question)
        
        if 'error' in result:
            return {"error": result['error']}
        
        filtered_count = 0
        excluded_count = 0
        
        if result.get('type') == 'exclude':
            filtered_count = result['result'].get('final_count', 0)
            excluded_count = result['result'].get('excluded_count', 0)
        
        return {
            "filtered_count": filtered_count,
            "excluded_count": excluded_count,
            "exclude_terms": exclude_terms,
            "include_terms": include_terms or [],
            "details": result
        }
        
    except Exception as e:
        return {"error": f"제외 필터링 오류: {e}"}
    
@tool
def exclude_contaminant_search(include_reason: str, 
                              exclude_contaminant: str,
                              limit: int = 10) -> Dict[str, Any]:
    """특정 오염물질을 제외한 리콜 사례 검색 (SQLite + ChromaDB 하이브리드)"""
    
    sqlite_conn, vectorstore, _ = _get_system_components()
    
    if not sqlite_conn:
        return {"error": "SQLite 데이터베이스 연결 실패"}
    
    try:
        # 1단계: SQLite에서 조건부 필터링
        sql = """
        SELECT id, title, ont_company, ont_food_type, ont_food, 
            ont_recall_reason, ont_allergen, ont_contaminant, effective_date, url
        FROM recalls 
        WHERE LOWER(ont_recall_reason) LIKE LOWER(?)
        AND NOT (
            LOWER(ont_contaminant) LIKE LOWER(?) OR
            LOWER(ont_allergen) LIKE LOWER(?) OR
            LOWER(ont_food) LIKE LOWER(?)
        )
        ORDER BY effective_date DESC
        LIMIT ?
        """
        
        params = [
            f"%{include_reason}%",
            f"%{exclude_contaminant}%", 
            f"%{exclude_contaminant}%",
            f"%{exclude_contaminant}%",
            limit * 2
        ]
        
        print(f"🔧 제외 검색 SQL: {sql}")
        print(f"🔧 파라미터: {params}")
        
        cursor = sqlite_conn.cursor()
        cursor.execute(sql, params)
        results = cursor.fetchall()
        
        # 2단계: ChromaDB에서 추가 사례 검색 (영어 키워드)
        additional_cases = []
        if vectorstore and len(results) < limit:
            english_queries = [
                f"{include_reason} -salmonella",
                "bacterial contamination listeria",
                "bacterial contamination cronobacter", 
                "bacterial contamination clostridium"
            ]
            
            for query in english_queries:
                try:
                    docs = vectorstore.similarity_search(
                        query, 
                        k=5,
                        filter={"document_type": "recall"}
                    )
                    
                    for doc in docs:
                        contaminant = doc.metadata.get("ont_contaminant", "").lower()
                        if exclude_contaminant.lower() not in contaminant:
                            additional_cases.append(doc)
                            
                except Exception as e:
                    print(f"ChromaDB 검색 오류: {e}")
        
        # 3단계: 결과 통합 및 포맷팅
        final_cases = []
        
        # SQLite 결과 포맷팅
        for row in results:
            final_cases.append({
                "title": row["title"],
                "company": row["ont_company"],
                "food_type": row["ont_food_type"],
                "food_product": row["ont_food"],
                "reason": row["ont_recall_reason"],
                "contaminant": row["ont_contaminant"],
                "allergen": row["ont_allergen"],
                "date": row["effective_date"],
                "url": row["url"]
            })
        
        # ChromaDB 결과 추가
        for doc in additional_cases[:limit-len(final_cases)]:
            final_cases.append({
                "title": doc.metadata.get("title", ""),
                "company": doc.metadata.get("ont_company", "Unknown"),
                "food_type": doc.metadata.get("ont_food_type", ""),
                "food_product": doc.metadata.get("ont_food", ""),
                "reason": doc.metadata.get("ont_recall_reason", ""),
                "contaminant": doc.metadata.get("ont_contaminant", ""),
                "allergen": doc.metadata.get("ont_allergen", ""),
                "date": doc.metadata.get("effective_date", ""),
                "url": doc.metadata.get("url", "")
            })
        
        return {
            "cases": final_cases[:limit],
            "total_count": len(final_cases),
            "include_reason": include_reason,
            "exclude_contaminant": exclude_contaminant,
            "sqlite_results": len(results),
            "chromadb_results": len(additional_cases),
            "query_type": "exclude_contaminant_search"
        }
        
    except Exception as e:
        return {"error": f"제외 검색 오류: {e}"}
    
@tool
def filter_by_conditions(include_conditions: Optional[List[str]] = None,
                        exclude_conditions: Optional[List[str]] = None,
                        food_type: Optional[str] = None,
                        year: Optional[str] = None,
                        contaminants: Optional[List[str]] = None) -> Dict[str, Any]:  # 🆕 contaminants 파라미터 추가
    """조건별 필터링 함수 (SQLite 기반) - OR 조건 지원"""
    
    sqlite_conn, _, _ = _get_system_components()
    
    if not sqlite_conn:
        return {"error": "SQLite 데이터베이스 연결 실패"}
    
    try:
        sql = """
        SELECT id, title, ont_company, ont_food_type, ont_food, 
               ont_recall_reason, ont_allergen, ont_contaminant, effective_date, url
        FROM recalls 
        WHERE 1=1
        """
        params = []
        
        # 식품 유형 필터
        if food_type:
            english_food_type = translate_to_english(food_type)
            food_type_terms = [food_type, english_food_type] if english_food_type != food_type else [food_type]
            food_type_conditions = []
            for term in food_type_terms:
                food_type_conditions.append("LOWER(ont_food_type) LIKE LOWER(?)")
                params.append(f"%{term}%")
            sql += f" AND ({' OR '.join(food_type_conditions)})"
        
        # 🆕 오염물질 OR 조건 (ont_contaminant 필드 사용)
        if contaminants:
            contaminant_conditions = []
            for contaminant in contaminants:
                english_contaminant = translate_to_english(contaminant)
                contaminant_terms = [contaminant, english_contaminant] if english_contaminant != contaminant else [contaminant]
                
                for term in contaminant_terms:
                    contaminant_conditions.append("LOWER(ont_contaminant) LIKE LOWER(?)")
                    params.append(f"%{term}%")
            
            sql += f" AND ({' OR '.join(contaminant_conditions)})"
        
        # 기존 포함 조건 (ont_contaminant 추가)
        if include_conditions:
            expanded_include = []
            for condition in include_conditions:
                expanded_include.append(condition)
                translated = translate_to_english(condition)
                if translated != condition:
                    expanded_include.append(translated)
            
            include_sql = []
            for condition in expanded_include:
                include_sql.append("""(
                    LOWER(ont_recall_reason) LIKE LOWER(?) OR 
                    LOWER(ont_food_type) LIKE LOWER(?) OR 
                    LOWER(ont_allergen) LIKE LOWER(?) OR
                    LOWER(ont_contaminant) LIKE LOWER(?) OR  # 🆕 추가
                    LOWER(ont_food) LIKE LOWER(?)
                )""")
                params.extend([f"%{condition}%"] * 5)
            
            sql += " AND (" + " OR ".join(include_sql) + ")"
        
        # 기존 제외 조건들...
        
        sql += " ORDER BY effective_date DESC LIMIT 10"
        
        print(f"🔧 개선된 필터링 SQL: {sql}")
        print(f"🔧 파라미터: {params}")
        
        cursor = sqlite_conn.cursor()
        cursor.execute(sql, params)
        results = cursor.fetchall()
        
        # 결과 포맷팅
        cases = []
        for row in results:
            cases.append({
                "title": row["title"],
                "company": row["ont_company"],
                "food_type": row["ont_food_type"],
                "food_product": row["ont_food"],
                "reason": row["ont_recall_reason"],
                "contaminant": row["ont_contaminant"],  # 🆕 추가
                "allergen": row["ont_allergen"],
                "date": row["effective_date"],
                "url": row["url"]
            })
        
        return {
            "cases": cases,
            "total_count": len(cases),
            "include_conditions": include_conditions or [],
            "exclude_conditions": exclude_conditions or [],
            "contaminants": contaminants or [],  # 🆕 추가
            "query_type": "enhanced_filtered_search"
        }
        
    except Exception as e:
        return {"error": f"필터링 오류: {e}"}

@tool  
def compare_years(year1: str, year2: str, 
                 analysis_type: str = "total") -> Dict[str, Any]:
    """연도별 비교 분석 (자연어 날짜 지원 강화)"""
    
    sqlite_conn, _, _ = _get_system_components()
    
    if not sqlite_conn:
        return {"error": "SQLite 데이터베이스 연결 실패"}
    
    try:
        # 🚀 향상된 날짜 변환 (더 명확한 로직)
        actual_year1 = parse_relative_dates(year1)
        actual_year2 = parse_relative_dates(year2)
        
        # 🆕 날짜 변환 검증
        if not actual_year1.isdigit() or not actual_year2.isdigit():
            return {"error": f"날짜 변환 실패: '{year1}' → {actual_year1}, '{year2}' → {actual_year2}"}
        
        print(f"🔧 연도 변환 확인: '{year1}' → {actual_year1}, '{year2}' → {actual_year2}")
        
        cursor = sqlite_conn.cursor()
        
        # 각 연도별 데이터 조회
        results = {}
        for original_year, actual_year in [(year1, actual_year1), (year2, actual_year2)]:
            
            # 🆕 연도별 데이터 존재 확인
            cursor.execute("SELECT COUNT(*) as total FROM recalls WHERE strftime('%Y', effective_date) = ?", [actual_year])
            total_count = cursor.fetchone()["total"]
            
            if analysis_type == "total":
                results[f"{original_year}({actual_year})"] = {
                    "total": total_count,
                    "year": actual_year  # 🆕 명시적 연도 추가
                }
                
            elif analysis_type == "reasons":
                # 🔧 올바른 컬럼명 사용
                cursor.execute("""
                    SELECT ont_recall_reason, COUNT(*) as count 
                    FROM recalls 
                    WHERE strftime('%Y', effective_date) = ? AND ont_recall_reason IS NOT NULL
                    GROUP BY ont_recall_reason 
                    ORDER BY count DESC 
                    LIMIT 5
                """, [actual_year])
                
                reasons = [{"reason": row["ont_recall_reason"], "count": row["count"]} for row in cursor.fetchall()]
                results[f"{original_year}({actual_year})"] = {
                    "top_reasons": reasons,
                    "total_with_reasons": len(reasons),
                    "year": actual_year  # 🆕 명시적 연도 추가
                }
        
        # 🆕 변화율 계산 추가 (total 타입일 때)
        change_info = {}
        if analysis_type == "total":
            year1_count = results[f"{year1}({actual_year1})"]["total"]
            year2_count = results[f"{year2}({actual_year2})"]["total"]
            change = year2_count - year1_count
            change_percent = (change / year1_count * 100) if year1_count > 0 else 0
            
            change_info = {
                "change": change,
                "change_percent": round(change_percent, 1),
                "trend": "증가" if change > 0 else "감소" if change < 0 else "변화없음"
            }
        
        return {
            "comparison": results,
            "year1": f"{year1}({actual_year1})",
            "year2": f"{year2}({actual_year2})",
            "analysis_type": analysis_type,
            "actual_years": [actual_year1, actual_year2],
            "change_info": change_info,  # 🆕 변화 정보 추가
            "query_type": "enhanced_year_comparison"
        }
        
    except Exception as e:
        print(f"❌ 연도 비교 오류: {e}")
        return {"error": f"연도 비교 오류: {e}"}

# ======================
# 메인 시스템 클래스
# ======================

class FunctionCallRecallSystem:
    """Function Calling 기반 하이브리드 리콜 시스템 - 전문 프롬프트 통합"""
    
    def __init__(self):
        self.tools = [
            count_recalls, rank_by_field, get_monthly_trend, 
            compare_periods, search_recall_cases, filter_by_conditions,
            compare_years, exclude_contaminant_search 
        ]
        
        # OpenAI Function Calling 모델
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        ).bind_tools(self.tools)
        
        # 🆕 답변 생성용 LLM (프롬프트 템플릿 적용)
        self.answer_llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.3
        )
    
    def process_question(self, question: str, chat_history: List = None) -> Dict[str, Any]:
        """Function Calling으로 질문 처리 - 기존과 동일"""
        
        if chat_history is None:
            chat_history = []
        
        try:
            # 하이브리드 시스템 프롬프트 (기존과 동일)
            system_prompt = """
당신은 FDA 리콜 데이터 전문 분석가입니다. 
사용자의 질문을 분석하여 적절한 함수들을 호출해서 정확한 답변을 제공하세요.

🔧 ChromaDB 검색 시 주의사항:
- 한국어 질문이지만 데이터는 영어로 저장되어 있음
- 검색 시 영어 키워드 우선 사용
- "소스 복합식품" → "sauce Processed Foods"로 검색

🗓️ 현재 시점 정보 (자동 업데이트):
- 현재 연도: {current_year}년
- 작년: {last_year}년  
- 재작년: {two_years_ago}년

⚠️ **날짜 함수 호출 시 절대 규칙**:
- 사용자가 "작년"이라고 하면 → year1="작년" 또는 year2="작년"으로 그대로 전달
- 사용자가 "올해"라고 하면 → year1="올해" 또는 year2="올해"로 그대로 전달  
- 절대로 "작년"을 "2024"로 바꾸거나 "올해"를 "2025"로 바꾸지 말고 원본 그대로 전달

❌ 잘못된 호출:
- compare_periods(period1="2024", period2="2025")  ← 이렇게 하지 마세요
- compare_years(year1="2024", year2="2025")  ← 이렇게 하지 마세요

✅ 올바른 호출:
- compare_periods(period1="작년", period2="올해")  ← 이렇게 하세요
- compare_years(year1="작년", year2="올해")  ← 이렇게 하세요

🎯 함수 선택 가이드라인:

1. **SQLite 함수** (수치형/통계 질문):
   - "총 건수", "몇 건" → count_recalls()
   - "상위 N개", "순위", "가장 많은" → rank_by_field()  
   - "월별 트렌드", "증가/감소" → get_monthly_trend()
   - "2023년 vs 2024년" → compare_periods()
   - "작년과 올해 리콜 원인 비교" → compare_years("작년", "올해", analysis_type="reasons")

2. **ChromaDB 함수** (내용/사례 검색):
   - "사례 알려줘", "어떤 제품", "구체적인 내용" → search_recall_cases()

예시:
- "작년과 올해 리콜 비교" → compare_periods("작년", "올해")  # {last_year} vs {current_year}
- "전년 대비 증가율" → compare_periods("전년", "현재")      # {last_year} vs {current_year}

**중요**: 작년={last_year}년, 올해={current_year}년으로 정확히 인식하여 함수 호출하세요.
**특정 식품 카테고리(계란, 우유, 견과류 등) 질문은 keyword 파라미터로 통합 검색하세요.**

3. **조건별 필터링** (복합 조건 + OR 조건 지원):
   - "A 중에서 B를 제외한" → filter_by_conditions()
   - "해산물에서 살모넬라 또는 리스테리아" → filter_by_conditions(food_type="해산물", contaminants=["살모넬라", "리스테리아"])
   - "과자류에서 견과류 또는 우유 알레르겐" → filter_by_conditions(food_type="과자류", include_conditions=["견과류", "우유"])

예시:
- "해산물 중에서 살모넬라균 또는 리스테리아가 원인인 사례" → filter_by_conditions(food_type="해산물", contaminants=["살모넬라", "리스테리아"])
- "유제품에서 화학물질 또는 이물질" → filter_by_conditions(food_type="유제품", contaminants=["화학물질", "이물질"])

4. **특수 검색** (제외 조건):
   - "A를 제외한 B" → exclude_contaminant_search()
   - "살모넬라 빼고 세균 오염" → exclude_contaminant_search("bacterial", "salmonella")

5. **통합 키워드 검색** (특정 카테고리/식품 관련):
   - "계란 관련 리콜 총 몇 건?" → count_recalls(keyword="계란")
   - "우유 제품 리콜" → count_recalls(keyword="우유")  
   - "견과류 총 건수" → count_recalls(keyword="견과류")
   - keyword 파라미터는 모든 관련 필드에서 자동 통합 검색

🔧 중요한 매핑:
- "복합 가공식품" = "Processed Foods"
- "주요 리콜 사유" = field="ont_recall_reason"으로 순위 분석
- "살모넬라" → ChromaDB에서 의미적 검색

🔧 다층 필드 검색:
- "육류 vs 해산물" → rank_by_field(field="food") # ont_food 검색
- "제품 유형별" → rank_by_field(field="food_type") # ont_food_type 검색

🔧 키워드 통합 검색:
- "계란", "우유", "견과류" 등 특정 식품 카테고리 질문
- ont_food_type, ont_food, ont_allergen, ont_contaminant 등 모든 관련 필드에서 동시 검색  # 수정
- 한국어 키워드를 영어로 자동 확장 ("계란" → "egg", "eggs")

🗓️ 자연어 날짜 지원:
- "작년과 올해" → 자동으로 2024년과 2025년으로 변환
- "전년 대비" → 2024년과 2025년 비교
- "지난해" → 2024년

예시:
- "2024년 총 리콜 건수는?" → count_recalls(year="2024")
- "계란 관련 리콜 총 몇 건?" → count_recalls(keyword="계란")
- "우유 제품 2023년 리콜" → count_recalls(keyword="우유", year="2023")
- "복합 가공식품 주요 리콜 사유 4가지" → rank_by_field(field="ont_recall_reason", food_type="Processed Foods", limit=4)
- "살모넬라 관련 사례 알려줘" → search_recall_cases("살모넬라")
- "소스를 포함한 복합식품 사례" → search_recall_cases("sauce processed food")
- "작년과 올해 리콜 비교" → compare_periods("작년", "올해")
- "전년 대비 증가율" → compare_periods("전년", "현재")

**수치는 SQLite, 내용 검색은 ChromaDB를 사용하세요.**
**특정 식품 카테고리(계란, 우유, 견과류 등) 질문은 keyword 파라미터로 통합 검색하세요.**
"""
            
            # 대화 메시지 구성
            messages = [
                {"role": "system", "content": system_prompt},
                *[{"role": msg.type, "content": msg.content} for msg in chat_history[-6:]],
                {"role": "user", "content": question}
            ]
            
            # Function Calling 실행
            response = self.llm.invoke(messages)
            
            # 함수 호출이 있는 경우 실행
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"🔧 Function Calls: {len(response.tool_calls)}개")
                
                tool_results = []
                for tool_call in response.tool_calls:
                    func_name = tool_call['name']
                    func_args = tool_call.get('args', {})
                    
                    print(f"  → {func_name}({func_args})")
                    
                    # 함수 실행
                    for tool in self.tools:
                        if tool.name == func_name:
                            result = tool.invoke(func_args)
                            tool_results.append({
                                "function": func_name,
                                "args": func_args,
                                "result": result
                            })
                            break
                
                # 🆕 전문 프롬프트 템플릿으로 최종 답변 생성
                final_answer = self._generate_final_answer(question, tool_results)
                
                return {
                    "answer": final_answer,
                    "function_calls": tool_results,
                    "processing_type": "function_calling"
                }
            else:
                # 일반 답변
                return {
                    "answer": response.content,
                    "function_calls": [],
                    "processing_type": "direct_answer"
                }
                
        except Exception as e:
            return {
                "answer": f"처리 중 오류가 발생했습니다: {e}",
                "function_calls": [],
                "processing_type": "error"
            }
    
    def _generate_final_answer(self, question: str, tool_results: List[Dict]) -> str:
        """전문 프롬프트 템플릿을 활용한 답변 생성"""
        
        if not tool_results:
            return "죄송합니다. 관련 정보를 찾을 수 없습니다."
        
        # 🆕 질문 유형별 프롬프트 선택 및 컨텍스트 구성
        answer_context = self._build_answer_context(tool_results)
        selected_prompt = self._select_prompt_template(question, tool_results)
        
        try:
            # 🆕 전문 프롬프트로 최종 답변 생성
            final_prompt = selected_prompt.format(
                question=question,
                **answer_context
            )
            
            response = self.answer_llm.invoke([
                {"role": "system", "content": "당신은 FDA 리콜 데이터 전문 분석가입니다."},
                {"role": "user", "content": final_prompt}
            ])
            
            return response.content
            
        except Exception as e:
            print(f"답변 생성 오류: {e}")
            # 폴백: 기존 방식 사용
            return self._generate_basic_answer(question, tool_results)
    
    def _select_prompt_template(self, question: str, tool_results: List[Dict]) -> str:
        """질문과 결과 유형에 따른 프롬프트 템플릿 선택"""
        
        # 사례 검색 결과가 있는 경우
        if any("search_recall_cases" == tr["function"] for tr in tool_results):
            return RecallPrompts.RECALL_ANSWER
        
        # 수치/통계 분석 결과
        elif any(tr["function"] in ["count_recalls", "rank_by_field", "get_monthly_trend"] 
                for tr in tool_results):
            return RecallPrompts.NUMERICAL_ANSWER
        
        # 논리 연산/비교 분석 결과  
        elif any(tr["function"] in ["compare_periods", "compare_years", 
                                   "filter_by_conditions", "exclude_contaminant_search"] 
                for tr in tool_results):
            return RecallPrompts.LOGICAL_ANSWER
        
        # 기본: 리콜 답변 템플릿
        else:
            return RecallPrompts.RECALL_ANSWER
    
    def _build_answer_context(self, tool_results: List[Dict]) -> Dict[str, str]:
        """프롬프트 템플릿용 컨텍스트 구성"""
        
        context = {}
        
        # 사례 검색 결과 처리
        search_results = [tr for tr in tool_results if tr["function"] == "search_recall_cases"]
        if search_results:
            cases = search_results[0]["result"].get("cases", [])
            context["recall_context"] = self._format_cases_for_prompt(cases)
        
        # 수치 분석 결과 처리
        numerical_results = [tr for tr in tool_results 
                           if tr["function"] in ["count_recalls", "rank_by_field", "get_monthly_trend"]]
        if numerical_results:
            context.update(self._format_numerical_for_prompt(numerical_results))
        
        # 논리 분석 결과 처리
        logical_results = [tr for tr in tool_results 
                         if tr["function"] in ["compare_periods", "compare_years", 
                                              "filter_by_conditions", "exclude_contaminant_search"]]
        if logical_results:
            context.update(self._format_logical_for_prompt(logical_results))
        
        return context
    
    def _format_cases_for_prompt(self, cases: List[Dict]) -> str:
        """사례 데이터를 프롬프트용 텍스트로 변환"""
        
        if not cases:
            return "관련 사례를 찾을 수 없습니다."

        formatted_cases = []
        
        for i, case in enumerate(cases[:8], 1):  # 최대 8건
            case_text = f"""사례 {i}:
- 회사: {case.get('company', 'Unknown')}
- 제품: {case.get('food', case.get('food_product', 'N/A'))}
- 리콜 사유: {case.get('reason', 'N/A')}
- 알레르겐: {case.get('allergen', 'N/A')}
- 오염물질: {case.get('contaminant', 'N/A')}
- 날짜: {case.get('date', 'N/A')}
- URL: {case.get('url', 'N/A')}"""
            formatted_cases.append(case_text.strip())
        
        return "\n\n".join(formatted_cases)
    
    def _format_numerical_for_prompt(self, numerical_results: List[Dict]) -> Dict[str, str]:
        """수치 분석 결과를 프롬프트용으로 변환"""
        
        context = {}
        
        for tool_result in numerical_results:
            func_name = tool_result["function"]
            result = tool_result["result"]
            
            if func_name == "count_recalls":
                context["analysis_type"] = "리콜 건수 통계"
                context["result"] = f"총 {result.get('count', 0):,}건"
                context["description"] = f"필터 조건: {result.get('filters', {})}"
                
            elif func_name == "rank_by_field":
                context["analysis_type"] = f"{result.get('field', '')}별 상위 순위"
                ranking = result.get("ranking", [])
                context["result"] = "\n".join([
                    f"{i+1}위: {item['name']} ({item['count']}건)" 
                    for i, item in enumerate(ranking[:10])
                ])
                context["description"] = f"총 {len(ranking)}개 항목 분석"
                
            elif func_name == "get_monthly_trend":
                context["analysis_type"] = "월별 리콜 트렌드"
                trend = result.get("trend", [])
                context["result"] = "\n".join([
                    f"{item['month']}: {item['count']}건" 
                    for item in trend[:12]
                ])
                context["description"] = f"최근 {len(trend)}개월 데이터"
        
        return context
    
    def _format_logical_for_prompt(self, logical_results: List[Dict]) -> Dict[str, str]:
        """논리 분석 결과를 프롬프트용으로 변환"""
        
        context = {}
        
        for tool_result in logical_results:
            func_name = tool_result["function"]
            result = tool_result["result"]
            
            if func_name == "compare_periods":
                context["operation"] = "기간별 비교 분석"
                period1 = result.get("period1", {})
                period2 = result.get("period2", {})
                change_percent = result.get("change_percent", 0)
                
                context["result"] = {
                    "period1_data": period1,
                    "period2_data": period2, 
                    "change_percent": change_percent
                }
                context["description"] = f"변화율: {change_percent:+.1f}%"
                
            elif func_name == "compare_years":
                context["operation"] = "연도별 상세 비교"
                comparison = result.get("comparison", {})
                context["result"] = comparison
                context["description"] = f"{result.get('year1', '')} vs {result.get('year2', '')} 분석"
                
            elif func_name == "filter_by_conditions":
                context["operation"] = "조건별 필터링"
                cases = result.get("cases", [])
                context["result"] = f"필터링된 사례 {len(cases)}건"
                context["description"] = f"포함조건: {result.get('include_conditions', [])}, 제외조건: {result.get('exclude_conditions', [])}"
                
            elif func_name == "exclude_contaminant_search":
                context["operation"] = "제외 조건 검색"
                cases = result.get("cases", [])
                context["result"] = f"제외 검색 결과 {len(cases)}건"
                context["description"] = f"'{result.get('exclude_contaminant', '')}'를 제외한 '{result.get('include_reason', '')}' 검색"
        
        # JSON을 문자열로 변환
        context["result"] = str(context.get("result", ""))
        context["related_links"] = "FDA 공식 데이터베이스 기반 분석"
        
        return context
    
    def _generate_basic_answer(self, question: str, tool_results: List[Dict]) -> str:
        """폴백용 기본 답변 생성 (기존 방식)"""
        
        answer_parts = []
        answer_parts.append("## 🔍 FDA 리콜 데이터 분석 결과")
        answer_parts.append("")
        
        for tool_result in tool_results:
            func_name = tool_result["function"] 
            result = tool_result["result"]
            
            if "error" in result:
                answer_parts.append(f"⚠️ {func_name} 오류: {result['error']}")
                continue
        
        return "\n".join(answer_parts) if len(answer_parts) > 2 else "관련 정보를 찾을 수 없습니다."

# ======================
# 외부 인터페이스 함수들
# ======================

def create_function_calling_system():
    """Function Calling 시스템 초기화"""
    try:
        return FunctionCallRecallSystem()
    except Exception as e:
        print(f"Function Calling 시스템 초기화 오류: {e}")
        return None

def ask_recall_question_fc(question: str, chat_history: List = None) -> Dict[str, Any]:
    """Function Calling 기반 질문 처리"""
    system = create_function_calling_system()
    if system:
        return system.process_question(question, chat_history)
    else:
        return {
            "answer": "시스템 초기화에 실패했습니다.",
            "function_calls": [],
            "processing_type": "error"
        }

def ask_recall_question(question: str, chat_history: List = None) -> Dict[str, Any]:
    """통합된 리콜 질문 처리 함수 (tab_recall.py 호환용)"""
    
    if chat_history is None:
        chat_history = []
    
    try:
        # Function Calling 시스템 사용
        result = ask_recall_question_fc(question, chat_history)
        
        # 기존 UI 호환 형식으로 반환
        return {
            "answer": result["answer"],
            "recall_documents": [],
            "chat_history": chat_history + [
                HumanMessage(content=question),
                AIMessage(content=result["answer"])
            ],
            "processing_type": result["processing_type"],
            "function_calls": result.get("function_calls", []),
            "has_realtime_data": True,
            "realtime_count": len(result.get("function_calls", []))
        }
        
    except Exception as e:
        return {
            "answer": f"처리 중 오류가 발생했습니다: {e}",
            "recall_documents": [],
            "chat_history": chat_history,
            "processing_type": "error",
            "function_calls": []
        }