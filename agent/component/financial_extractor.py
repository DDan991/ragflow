#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import logging
import re
import pandas as pd
import os
from datetime import datetime
import requests
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
from agent.component.generate import Generate, GenerateParam
from api.db import LLMType
from api.db.services.llm_service import LLMBundle
from rag.prompts import message_fit_in

from ..indicator_config.department_matcher import MinimalDepartmentMatcher


# class SemanticSimilarityChecker:
#     def __init__(self, model_name='DMetaSoul/sbert-chinese-general-v2'):
#         """
#         初始化语义相似度检查器
#         可选模型：
#         - 'DMetaSoul/sbert-chinese-general-v2' (通用中文)
#         - 'shibing624/text2vec-base-chinese' (中文语义)
#         - 'moka-ai/m3e-base' (中文嵌入)
#         """
#         try:
#             self.model = SentenceTransformer(model_name)
#         except Exception as e:
#             logging.warning(f"Could not load semantic model {model_name}: {e}")
#             self.model = None
        
#     def calculate_similarity(self, text1, text2):
#         """计算两个文本的语义相似度"""
#         if self.model is None:
#             # Fallback to simple string similarity if model unavailable
#             return self._simple_similarity(text1, text2)
            
#         try:
#             embeddings = self.model.encode([text1, text2])
#             similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
#             return similarity
#         except Exception as e:
#             logging.warning(f"Semantic similarity calculation failed: {e}")
#             return self._simple_similarity(text1, text2)
    
#     def _simple_similarity(self, text1, text2):
#         """Simple string similarity fallback"""
#         from difflib import SequenceMatcher
#         return SequenceMatcher(None, text1, text2).ratio()
    
#     def are_same_indicator(self, indicators, threshold=0.85):
#         """判断多个指标是否为同一指标"""
#         if len(indicators) < 2:
#             return True
            
#         if self.model is None:
#             return False
            
#         try:
#             embeddings = self.model.encode(indicators)
            
#             # 计算所有指标对的相似度
#             similarities = []
#             for i in range(len(indicators)):
#                 for j in range(i+1, len(indicators)):
#                     sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
#                     similarities.append(sim)
#                     logging.debug(f"'{indicators[i]}' vs '{indicators[j]}': {sim:.3f}")
            
#             # 如果所有相似度都高于阈值，认为是同一指标
#             return all(sim >= threshold for sim in similarities)
#         except Exception as e:
#             logging.warning(f"Indicator comparison failed: {e}")
#             return False


class FinancialExtractorParam(GenerateParam):
    """
    Define the FinancialExtractor component parameters.
    """

    def __init__(self):
        super().__init__()
        # Set default prompt - using {input} placeholder for Canvas integration
        self.prompt = """
你是一个专业的财务信息提取助手。请分析用户的问题，提取其中的时间信息、财务指标和层级信息，并同时提供自然语言回答。

**重要提示：当前系统的查询时间条件仅支持按年度查询。如果您输入的是具体月份（如2025年6月）、具体日期（如2025年6月1日）或季度（如2025年第一季度）等更细粒度的时间信息，系统会自动将其转换为对应的年份（如统一转换为2025年）进行查询。**

请根据今年是2025年，来判断用户问题中的时间信息。

分析用户问题：{input}

请按照以下JSON格式返回结果：
{
  "extracted": {
    "year": ["提取到的所有时间信息，每个时间作为列表中的一个元素（如：2023年、2023-Q1、2023年第一季度等），只返回年份的数字（如：2023，2024）。如果没有时间信息则返回空列表[]。注意：所有具体月份、日期、季度都会被转换为对应年份"],
    "financial_indicator": ["提取到的所有财务指标，每个指标作为列表中的一个元素（如：合同、研发成本、营收账面、营收额外确认、收费财务暂估值、收费待分割、收费 最终考核口径、现供应链票据兑付等）。如果不是财务查询则返回空列表[]"],
    "level": ["提取到的所有层级信息, 每个层级作为列表中的一个元素（如：主业，总承包，总承包勘测设计，总承包工程施工，其他，其他业务收入，投资收益，营业外收入，其他收益) ，如果没有明确层级则为空列表[]）"],
    "department": ["提取到的所有部门信息（如数研院、能研院、科信部、电网公司、川能建等） ，每个部门作为列表中的一个元素 如果没有明确部门则返回空列表[]"]
  },
  "answer": "对用户问题的自然语言回答，解释你识别到的信息并询问缺失的必要参数。如果用户查询了具体月份、日期或季度，需要说明系统会按整年数据返回结果"
}

注意：
1. 时间信息要尽可能标准化，一个查询可能包含多个时间点（如：对比2023年和2024年）
2. 财务指标要使用标准术语，一个查询可能涉及多个指标（如：营收账面和营收额外确认）
3. year和financial_indicator必须返回列表格式，即使为空也要返回[]
4. 如果信息不完整，在answer中要说明需要补充的信息
5. 必须严格按照JSON格式返回，不要添加其他文字, 确保字符串内的单引号不需要转义
6. 如果上一轮的department的列表不为空，且本轮用户输入中又有department，则department只取这一次的部门信息
7. 如果用户查询了月份、季度等细粒度时间，在answer中需要明确告知用户系统将返回对应年份的完整数据

示例：
- 用户输入："2022年和2023年的数研院的营收账面和营收额外确认的总承包数据"
- year应返回：["2022", "2023"]
- financial_indicator应返回：["营收账面", "营收额外确认"]
- department应返回：["数研院"]
- level应返回：["总承包"]

- 用户输入："2025年6月的营收账面数据"
- year应返回：["2025"]
- financial_indicator应返回：["营收账面"]
- department应返回：[]
- level应返回：[]
- answer应包含："您查询的是2025年6月的数据，系统将为您返回2025年全年的营收账面数据"
"""

    def check(self):
        # Call parent check method
        super().check()
        return True


class FinancialExtractor(Generate):
    component_name = "FinancialExtractor"

    def _run(self, history, **kwargs):
        # Use the parent class logic but override the final processing
        chat_mdl = LLMBundle(self._canvas.get_tenant_id(), LLMType.CHAT, self._param.llm_id)

        # Handle prompt parameter substitution like the parent class
        prompt = self._param.prompt
        
        # Process input elements like Generate does
        retrieval_res = []
        self._param.inputs = []
        for para in self.get_input_elements()[1:]:
            if para["key"].lower().find("begin@") == 0:
                cpn_id, key = para["key"].split("@")
                for p in self._canvas.get_component(cpn_id)["obj"]._param.query:
                    if p["key"] == key:
                        kwargs[para["key"]] = p.get("value", "")
                        self._param.inputs.append(
                            {"component_id": para["key"], "content": kwargs[para["key"]]})
                        break
                else:
                    assert False, f"Can't find parameter '{key}' for {cpn_id}"
                continue

            component_id = para["key"]
            cpn = self._canvas.get_component(component_id)["obj"]
            if cpn.component_name.lower() == "answer":
                hist = self._canvas.get_history(1)
                if hist:
                    hist = hist[0]["content"]
                else:
                    hist = ""
                kwargs[para["key"]] = hist
                continue
            _, out = cpn.output(allow_partial=False)
            if "content" not in out.columns:
                kwargs[para["key"]] = ""
            else:
                kwargs[para["key"]] = "  - " + "\n - ".join([o if isinstance(o, str) else str(o) for o in out["content"]])
            self._param.inputs.append({"component_id": para["key"], "content": kwargs[para["key"]]})

        if retrieval_res:
            retrieval_res = pd.concat(retrieval_res, ignore_index=True)
        else:
            retrieval_res = pd.DataFrame([])

        # Substitute parameters in prompt
        for n, v in kwargs.items():
            prompt = re.sub(r"\{%s\}" % re.escape(n), str(v).replace("\\", " "), prompt)

        # Handle {input} placeholder if no other inputs
        if not self._param.inputs and prompt.find("{input}") >= 0:
            retrieval_res = self.get_input()
            input_text = ("  - " + "\n  - ".join(
                [c for c in retrieval_res["content"] if isinstance(c, str)])) if "content" in retrieval_res else ""
            prompt = re.sub(r"\{input\}", input_text.replace("\\", " "), prompt)

        # Get conversation history and fit message
        msg = self._canvas.get_history(self._param.message_history_window_size)
        if len(msg) < 1:
            msg.append({"role": "user", "content": "Output: "})
        _, msg = message_fit_in([{"role": "system", "content": prompt}, *msg], int(chat_mdl.max_length * 0.97))
        if len(msg) < 2:
            msg.append({"role": "user", "content": "Output: "})
        
        # Call LLM
        ans = chat_mdl.chat(msg[0]["content"], msg[1:], self._param.gen_conf())
        ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
        
        # Store component info for debugging
        self._canvas.set_component_infor(self._id, {
            "prompt": msg[0]["content"],
            "messages": msg[1:],
            "conf": self._param.gen_conf()
        })
        
        logging.debug(f"FinancialExtractor raw response: {ans}")
        
        # Format and return output
        return self.format_dual_output(ans)

    def format_dual_output(self, llm_result):
        """
        Format output with structured data and natural language answer.
        """
        try:
            # Try to parse JSON response
            # First, try to extract JSON from the response if it's embedded in text
            json_match = re.search(r'\{.*\}', llm_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                fixed = re.sub(r"\\'", "'", json_str)
                parsed = json.loads(fixed)
            else:
                parsed = json.loads(llm_result.strip())
            
            if "extracted" in parsed and "answer" in parsed:
                extracted_data = parsed["extracted"]

                #The dictionary for queryParam, processed by JAVA backend
                extracted_data["queryParam"] = {}
                extracted_data["queryParam"]["from"] = "营收"
                extracted_data["queryParam"]["select"] = ["year"]
                extracted_data["queryParam"]["where"] = []
                # Check if all fields have values (not empty or null)
                year_list = extracted_data.get("year", [])
                financial_indicator_list = extracted_data.get("financial_indicator", [])
                level_list = extracted_data.get("level", [])
                department_list = extracted_data.get("department", [])
                
                # Set statusCode based on completeness
                if year_list and financial_indicator_list and (level_list or department_list):
                    status_code = 200
                else:
                    status_code = 100

                if year_list:
                    processed_years = self.year_process(year_list)
                    extracted_data["year"] = processed_years
                    if processed_years:
                        extracted_data["queryParam"]["where"].append({"key": "year", "values": processed_years})                    

                if level_list:
                    extracted_data["queryParam"]["where"].append({"key": "deptName", "values": level_list})  


                if financial_indicator_list:
                    category, processed_indicators = self.indicator_process_2(financial_indicator_list)
                    if processed_indicators:
                        extracted_data["queryParam"]["from"] = category
                        for indicator in processed_indicators:
                            extracted_data["queryParam"]["select"].append(indicator)
                    # processed_indicators = self.indicator_process(financial_indicator_list)
                    # extracted_data['financial_indicator'] = processed_indicators
                    # if processed_indicators:
                    #     for indicator in processed_indicators:
                    #         extracted_data["queryParam"]["select"].append(indicator)

                if department_list:
                    department_matcher = MinimalDepartmentMatcher()
                    department_result = []
                    for department in department_list:
                        dept_name, dept_code = department_matcher.match_department(department)
                        department_result.append(dept_code)
                    if department_result:
                        extracted_data['department'] = department_result
                        extracted_data["queryParam"]["where"].append({"key": "deptCode", "values": department_result})  

                
                
                # Add statusCode to extracted_data
                extracted_data["statusCode"] = status_code
                
                # Return formatted result
                return pd.DataFrame([{
                    "content": parsed["answer"],
                    "extracted_data": json.dumps(extracted_data, ensure_ascii=False)
                }])
            else:
                logging.warning("JSON format incorrect, using fallback")
                return self._fallback_output(llm_result)
                
        except (json.JSONDecodeError, AttributeError) as e:
            logging.warning(f"JSON parsing failed: {e}, using fallback")
            return self._fallback_output(llm_result)
    
    def _fallback_output(self, raw_result):
        """
        Fallback for non-JSON responses.
        """
        return pd.DataFrame([{
            "content": raw_result,
            "extracted_data": json.dumps({
                "year": "",
                "financial_indicator": "",
                "level": "",
                "department": "",
                "statusCode": 500  # Internal Error, LLM doesn't return valid JSON
            }, ensure_ascii=False)
        }])

    def debug(self, **kwargs):
        """
        Debug method for testing the component.
        """
        return self._run([], **kwargs)

    def year_process(self, year_list):
        """
        Process year list to convert years to specific dates.
        - If year is not current year: use last day of that year (YYYY-12-31)
        - If year is current year: use today's date (YYYY-MM-DD)
        
        Args:
            year_list: List of year strings (e.g., ["2022", "2023", "2024"])
        
        Returns:
            List of date strings in YYYY-MM-DD format
        """
        if not year_list:
            return []
        
        result_list = []
        current_year = datetime.now().year
        today = datetime.now().strftime("%Y-%m-%d")
        
        for year_str in year_list:
            try:
                # Extract year number from string (handle cases like "2024年")
                year_num = int(year_str.strip().replace("年", ""))
                
                if year_num == current_year:
                    # Current year: use today's date
                    result_list.append(today)
                else:
                    # Not current year: use last day of that year
                    result_list.append(f"{year_num}-12-31")
                    
            except (ValueError, TypeError):
                # If year string is invalid, skip it
                continue
        
        return result_list
    
    def indicator_process(self, indicator_list):
        """
        Process financial indicators against revenue.json with semantic matching fallback.
        
        Args:
            indicator_list: List of indicator strings to process
        
        Returns:
            List of processed indicators with mappings
        """
        if not indicator_list:
            return []
        
        # Load revenue.json
        revenue_data = self.load_revenue_config()
        fuzzy_revenue_data = self.load_fuzzy_revenue_config()

        if not (revenue_data or fuzzy_revenue_data):
            # Fallback to original format if config not found
            return []
        
        result_list = []
        # similarity_checker = SemanticSimilarityChecker()
        
        for indicator in indicator_list:
            # Step 1: Direct match against keys
            if indicator in revenue_data:
                # result_list.append({
                #     "original": indicator,
                #     "mapped_key": indicator,
                #     "mapped_value": revenue_data[indicator],
                #     "match_type": "exact",
                #     "confidence": 1.0
                # })
                result_list.append(revenue_data[indicator])
                continue
            else:
                if indicator in fuzzy_revenue_data:
                    result_list.extend(fuzzy_revenue_data[indicator])
            
            
            # Step 2: Semantic similarity matching against values
            # best_match = None
            # best_score = 0.0
            
            # for key, value in revenue_data.items():
            #     similarity = similarity_checker.calculate_similarity(indicator, value)
            #     if similarity > best_score and similarity >= 0.85:  # 85% threshold
            #         best_score = similarity
            #         best_match = value
            
            # if best_match:
            #     result_list.append(best_match)
            # else:
            #     # No good match found
            #     result_list.append({
            #         "original": indicator,
            #         "mapped_key": "",
            #         "mapped_value": "",
            #         "match_type": "none",
            #         "confidence": 0.0
            #     })
        
        return result_list

    def load_revenue_config(self):
        """Load revenue.json configuration file."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "..", "indicator_config", "revenue.json")
            config_path = os.path.abspath(config_path)
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Could not load revenue.json: {e}")
            return {}
        
    def load_fuzzy_revenue_config(self):
        """Load revenue.json configuration file."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "..", "indicator_config", "fuzzy_revenue.json")
            config_path = os.path.abspath(config_path)
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Could not load fuzzy_revenue.json: {e}")
            return {}
        
    def load_eng_category_config(self):
        """Load eng_category.json configuration file."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "..", "indicator_config", "eng_category.json")
            config_path = os.path.abspath(config_path)
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Could not load eng_category.json: {e}")
            return {}

    def indicator_process_2(self, indicator_list, top_k=5, semantic_weight=0.3, 
                           lexical_weight=0.2, rerank_weight=0.5, 
                           service_url=None, timeout=20):
        """
        Process financial indicators using an external HTTP service with advanced search capabilities.
        Falls back to the original indicator_process method if the service is unavailable.
        
        Args:
            indicator_list: List of indicator strings to process
            top_k: Number of top results to consider for each indicator (default: 5)
            semantic_weight: Weight for semantic similarity (default: 0.3)
            lexical_weight: Weight for lexical matching (default: 0.2)
            rerank_weight: Weight for reranking score (default: 0.5)
            service_url: URL of the indicator search service (default: "http://localhost:5000")
            timeout: Request timeout in seconds (default: 5)
        
        Returns:
            List of English field names corresponding to the input indicators
        """
        if not indicator_list:
            return []
        # Get service URL from environment variable if not provided
        if service_url is None:
            service_url = os.environ.get('FINANCIAL_EXTRACTOR_SERVICE_URL', 'http://localhost:5000')
        
        try:
            # Prepare request payload
            request_data = {
                "queries": indicator_list,
                "top_k": top_k,
                "semantic_weight": semantic_weight,
                "lexical_weight": lexical_weight,
                "rerank_weight": rerank_weight
            }
            
            # Make HTTP request to the service
            logging.info(f"Calling indicator service at {service_url}/search with {len(indicator_list)} indicators")
            response = requests.post(
                f"{service_url}/search",
                json=request_data,
                timeout=timeout
            )

            eng_category_data = self.load_eng_category_config()
            
            # Check if request was successful
            if response.status_code == 200:
                response_data = response.json()
                results = response_data.get("results", [])
                
                # Process results - extract english_name from top result for each query
                processed_indicators = []
                first_category = None

                for i, query_result in enumerate(results[0]):
                    if query_result:
                        # Get the top result (highest final_score)
                        english_name = query_result.get("english_name", "")
                        
                        if english_name:
                            # Get category for this english_name
                            category = eng_category_data.get(english_name, None)
                            
                            # Set first category if not already set
                            if first_category is None and category:
                                first_category = category
                                logging.info(f"First category determined: '{first_category}'")
                            
                            # Only add if category matches the first category or if no category info
                            if category is None or first_category is None or category == first_category:
                                processed_indicators.append(english_name)
                                # logging.debug(f"Indicator '{indicator_list[i]}' mapped to '{english_name}' "
                                #             f"(category: {category}, score: {top_result.get('final_score', 0):.3f}, "
                                #             f"match_type: {top_result.get('match_type', 'unknown')})")
                            else:
                                logging.debug(f"Skipping indicator '{english_name}' "
                                            f"(category '{category}' doesn't match first category '{first_category}')")
                        else:
                            # No valid mapping found for this indicator
                            logging.warning(f"No English name found for indicator ")
                    else:
                        # No results returned for this query
                        logging.warning(f"No results returned for indicator")
                
                logging.info(f"Successfully processed {len(processed_indicators)} indicators "
                           f"(filtered to category: '{first_category}'")
                return first_category, processed_indicators
            
            else:
                # HTTP request failed
                logging.error(f"Indicator service returned status code {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            logging.warning(f"Indicator service timeout after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            logging.warning(f"Could not connect to indicator service at {service_url}")
        # except Exception as e:
        #     logging.error(f"Unexpected error calling indicator service: {e}")
        
        # Fallback to original indicator_process method
        logging.info("Falling back to original indicator_process method")
        return self.indicator_process(indicator_list)

    
