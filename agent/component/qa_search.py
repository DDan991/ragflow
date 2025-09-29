import json
import logging
import os
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

from agent.component.generate import Generate, GenerateParam
from api.db import LLMType
from api.db.services.llm_service import LLMBundle


# ===========================
# 参数配置类
# ===========================
class QASearchParam(GenerateParam):
    """
    QA搜索组件的参数配置
    """
    
    def __init__(self):
        super().__init__()
        
        # 搜索服务配置
        self.search_service_url = os.environ.get('QA_SEARCH_SERVICE_URL', 'http://localhost:5001')
        self.search_top_k = 3  # 搜索返回的结果数量
        self.use_rerank = True  # 是否使用重排序
        self.similarity_threshold = 0.7  # 相似度阈值
        self.search_timeout = 30  # 搜索超时时间（秒）
        
        # 多轮对话配置
        self.max_concat_rounds = 5  # 最大拼接轮数
        self.match_threshold = 0.8  # 匹配阈值，高于此值认为是好的匹配
        
        # 默认提示词模板
        self.prompt = """
你是一个专业的问答助手。请根据用户的问题进行搜索和回答。

用户问题：{input}

请分析用户的问题，如果找到了匹配的问题和答案，请直接回答。如果没有找到匹配的问题，请提示用户重新描述问题。

搜索结果：{search_results}

请根据搜索结果为用户提供准确的回答。如果搜索结果中没有相关信息，请礼貌地告知用户。
"""

    def check(self):
        # Don't call super().check() to avoid GenerateParam's llm_id validation
        # QASearch doesn't need an LLM since it calls an external service
        
        # Validate QASearch-specific parameters
        self.check_positive_integer(self.search_top_k, "[QASearch] Search top K")
        self.check_decimal_float(self.similarity_threshold, "[QASearch] Similarity threshold")
        self.check_positive_integer(self.search_timeout, "[QASearch] Search timeout")
        self.check_positive_integer(self.max_concat_rounds, "[QASearch] Max concat rounds")
        self.check_decimal_float(self.match_threshold, "[QASearch] Match threshold")
        self.check_empty(self.search_service_url, "[QASearch] Search service URL")
        
        return True


# ===========================
# 主要组件类
# ===========================
class QASearch(Generate):
    """
    QA搜索组件
    
    功能：
    1. 调用外部搜索服务API
    2. 获取用户对话上下文
    3. 直接返回服务响应
    """
    
    component_name = "QASearch"
    
    
    # ===========================
    # 主要执行逻辑
    # ===========================
    def _run(self, history, **kwargs):
        """
        主要执行方法
        
        处理流程：
        1. 获取用户输入和对话历史
        2. 调用搜索服务
        3. 直接返回搜索结果
        """
        # 获取用户输入和对话历史
        user_input = self._get_user_input(kwargs)
        if not user_input:
            # 返回空响应，让switch.py能正确检测为empty
            return Generate.be_output("")
        
        logging.info(f"QASearch received input: {user_input}")
        
        # 调用搜索服务
        response = self._perform_search(user_input)
        
        # 直接返回结果，包括空响应
        return Generate.be_output(response)
    
    
    # ===========================
    # 输入处理相关方法
    # ===========================
    def _get_user_input(self, kwargs) -> str:
        """
        从组件参数中提取用户输入，并添加对话历史上下文
        """
        self._param.inputs = []
        
        # 处理来自其他组件的输入
        for para in self.get_input_elements()[1:]:
            if para["key"].lower().find("begin@") == 0:
                cpn_id, key = para["key"].split("@")
                for p in self._canvas.get_component(cpn_id)["obj"]._param.query:
                    if p["key"] == key:
                        kwargs[para["key"]] = p.get("value", "")
                        self._param.inputs.append({
                            "component_id": para["key"], 
                            "content": kwargs[para["key"]]
                        })
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
                kwargs[para["key"]] = "  - " + "\n - ".join([
                    o if isinstance(o, str) else str(o) for o in out["content"]
                ])
            self._param.inputs.append({
                "component_id": para["key"], 
                "content": kwargs[para["key"]]
            })
        
        # 获取当前用户输入
        current_input = ""
        if not self._param.inputs:
            retrieval_res = self.get_input()
            if "content" in retrieval_res:
                current_input = "\n".join([
                    c for c in retrieval_res["content"] if isinstance(c, str)
                ])
        else:
            current_input = self._param.inputs[0]["content"]
        
        # 直接返回当前用户输入，不使用对话历史
        return current_input.strip()
    
    
    # ===========================
    # 搜索相关方法
    # ===========================
    def _perform_search(self, query: str) -> str:
        """
        调用外部搜索服务API，直接返回响应内容
        """
        try:
            search_payload = {
                "query": query,
                "top_k": self._param.search_top_k,
                "use_rerank": self._param.use_rerank,
                "similarity_threshold": self._param.similarity_threshold
            }
            
            response = requests.post(
                f"{self._param.search_service_url}/search",
                json=search_payload,
                timeout=self._param.search_timeout
            )
            
            if response.status_code == 200:
                # 处理空字符串响应（没有高分结果）
                if response.text == "":
                    logging.info("Search service returned empty response (no high-score results)")
                    return ""
                
                # 解析JSON响应
                response_data = response.json()
                
                # 从results数组中提取答案
                results = response_data.get("results", [])
                if results and len(results) > 0:
                    # 返回第一个（最佳）结果的答案
                    answer = results[0].get("answer", "")
                    logging.info(f"Search service returned answer: {answer}")
                    return answer
                else:
                    logging.info("Search service returned JSON but no results")
                    return ""
            else:
                logging.error(f"Search service returned status {response.status_code}: {response.text}")
                return ""
                
        except requests.exceptions.Timeout:
            logging.error(f"Search service timeout after {self._param.search_timeout} seconds")
            return ""
        except requests.exceptions.ConnectionError:
            logging.error(f"Could not connect to search service at {self._param.search_service_url}")
            return ""
        except Exception as e:
            logging.error(f"Unexpected error during search: {e}")
            return ""
    
    
    
    
    # ===========================
    # 调试方法
    # ===========================
    def debug(self, **kwargs):
        """
        调试方法，用于测试组件
        """
        return self._run([], **kwargs)