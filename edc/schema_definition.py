from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class SchemaDefiner:
    # The class to handle the first stage: Open Information Extraction
    def __init__(self, model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None, openai_model=None) -> None:
        assert openai_model is not None or (model is not None and tokenizer is not None)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model

    def define_schema(
        self,
        input_text_str: str,
        extracted_triplets_list: List[str],
        few_shot_examples_str: str,
        prompt_template_str: str,
    ) -> List[List[str]]:
        # Given a piece of text and a list of triplets extracted from it, define each of the relation present
       
        relations_present = set()
        for t in extracted_triplets_list:
            relations_present.add(t[1])

        filled_prompt = prompt_template_str.format_map(
            {
                "text": input_text_str,
                "few_shot_examples": few_shot_examples_str,
                "relations": relations_present,
                "triples": extracted_triplets_list,
            }
        )
        logger.info(f"填充后的完整提示前200个字符: {filled_prompt[:200]}...")
        messages = [{"role": "user", "content": filled_prompt}]

        if self.openai_model is None:
            # 对于Qwen模型，我们不使用answer_prepend，因为它可能影响输出格式
            completion = llm_utils.generate_completion_transformers(
                messages, self.model, self.tokenizer, max_new_token=512
            )
            logger.info(f"模型原始输出: {completion}")
        else:
            completion = llm_utils.openai_chat_completion(self.openai_model, None, messages)
            
        # 尝试使用标准解析方法
        relation_definition_dict = llm_utils.parse_relation_definition(completion)
        logger.info(f"标准解析后的关系定义: {relation_definition_dict}")
        
        # 如果标准解析失败，尝试使用更宽松的解析方法
        if not relation_definition_dict:
            logger.info("标准解析失败，尝试使用更宽松的解析方法")
            relation_definition_dict = self._custom_parse_relation_definition(completion, relations_present)
            logger.info(f"自定义解析后的关系定义: {relation_definition_dict}")
        missing_relations = [rel for rel in relations_present if rel not in relation_definition_dict]
        if len(missing_relations) != 0:
            logger.debug(f"Relations {missing_relations} are missing from the relation definition!")
        return relation_definition_dict
    
    def _custom_parse_relation_definition(self, raw_text, relations_present):
        """更宽松的解析方法，适用于Qwen基座模型的输出格式"""
        result = {}
        
        # 尝试多种可能的格式
        # 1. 尝试查找"关系名: 描述"格式
        for relation in relations_present:
            pattern = f"{relation}[：:](.*?)(?=\n\n|\n[^\n]|$)"
            matches = re.findall(pattern, raw_text, re.DOTALL)
            if matches:
                result[relation] = matches[0].strip()
                continue
                
            # 2. 尝试查找"关系名"后面跟着的描述
            pattern = f"{relation}[：:.](.*?)(?=\n\n|\n[^\n]|$)"
            matches = re.findall(pattern, raw_text, re.DOTALL)
            if matches:
                result[relation] = matches[0].strip()
                continue
                
            # 3. 尝试查找描述中包含关系名的段落
            lines = raw_text.split('\n')
            for line in lines:
                if relation in line and (':' in line or '：' in line):
                    parts = re.split('[：:]', line, 1)
                    if len(parts) == 2 and relation in parts[0]:
                        result[relation] = parts[1].strip()
                        break
        
        return result