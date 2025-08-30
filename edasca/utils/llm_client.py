"""
LLM回答获取模块
使用OpenAI Python库获取大模型回答
"""

import json
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: 未安装openai库，请运行: pip install openai")

from ..utils.env_config import Config


@dataclass
class LLMResponse:
    """LLM回答数据结构"""
    content: str
    role: str = "assistant"
    finish_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None
    model: str = "gpt-3.5-turbo"
    created: Optional[int] = None
    response_id: Optional[str] = None
    system_fingerprint: Optional[str] = None


class LLMClient:
    """LLM客户端 - 使用最新OpenAI SDK"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """初始化LLM客户端"""
        if not OPENAI_AVAILABLE:
            raise ImportError("请先安装openai库: pip install openai")
            
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.base_url = base_url or Config.OPENAI_BASE_URL
        self.model = model or Config.OPENAI_MODEL
        
        # 验证配置
        if not self.api_key:
            raise ValueError("未设置OPENAI_API_KEY")
        
        # 初始化同步和异步客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=30.0,
            max_retries=2
        )
        
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=30.0,
            max_retries=2
        )
        
        # 对话历史
        self.conversation_history: List[ChatCompletionMessageParam] = []
    
    def add_system_message(self, content: str):
        """添加系统消息"""
        system_msg: ChatCompletionMessageParam = {
            "role": "system",
            "content": content
        }
        if not self.conversation_history or self.conversation_history[0]["role"] != "system":
            self.conversation_history.insert(0, system_msg)
        else:
            self.conversation_history[0] = system_msg
    
    def add_user_message(self, content: str):
        """添加用户消息"""
        user_msg: ChatCompletionMessageParam = {
            "role": "user",
            "content": content
        }
        self.conversation_history.append(user_msg)
    
    def add_assistant_message(self, content: str):
        """添加助手消息"""
        assistant_msg: ChatCompletionMessageParam = {
            "role": "assistant",
            "content": content
        }
        self.conversation_history.append(assistant_msg)
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
    
    def get_chat_completion(self, 
                          user_input: str, 
                          system_prompt: str = None,
                          temperature: float = 0.7,
                          max_tokens: int = 1000,
                          stream: bool = False,
                          tools: List[Dict] = None,
                          tool_choice: Union[str, Dict] = None) -> Union[LLMResponse, ChatCompletion]:
        """获取聊天完成"""
        
        # 构建消息列表
        messages: List[ChatCompletionMessageParam] = []
        
        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加对话历史
        messages.extend(self.conversation_history)
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        try:
            # 准备参数
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
            
            # 添加工具相关参数（如果有）
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            
            # 调用OpenAI API
            if stream:
                # 返回原始流对象，让调用者处理
                return self.client.chat.completions.stream(**kwargs)
            else:
                response = self.client.chat.completions.create(**kwargs)
                
                # 创建响应对象
                llm_response = LLMResponse(
                    content=response.choices[0].message.content or "",
                    role=response.choices[0].message.role,
                    finish_reason=response.choices[0].finish_reason or "stop",
                    usage=response.usage.model_dump() if response.usage else None,
                    model=response.model,
                    created=response.created,
                    response_id=response.id,
                    system_fingerprint=response.system_fingerprint
                )
                
                # 更新对话历史
                self.add_user_message(user_input)
                self.add_assistant_message(llm_response.content)
                
                return llm_response
                
        except Exception as e:
            print(f"LLM API调用失败: {e}")
            # 返回错误响应
            return LLMResponse(
                content=f"抱歉，我暂时无法回答这个问题。错误信息: {str(e)}",
                role="assistant",
                finish_reason="stop"
            )
    
    def get_simple_response(self, user_input: str, context: str = None) -> str:
        """获取简单回答"""
        system_prompt = "你是一个有帮助的AI助手。请简洁地回答用户的问题。"
        
        if context:
            system_prompt += f"\n\n上下文信息：{context}"
        
        response = self.get_chat_completion(
            user_input=user_input,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=500
        )
        
        return response.content
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """分析文本情感"""
        system_prompt = """你是一个情感分析专家。请分析以下文本的情感，并以JSON格式返回结果。
包含以下字段：
- sentiment: 情感倾向 (positive/negative/neutral)
- confidence: 置信度 (0-1)
- emotions: 情绪列表
- explanation: 分析解释
"""
        
        response = self.get_chat_completion(
            user_input=f"请分析这段文本的情感：{text}",
            system_prompt=system_prompt,
            temperature=0.3
        )
        
        try:
            # 尝试解析JSON
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            # 如果不是JSON格式，返回简单分析
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "emotions": [],
                "explanation": "无法解析情感分析结果"
            }
    
    def generate_response_with_context(self, 
                                     user_input: str, 
                                     context_info: Dict[str, Any]) -> str:
        """基于上下文生成回答"""
        
        # 构建上下文提示
        context_prompt = f"""你是一个智能助手。请基于以下上下文信息回答用户的问题。

上下文信息：
{json.dumps(context_info, ensure_ascii=False, indent=2)}

请记住：
1. 基于提供的上下文信息回答
2. 如果上下文中没有相关信息，请诚实说明
3. 保持回答简洁明了
4. 如果是事实性问题，确保准确性
"""
        
        response = self.get_chat_completion(
            user_input=user_input,
            system_prompt=context_prompt,
            temperature=0.5,
            max_tokens=800
        )
        
        return response.content
    
    def creative_writing(self, prompt: str, style: str = None) -> str:
        """创意写作"""
        system_prompt = "你是一个创意写作专家。请根据用户的提示进行创意写作。"
        
        if style:
            system_prompt += f"\n写作风格：{style}"
        
        response = self.get_chat_completion(
            user_input=prompt,
            system_prompt=system_prompt,
            temperature=0.8,  # 更高的温度以增加创意性
            max_tokens=1000
        )
        
        return response.content
    
    def code_generation(self, requirements: str, language: str = "Python") -> str:
        """代码生成"""
        system_prompt = f"""你是一个专业的程序员。请根据用户的要求生成{language}代码。

要求：
1. 代码要简洁、高效、可读性强
2. 添加必要的注释
3. 确保代码可以正常运行
4. 如果有多个文件，请说明文件结构
"""
        
        response = self.get_chat_completion(
            user_input=requirements,
            system_prompt=system_prompt,
            temperature=0.3,  # 较低的温度以保持准确性
            max_tokens=1500
        )
        
        return response.content
    
    def translate_text(self, text: str, target_language: str) -> str:
        """文本翻译"""
        system_prompt = f"""你是一个专业的翻译专家。请将以下文本翻译成{target_language}。

要求：
1. 保持原文的意思和语气
2. 确保翻译自然流畅
3. 如果是技术术语，请使用标准的翻译
4. 如果有文化特定的内容，请适当调整
"""
        
        response = self.get_chat_completion(
            user_input=f"请将以下文本翻译成{target_language}：{text}",
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=800
        )
        
        return response.content
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """文本摘要"""
        system_prompt = f"""你是一个文本摘要专家。请将以下文本总结成不超过{max_length}字的摘要。

要求：
1. 保留原文的主要信息
2. 摘要要连贯流畅
3. 突出重点内容
4. 避免添加原文中没有的信息
"""
        
        response = self.get_chat_completion(
            user_input=f"请总结以下文本：{text}",
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=max_length
        )
        
        return response.content
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            # 获取模型列表
            models = self.client.models.list()
            
            # 获取当前模型信息
            model_info = self.client.models.retrieve(self.model)
            
            return {
                "current_model": self.model,
                "model_info": {
                    "id": model_info.id,
                    "created": model_info.created,
                    "owned_by": model_info.owned_by,
                    "permissions": model_info.permissions,
                    "root": model_info.root,
                    "parent": model_info.parent
                },
                "available_models": len(models.data),
                "api_endpoint": self.base_url,
                "client_type": "openai-python-v1"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "current_model": self.model,
                "client_type": "openai-python-v1"
            }
    
    async def get_chat_completion_async(self, 
                                     user_input: str, 
                                     system_prompt: str = None,
                                     temperature: float = 0.7,
                                     max_tokens: int = 1000) -> LLMResponse:
        """异步获取聊天完成"""
        
        # 构建消息列表
        messages: List[ChatCompletionMessageParam] = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            llm_response = LLMResponse(
                content=response.choices[0].message.content or "",
                role=response.choices[0].message.role,
                finish_reason=response.choices[0].finish_reason or "stop",
                usage=response.usage.model_dump() if response.usage else None,
                model=response.model,
                created=response.created,
                response_id=response.id,
                system_fingerprint=response.system_fingerprint
            )
            
            return llm_response
            
        except Exception as e:
            return LLMResponse(
                content=f"抱歉，我暂时无法回答这个问题。错误信息: {str(e)}",
                role="assistant",
                finish_reason="stop"
            )
    
    def create_structured_response(self, 
                                user_input: str,
                                response_format: Any,
                                system_prompt: str = None) -> Any:
        """创建结构化响应（使用Pydantic模型）"""
        
        # 构建消息列表
        messages: List[ChatCompletionMessageParam] = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})
        
        try:
            # 使用parse方法获取结构化响应
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format
            )
            
            # 更新对话历史
            if response.choices[0].message.content:
                self.add_user_message(user_input)
                self.add_assistant_message(response.choices[0].message.content)
            
            return response
            
        except Exception as e:
            print(f"结构化响应生成失败: {e}")
            return None


# 全局LLM客户端实例
_global_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取全局LLM客户端实例"""
    global _global_llm_client
    
    if _global_llm_client is None:
        _global_llm_client = LLMClient()
    
    return _global_llm_client


def get_llm_response(user_input: str, 
                     system_prompt: str = None,
                     context: Dict[str, Any] = None) -> str:
    """获取LLM回答的便捷函数"""
    client = get_llm_client()
    
    if context:
        return client.generate_response_with_context(user_input, context)
    elif system_prompt:
        response = client.get_chat_completion(
            user_input=user_input,
            system_prompt=system_prompt
        )
        return response.content
    else:
        return client.get_simple_response(user_input)


# 测试函数
def test_llm_client():
    """测试LLM客户端"""
    print("=== LLM客户端测试 ===")
    
    try:
        # 创建客户端
        client = get_llm_client()
        
        # 获取模型信息
        model_info = client.get_model_info()
        print(f"当前模型: {model_info.get('current_model', 'Unknown')}")
        print(f"客户端类型: {model_info.get('client_type', 'Unknown')}")
        
        # 简单对话测试
        print("\n1. 简单对话测试:")
        response = client.get_simple_response("你好！请介绍一下你自己。")
        print(f"LLM: {response}")
        
        # 情感分析测试
        print("\n2. 情感分析测试:")
        sentiment = client.analyze_sentiment("今天天气真好，我很开心！")
        print(f"情感分析: {sentiment}")
        
        # 测试异步功能
        print("\n3. 异步功能测试:")
        import asyncio
        
        async def async_test():
            async_response = await client.get_chat_completion_async(
                "请用一句话描述异步编程的优势"
            )
            print(f"异步响应: {async_response.content}")
        
        asyncio.run(async_test())
        
        # 测试结构化输出
        print("\n4. 结构化输出测试:")
        from pydantic import BaseModel, Field
        from typing import List
        
        class Recipe(BaseModel):
            name: str = Field(..., description="菜名")
            ingredients: List[str] = Field(..., description="食材列表")
            steps: List[str] = Field(..., description="制作步骤")
            cooking_time: int = Field(..., description="烹饪时间（分钟）")
        
        structured_response = client.create_structured_response(
            "请提供一个简单的番茄炒蛋食谱",
            response_format=Recipe,
            system_prompt="你是一个烹饪专家，请提供详细的食谱信息。"
        )
        
        if structured_response and hasattr(structured_response.choices[0].message, 'parsed'):
            recipe = structured_response.choices[0].message.parsed
            print(f"菜名: {recipe.name}")
            print(f"食材: {', '.join(recipe.ingredients)}")
            print(f"烹饪时间: {recipe.cooking_time}分钟")
        else:
            print("结构化输出测试失败或不受支持")
        
        print("\n✅ LLM客户端测试通过！")
        
    except Exception as e:
        print(f"❌ LLM客户端测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_llm_client()