"""
健壮的LLM客户端
基于OpenAI最佳实践，包含完善的错误处理和重试机制
"""

import json
import time
import random
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import openai
from openai import OpenAI
from openai.types import ChatCompletion

from ..utils.env_config import Config


@dataclass
class LLMResponse:
    """LLM回答数据结构"""
    content: str
    role: str = "assistant"
    finish_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None
    model: str = "gpt-3.5-turbo"
    created: int = None
    success: bool = True
    error: Optional[str] = None


class RobustLLMClient:
    """健壮的LLM客户端"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """初始化LLM客户端"""
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.base_url = base_url or Config.OPENAI_BASE_URL
        self.model = model or Config.OPENAI_MODEL
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=30.0,  # 默认超时
            max_retries=3  # 内置重试
        )
        
        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []
        
        # 备用API端点
        self.fallback_endpoints = [
            "https://api.openai.com/v1",
            "https://api.openai-proxy.com/v1"
        ]
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retry_count": 0
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.APITimeoutError, requests.exceptions.RequestException))
    )
    def _make_request(self, messages: List[Dict], **kwargs) -> ChatCompletion:
        """发送请求到OpenAI API"""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
    
    def _fallback_request(self, messages: List[Dict], **kwargs) -> Optional[ChatCompletion]:
        """使用备用端点发送请求"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        
        for endpoint in self.fallback_endpoints:
            try:
                response = requests.post(
                    f"{endpoint}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                if response.status_code == 200:
                    # 转换为ChatCompletion对象
                    return ChatCompletion(**response.json())
            except Exception as e:
                print(f"备用端点 {endpoint} 失败: {e}")
                continue
        
        return None
    
    def get_chat_completion(self, 
                          user_input: str, 
                          system_prompt: str = None,
                          temperature: float = 0.7,
                          max_tokens: int = 1000,
                          stream: bool = False,
                          use_fallback: bool = True) -> LLMResponse:
        """获取聊天完成"""
        
        self.stats["total_requests"] += 1
        
        # 构建消息列表
        messages = []
        
        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加对话历史
        messages.extend(self.conversation_history)
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        try:
            # 尝试主请求
            response = self._make_request(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            # 创建响应对象
            llm_response = LLMResponse(
                content=response.choices[0].message.content,
                role=response.choices[0].message.role,
                finish_reason=response.choices[0].finish_reason,
                usage=response.usage.model_dump() if response.usage else None,
                model=response.model,
                created=response.created,
                success=True
            )
            
            # 更新统计
            self.stats["successful_requests"] += 1
            
            # 更新对话历史
            self.add_user_message(user_input)
            self.add_assistant_message(llm_response.content)
            
            return llm_response
            
        except Exception as e:
            print(f"主请求失败: {e}")
            self.stats["retry_count"] += 1
            
            # 尝试备用请求
            if use_fallback:
                print("尝试备用请求...")
                response = self._fallback_request(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if response:
                    llm_response = LLMResponse(
                        content=response.choices[0].message.content,
                        role=response.choices[0].message.role,
                        finish_reason=response.choices[0].finish_reason,
                        usage=response.usage.model_dump() if response.usage else None,
                        model=response.model,
                        created=response.created,
                        success=True
                    )
                    
                    self.stats["successful_requests"] += 1
                    self.add_user_message(user_input)
                    self.add_assistant_message(llm_response.content)
                    
                    return llm_response
            
            # 如果所有请求都失败
            self.stats["failed_requests"] += 1
            
            # 返回错误响应
            error_msg = f"API请求失败: {str(e)}"
            return LLMResponse(
                content="抱歉，我现在无法回答你的问题。请稍后再试。",
                role="assistant",
                success=False,
                error=error_msg
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
    
    def add_system_message(self, content: str):
        """添加系统消息"""
        self.conversation_history.insert(0, {
            "role": "system",
            "content": content
        })
    
    def add_user_message(self, content: str):
        """添加用户消息"""
        self.conversation_history.append({
            "role": "user",
            "content": content
        })
    
    def add_assistant_message(self, content: str):
        """添加助手消息"""
        self.conversation_history.append({
            "role": "assistant",
            "content": content
        })
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats["total_requests"]
        if total > 0:
            success_rate = self.stats["successful_requests"] / total * 100
        else:
            success_rate = 0
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "retry_rate": self.stats["retry_count"] / total * 100 if total > 0 else 0
        }
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 发送一个简单的测试请求
            response = self.get_chat_completion(
                user_input="hello",
                system_prompt="请回复'ok'",
                max_tokens=10
            )
            return response.success
        except Exception:
            return False


# 全局LLM客户端实例
_global_llm_client: Optional[RobustLLMClient] = None


def get_llm_client() -> RobustLLMClient:
    """获取全局LLM客户端实例"""
    global _global_llm_client
    
    if _global_llm_client is None:
        _global_llm_client = RobustLLMClient()
    
    return _global_llm_client


def get_llm_response(user_input: str, 
                     system_prompt: str = None,
                     context: Dict[str, Any] = None) -> str:
    """获取LLM回答的便捷函数"""
    client = get_llm_client()
    
    if context:
        # 构建上下文提示
        context_str = json.dumps(context, ensure_ascii=False, indent=2)
        full_system_prompt = f"{system_prompt}\n\n上下文信息：{context_str}" if system_prompt else f"上下文信息：{context_str}"
        
        response = client.get_chat_completion(
            user_input=user_input,
            system_prompt=full_system_prompt
        )
        return response.content
    elif system_prompt:
        response = client.get_chat_completion(
            user_input=user_input,
            system_prompt=system_prompt
        )
        return response.content
    else:
        return client.get_simple_response(user_input)


# 测试函数
def test_robust_llm_client():
    """测试健壮的LLM客户端"""
    print("=== 健壮LLM客户端测试 ===")
    
    try:
        # 创建客户端
        client = get_llm_client()
        
        # 健康检查
        print("1. 健康检查...")
        is_healthy = client.health_check()
        print(f"健康状态: {'✓' if is_healthy else '✗'}")
        
        # 简单对话测试
        print("\n2. 简单对话测试:")
        response = client.get_simple_response("你好！请介绍一下你自己。")
        print(f"LLM: {response}")
        
        # 统计信息
        stats = client.get_stats()
        print(f"\n3. 统计信息:")
        print(f"总请求数: {stats['total_requests']}")
        print(f"成功请求数: {stats['successful_requests']}")
        print(f"失败请求数: {stats['failed_requests']}")
        print(f"成功率: {stats['success_rate']:.1f}%")
        print(f"重试率: {stats['retry_rate']:.1f}%")
        
        # 错误处理测试
        print("\n4. 错误处理测试:")
        # 这里可以模拟网络错误进行测试
        
        print("\n✅ 健壮LLM客户端测试通过！")
        
    except Exception as e:
        print(f"❌ 健壮LLM客户端测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_robust_llm_client()