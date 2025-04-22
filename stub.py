from typing import Dict, Any, TypeVar, Type, Generator
from abc import ABC, abstractmethod
import os
def register_node(cls):
    """Development environment stub decorator"""
    return cls

class Node:
    """节点基类"""
    NAME = "节点1"
    DESCRIPTION = ""
    CATEGORY = ""
    INPUTS = {}
    OUTPUTS = {}


    # async def run_agent_async(self, agent_id: str, input_text: str) -> str:
    #     """异步运行Agent"""
    #     agent_executor = AgentExecutor(AgentManager())
    #     result = await agent_executor.run_once(agent_id=agent_id, text=input_text)
    #     return result.get("result", "") if result else ""
    
    

    async def run_agent(self, agent_id: str, input_text: str) -> str:
        """
        运行Agent - 使用工作流管理器执行
        
        Args:
            agent_id: Agent ID
            input_text: 输入文本
            
        Returns:
            str: Agent执行结果
        """
        pass

    
    @abstractmethod
    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        raise NotImplementedError("子类必须execute方法")
    
    async def stop(self) -> None:
        """
        Stop the node execution when interrupted.
        This is an optional method that nodes can implement to handle interruption.
        The default implementation does nothing.
        """
        pass

    @property
    def is_generator(self) -> bool:
        """是否是生成器节点"""
        return False
    
    def format_input(self, raw_input: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """格式化输入参数
        
        Args:
            raw_input: 原始输入
            config: 工具配置
            
        Returns:
            格式化后的输入参数
        """
        formatted_input = {}
        
        # 1. 首先添加配置参数
        formatted_input.update(config)
        
        # 2. 处理原始输入
        if isinstance(raw_input, str):
            # 查找第一个必需的字符串类型参数
            for param_name, param_info in self.INPUTS.items():
                if (param_info.get('required', False) and 
                    param_info.get('type', '').upper() == 'STRING'):
                    formatted_input[param_name] = raw_input
                    break
        elif isinstance(raw_input, dict):
            formatted_input.update(raw_input)
            
        # 3. 检查必需参数
        missing_params = []
        for param_name, param_info in self.INPUTS.items():
            if param_info.get('required', False) and param_name not in formatted_input:
                missing_params.append(param_name)
            
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
            
        return formatted_input

class GeneratorNode(Node):
    """Generator node base class for development environment"""
    
    @property
    def is_generator(self) -> bool:
        """Override parent's is_generator property"""
        return True
    
    @abstractmethod
    def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Generator:
        """
        Execute the generator node
        
        Args:
            node_inputs: Input parameters dictionary
            workflow_logger: Logger instance for workflow execution
            
        Returns:
            Generator that yields results
        """
        raise NotImplementedError

class ConditionalNode(Node):
    """Conditional branch node base class for development environment"""
    
    @property
    def is_conditional(self) -> bool:
        """Whether this is a conditional branch node"""
        return True

    @abstractmethod
    def get_active_branch(self, outputs: Dict[str, Any]) -> str:
        """
        Get the name of currently active branch
        
        Args:
            outputs: Node execution outputs
            
        Returns:
            str: Name of the active output port
        """
        raise NotImplementedError

def get_api_key(provider: str, key_name: str) -> str:
    """Get API key from environment variables"""
    return os.getenv(key_name)
