from typing import Dict, Any, TypeVar, Type, Generator
from abc import ABC, abstractmethod
import os
def register_node(cls):
    """Development environment stub decorator"""
    return cls

class Node(ABC):
    """Base node class for development environment"""
    NAME: str = ""
    DESCRIPTION: str = ""
    CATEGORY: str = "Uncategorized"
    INPUTS: Dict[str, Dict[str, Any]] = {}
    OUTPUTS: Dict[str, Dict[str, Any]] = {}
    
    @abstractmethod
    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        raise NotImplementedError
    

    @property
    def is_generator(self) -> bool:
        """Whether this is a generator node"""
        return False

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
