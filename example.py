try:
    from autotask.nodes import Node, GeneratorNode, ConditionalNode, register_node
    from autotask.api_keys import get_api_key
except ImportError:
    from stub import Node, GeneratorNode, ConditionalNode, register_node, get_api_key

from typing import Dict, Any, Generator, AsyncGenerator


# If an API KEY is needed, please use get_api_key to obtain it in the global environment.
API_KEY = get_api_key(provider="eva.ai", key_name="API_KEY")


@register_node
class ExampleNode(Node):
    NAME = "Example Node"
    DESCRIPTION = "This is an example node"

    INPUTS = {
        "input_param": {
            "label": "Input Parameter",
            "description": "Parameter Description",
            "type": "STRING", # 可用的type: STRING, INT, FLOAT, IMAGEUPLOAD, BOOLEAN, ASSISTANT，""(如果是空串 表示可以接收任何类型的输入参数)
            "required": True,
            # 可选字段widget: FILE | DIR （表示前端展示为文件上传框或者目录选择框）
        }
    }

    OUTPUTS = {
        "result": {
            "label": "Processing Result",
            "description": "Output Description",
            "type": "STRING",
        }
    }

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            param = node_inputs["input_param"]
            workflow_logger.info("Start processing...")
            return {"success": True, "result": param}

        except Exception as e:
            workflow_logger.error(f"Processing failed: {str(e)}")
            return {"success": False, "error_message": str(e)}


@register_node
class ExampleGeneratorNode(GeneratorNode):
    NAME = "Number Sequence Generator"
    DESCRIPTION = "Generate a sequence of numbers within a specified range"

    INPUTS = {
        "start": {
            "label": "Start Number",
            "description": "The starting number of the sequence",
            "type": "INT",
            "default": 0,
            "required": True,
        },
        "end": {
            "label": "End Number",
            "description": "The ending number of the sequence",
            "type": "INT",
            "default": 10,
            "required": True,
        },
        "step": {
            "label": "Step Size",
            "description": "The increment between numbers",
            "type": "INT",
            "default": 1,
            "required": True,
        },
    }

    OUTPUTS = {
        "number": {
            "label": "Generated Number",
            "description": "Current number in the sequence",
            "type": "INT",
        }
    }

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> AsyncGenerator[Any, None]:
        try:
            start = node_inputs.get("start", 0)
            end = node_inputs.get("end", 10)
            step = node_inputs.get("step", 1)

            workflow_logger.info(
                f"Starting number sequence generation: {start} to {end} with step {step}"
            )

            for number in range(start, end + 1, step):
                workflow_logger.debug(f"Generating number: {number}")
                yield {
                    "number": number
                }

            workflow_logger.info("Sequence generation completed successfully")

        except Exception as e:
            workflow_logger.error(f"Sequence generation failed: {str(e)}")
            return


@register_node
class ExampleConditionNode(ConditionalNode):
    """Conditional node to determine if a number is even"""
    NAME = "Even Number Check"
    DESCRIPTION = "Check if the input number is even and execute different branches accordingly"
    
    INPUTS = {
        "number": {
            "label": "Input Number",
            "description": "Number to check",
            "type": "INT",
            "required": True
        }
    }
    
    OUTPUTS = {
        "true_branch": {
            "label": "Even Branch",
            "description": "Output when number is even",
            "type": ""
        },
        "false_branch": {
            "label": "Odd Branch", 
            "description": "Output when number is odd",
            "type": ""
        }
    }

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            number = node_inputs.get("number")
            workflow_logger.debug(f"Checking number: {number}")
            
            # Check if number is even
            is_even = number % 2 == 0
            workflow_logger.debug(f"Is even: {is_even}")

            return {
                "condition_result": is_even
            }

        except Exception as e:
            error_msg = f"Even number check failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "condition_result": None
            }

    def get_active_branch(self, outputs: Dict[str, Any]) -> str:
        """Returns the name of the active branch"""
        return "true_branch" if outputs.get("condition_result") else "false_branch"


if __name__ == "__main__":
    # Setup basic logging
    import logging
    import asyncio
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test ExampleNode
    print("\n1. Testing ExampleNode:")
    node1 = ExampleNode()
    result1 = asyncio.run(node1.execute({"input_param": "test"}, logger))
    print(f"Result: {result1}")
    
    # Test ExampleGeneratorNode
    print("\n2. Testing ExampleGeneratorNode:")
    node2 = ExampleGeneratorNode()
    async def test_generator():
        async for num in node2.execute({"start": 0, "end": 4, "step": 1}, logger):
            print(f"Generated: {num}")
    asyncio.run(test_generator())
    
    # Test EvenNumberConditionNode
    print("\n3. Testing EvenNumberConditionNode:")
    node3 = ExampleConditionNode()
    even_result = asyncio.run(node3.execute({"number": 2}, logger))
    odd_result = asyncio.run(node3.execute({"number": 3}, logger))
    print(f"Even number branch: {node3.get_active_branch(even_result)}")
    print(f"Odd number branch: {node3.get_active_branch(odd_result)}")


