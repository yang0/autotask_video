try:
    from autotask.nodes import Node, register_node
except ImportError:
    from stub import Node, register_node

import os
import subprocess
from typing import Dict, Any
from datetime import datetime


@register_node
class AudioToWavNode(Node):
    NAME = "Audio to WAV Converter"
    DESCRIPTION = "Convert various audio formats to WAV format with customizable settings"

    INPUTS = {
        "input_file": {
            "label": "Input Audio File",
            "description": "Path to the input audio file",
            "type": "STRING",
            "required": True,
            "widget": "FILE"
        },
        "output_dir": {
            "label": "Output Directory",
            "description": "Directory to save the converted WAV file",
            "type": "STRING",
            "required": True,
            "widget": "DIR"
        },
        "sample_rate": {
            "label": "Sample Rate",
            "description": "Audio sample rate in Hz",
            "type": "INT",
            "required": False,
            "default": 16000
        },
        "channels": {
            "label": "Channels",
            "description": "Number of audio channels (1 for mono, 2 for stereo)",
            "type": "INT",
            "required": False,
            "default": 1
        },
        "bit_depth": {
            "label": "Bit Depth",
            "description": "Audio bit depth (16 or 24)",
            "type": "INT",
            "required": False,
            "default": 16
        },
        "overwrite": {
            "label": "Overwrite Existing",
            "description": "Overwrite output file if it already exists",
            "type": "STRING",
            "required": False,
            "default": "true"
        }
    }

    OUTPUTS = {
        "output_file": {
            "label": "Output WAV File",
            "description": "Path to the converted WAV file",
            "type": "STRING"
        }
    }

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            input_file = node_inputs["input_file"]
            output_dir = node_inputs["output_dir"]
            sample_rate = node_inputs.get("sample_rate", 16000)
            channels = node_inputs.get("channels", 1)
            bit_depth = node_inputs.get("bit_depth", 16)
            overwrite = node_inputs.get("overwrite", "true").lower() == "true"

            workflow_logger.info(f"Starting audio conversion for: {input_file}")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Prepare output filename with datetime
            input_filename = os.path.basename(input_file)
            name_without_ext = os.path.splitext(input_filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{name_without_ext}_{timestamp}.wav"
            output_file = os.path.join(output_dir, output_filename)

            # Check if output file exists
            if os.path.exists(output_file) and not overwrite:
                error_msg = f"Output file already exists and overwrite is disabled: {output_file}"
                workflow_logger.error(error_msg)
                return {"success": False, "error_message": error_msg}

            # Build ffmpeg command
            cmd = [
                "ffmpeg",
                "-i", input_file,
                "-ac", str(channels),
                "-ar", str(sample_rate),
                "-sample_fmt", f"s{bit_depth}",
            ]

            if overwrite:
                cmd.append("-y")
            else:
                cmd.append("-n")

            cmd.append(output_file)
            
            workflow_logger.debug(f"Running command: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if process.returncode != 0:
                error_msg = f"Audio conversion failed: {process.stderr}"
                workflow_logger.error(error_msg)
                return {"success": False, "error_message": error_msg}

            # Verify the output file exists and has content
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                error_msg = "Converted audio file is empty or does not exist"
                workflow_logger.error(error_msg)
                return {"success": False, "error_message": error_msg}

            workflow_logger.info(f"Audio converted successfully to: {output_file}")
            return {
                "success": True,
                "output_file": output_file
            }

        except Exception as e:
            error_msg = f"Audio conversion failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {"success": False, "error_message": error_msg}


if __name__ == "__main__":
    # Setup basic logging
    import logging
    import asyncio
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test AudioToWavNode
    print("\nTesting AudioToWavNode:")
    node = AudioToWavNode()
    test_inputs = {
        "input_file": "test.mp3",
        "output_dir": "output",
        "sample_rate": 44100,
        "channels": 2,
        "bit_depth": 24,
        "overwrite": "true"
    }
    result = asyncio.run(node.execute(test_inputs, logger))
    print(f"Result: {result}")
