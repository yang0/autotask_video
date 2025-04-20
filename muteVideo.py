try:
    from autotask.nodes import Node, register_node
except ImportError:
    from stub import Node, register_node

from typing import Dict, Any
import os
import subprocess


@register_node
class VideoMutingNode(Node):
    NAME = "Video Muting"
    DESCRIPTION = "Remove audio from a video file"

    INPUTS = {
        "video_path": {
            "label": "Video File",
            "description": "Path to the video file to be muted",
            "type": "STRING",
            "required": True,
            "widget": "FILE"
        }
    }

    OUTPUTS = {
        "output_path": {
            "label": "Output Video Path",
            "description": "Path to the muted video file",
            "type": "STRING"
        }
    }

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            # Extract parameters from inputs
            video_path = node_inputs["video_path"]
            
            workflow_logger.info(f"Starting video muting for: {video_path}")
            
            # Determine output directory and filename
            video_name = os.path.basename(video_path)
            video_name_without_ext, video_ext = os.path.splitext(video_name)
            muted_video_name = f"mute_{video_name_without_ext}{video_ext}"
            
            # Always save in the same directory as the input video
            output_path = os.path.join(os.path.dirname(video_path), muted_video_name)
            
            # Use ffmpeg to mute the video (remove audio)
            workflow_logger.info("Removing audio from video...")
            ffmpeg_cmd = [
                "ffmpeg", 
                "-i", video_path, 
                "-c:v", "copy", 
                "-an", 
                "-y", 
                output_path
            ]
            
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                error_msg = f"FFmpeg process failed: {stderr}"
                workflow_logger.error(error_msg)
                return {
                    "success": False,
                    "error_message": error_msg
                }
            
            workflow_logger.info(f"Video muting completed successfully. Output file: {output_path}")
            
            return {
                "success": True,
                "output_path": output_path
            }
            
        except Exception as e:
            error_msg = f"Video muting failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg
            }


if __name__ == "__main__":
    # Setup basic logging
    import logging
    import asyncio
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test VideoMutingNode
    print("\nTesting VideoMutingNode:")
    node = VideoMutingNode()
    result = asyncio.run(node.execute({
        "video_path": "test_video.mp4"
    }, logger))
    print(f"Result: {result}")
