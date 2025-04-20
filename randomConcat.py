try:
    from autotask.nodes import Node, register_node
except ImportError:
    from stub import Node, register_node

from typing import Dict, Any, List
import os
import random
import subprocess
import time
import datetime
from pathlib import Path
import traceback


@register_node
class RandomVideoConcatenationNode(Node):
    NAME = "Random Video Concatenation"
    DESCRIPTION = "Combine multiple video clips into a single video of specified duration"

    INPUTS = {
        "input_directory": {
            "label": "Input Directory",
            "description": "Directory containing video clips to concatenate",
            "type": "STRING",
            "required": True,
            "widget": "DIR"
        },
        "target_duration": {
            "label": "Target Duration (seconds)",
            "description": "Target duration of the output video in seconds",
            "type": "INT",
            "required": True,
            "default": 60
        },
        "output_directory": {
            "label": "Output Directory",
            "description": "Directory where the concatenated video will be saved",
            "type": "STRING",
            "required": False,
            "widget": "DIR"
        }
    }

    OUTPUTS = {
        "output_video_path": {
            "label": "Output Video Path",
            "description": "Path to the generated concatenated video",
            "type": "STRING"
        },
        "actual_duration": {
            "label": "Actual Duration",
            "description": "Actual duration of the generated video in seconds",
            "type": "FLOAT"
        },
        "clips_used": {
            "label": "Clips Used",
            "description": "Number of clips used in the concatenation",
            "type": "INT"
        }
    }

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            # Extract parameters from inputs
            input_directory = node_inputs["input_directory"]
            target_duration = node_inputs["target_duration"]
            output_directory = node_inputs.get("output_directory", None)
            
            workflow_logger.info(f"Starting random video concatenation from directory: {input_directory}")
            workflow_logger.info(f"Target duration: {target_duration} seconds")
            
            # Find all video files in the input directory and its subdirectories
            video_files = self._find_video_files(input_directory)
            workflow_logger.info(f"Found {len(video_files)} video files")
            
            if not video_files:
                error_msg = "No video files found in the input directory"
                workflow_logger.error(error_msg)
                return {
                    "success": False,
                    "error_message": error_msg
                }
            
            # Generate metadata file with video durations
            meta_file_path = os.path.join(input_directory, "meta.txt")
            self._generate_metadata_file(video_files, meta_file_path, workflow_logger)
            
            # Select random videos to meet the target duration
            selected_videos, total_duration = self._select_random_videos(
                video_files, target_duration, workflow_logger
            )
            
            if not selected_videos:
                error_msg = "Could not select videos to meet the target duration"
                workflow_logger.error(error_msg)
                return {
                    "success": False,
                    "error_message": error_msg
                }
            
            # Determine output directory and filename
            if output_directory:
                output_dir = output_directory
            else:
                output_dir = os.path.dirname(input_directory)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"output_{timestamp}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Concatenate the selected videos
            workflow_logger.info(f"Concatenating {len(selected_videos)} videos...")
            success = self._concatenate_videos(selected_videos, output_path, workflow_logger)
            
            if not success:
                error_msg = "Failed to concatenate videos"
                workflow_logger.error(error_msg)
                return {
                    "success": False,
                    "error_message": error_msg
                }
            
            workflow_logger.info(f"Video concatenation completed successfully. Output: {output_path}")
            
            return {
                "success": True,
                "output_video_path": output_path,
                "actual_duration": total_duration,
                "clips_used": len(selected_videos)
            }
            
        except Exception as e:
            error_msg = f"Video concatenation failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg
            }
    
    def _find_video_files(self, directory: str) -> List[str]:
        """Find all video files in the directory and its subdirectories"""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
        video_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, file))
        
        return video_files
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get the duration of a video file using ffprobe"""
        try:
            cmd = [
                'ffprobe', 
                '-v', 'error', 
                '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            return duration
        except Exception:
            traceback.print_exc()
            # If ffprobe fails, return a default duration
            return 0.0
    
    def _generate_metadata_file(self, video_files: List[str], meta_file_path: str, workflow_logger) -> None:
        """Generate a metadata file with video durations"""
        try:
            with open(meta_file_path, 'w', encoding='utf-8') as f:
                for video_path in video_files:
                    duration = self._get_video_duration(video_path)
                    relative_path = os.path.relpath(video_path, os.path.dirname(meta_file_path))
                    f.write(f"{relative_path} {duration}\n")
            
            workflow_logger.info(f"Generated metadata file: {meta_file_path}")
        except Exception as e:
            workflow_logger.warning(f"Failed to generate metadata file: {str(e)}")
    
    def _select_random_videos(self, video_files: List[str], target_duration: int, workflow_logger) -> tuple:
        """Select random videos to meet the target duration"""
        selected_videos = []
        total_duration = 0.0
        
        # Shuffle the video files
        random.shuffle(video_files)
        
        for video_path in video_files:
            duration = self._get_video_duration(video_path)
            
            if duration <= 0:
                workflow_logger.warning(f"Skipping video with invalid duration: {video_path}")
                continue
            
            selected_videos.append(video_path)
            total_duration += duration
            
            if total_duration >= target_duration:
                break
        
        return selected_videos, total_duration
    
    def _concatenate_videos(self, video_files: List[str], output_path: str, workflow_logger) -> bool:
        """Concatenate the selected videos into a single output video"""
        try:
            # Create a temporary file listing the videos to concatenate
            temp_list_file = os.path.join(os.path.dirname(output_path), "temp_concat_list.txt")
            
            with open(temp_list_file, 'w', encoding='utf-8') as f:
                for video_path in video_files:
                    # Use absolute paths to avoid issues
                    abs_path = os.path.abspath(video_path)
                    f.write(f"file '{abs_path}'\n")
            
            # Use ffmpeg to concatenate the videos
            cmd = [
                'ffmpeg', 
                '-f', 'concat', 
                '-safe', '0', 
                '-i', temp_list_file, 
                '-c', 'copy', 
                output_path
            ]
            
            workflow_logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up the temporary file
            if os.path.exists(temp_list_file):
                os.remove(temp_list_file)
            
            if result.returncode != 0:
                workflow_logger.error(f"FFmpeg error: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            workflow_logger.error(f"Error concatenating videos: {str(e)}")
            return False


if __name__ == "__main__":
    # Setup basic logging
    import logging
    import asyncio
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test RandomVideoConcatenationNode
    print("\nTesting RandomVideoConcatenationNode:")
    node = RandomVideoConcatenationNode()
    result = asyncio.run(node.execute({
        "input_directory": "path/to/video/clips",
        "target_duration": 60,
        "output_directory": "path/to/output"
    }, logger))
    print(f"Result: {result}")
