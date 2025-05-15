try:
    from autotask.nodes import Node, register_node
except ImportError:
    from stub import Node, register_node


from typing import Dict, Any
import os
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector, ThresholdDetector
import subprocess



@register_node
class VideoSlicingNode(Node):
    NAME = "Video Slicing"
    DESCRIPTION = "Split a video into scenes based on content changes"

    INPUTS = {
        "video_path": {
            "label": "Video File",
            "description": "Path to the video file to be sliced",
            "type": "STRING",
            "required": True,
            "widget": "FILE"
        },
        "output_dir": {
            "label": "Output Directory",
            "description": "Directory where sliced videos will be saved",
            "type": "STRING",
            "required": False,
            "widget": "DIR"
        },
        "threshold": {
            "label": "Detection Threshold",
            "description": "Threshold for scene detection (higher values = more sensitive)",
            "type": "INT",
            "default": 15,
            "required": False
        },
        "min_clip_frames": {
            "label": "Minimum Clip Frames",
            "description": "Minimum number of frames for a scene to be considered valid",
            "type": "INT",
            "default": 15,
            "required": False
        },
        "skip_end_frames": {
            "label": "Skip End Frames",
            "description": "Number of frames to skip at the end of each scene",
            "type": "INT",
            "default": 2,
            "required": False
        },
        "frame_skip": {
            "label": "Frame Skip",
            "description": "Number of frames to skip during detection (higher values = faster but less accurate)",
            "type": "INT",
            "default": 2,
            "required": False
        }
    }

    OUTPUTS = {
        "video_paths": {
            "label": "Video Paths",
            "description": "Array of paths to the sliced video files",
            "type": "List"
        }
    }
    
    def __init__(self):
        super().__init__()
        self._stop_flag = False
        self._current_process = None
    
    async def stop(self) -> None:
        self._stop_flag = True
        if self._current_process is not None:
            try:
                self._current_process.terminate()
                self._current_process = None
            except:
                pass

    def _custom_invoke_command(self, call_list):
        """Custom command invoker that tracks the process"""
        try:
            self._current_process = subprocess.Popen(
                call_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = self._current_process.communicate()
            return self._current_process.returncode
        finally:
            self._current_process = None

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            # Extract parameters from inputs
            video_path = node_inputs["video_path"]
            output_dir = node_inputs.get("output_dir", None)
            threshold = node_inputs.get("threshold", 10)
            min_clip_frames = node_inputs.get("min_clip_frames", 15)
            skip_end_frames = node_inputs.get("skip_end_frames", 5)
            frame_skip = node_inputs.get("frame_skip", 12)
            
            workflow_logger.info(f"Starting video slicing for: {video_path}")
            
            # Reset stop flag
            self._stop_flag = False
            
            # Open video and create scene manager
            video = open_video(video_path)
            scene_manager = SceneManager()
            
            # Add detectors
            scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=10))
            threshold_detector = ThresholdDetector(threshold=threshold, min_scene_len=10, fade_bias=-100)
            scene_manager.add_detector(threshold_detector)
            
            # Detect scenes
            workflow_logger.info("Detecting scenes...")
            scene_manager.detect_scenes(video, show_progress=True, frame_skip=frame_skip)
            scene_list = scene_manager.get_scene_list()
            
            workflow_logger.info(f"Detected {len(scene_list)} potential scenes")
            
            # Filter scenes based on minimum clip frames
            new_scene_list = []
            for start, end in scene_list:
                frames_in_scene = end.get_frames() - start.get_frames()
                if frames_in_scene < min_clip_frames:
                    workflow_logger.debug(f"Skipping scene with {frames_in_scene} frames (below minimum)")
                    continue
                else:
                    # Remove frames from the end of each scene
                    end -= skip_end_frames
                    new_scene_list.append((start, end))
            
            workflow_logger.info(f"Filtered to {len(new_scene_list)} valid scenes")
            
            # Determine output directory
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            if output_dir:
                video_output_dir = os.path.join(output_dir, video_name)
            else:
                video_output_dir = os.path.join(os.path.dirname(video_path), "clips")
            
            # Create output directory if it doesn't exist
            if not os.path.exists(video_output_dir):
                os.makedirs(video_output_dir)
                workflow_logger.info(f"Created output directory: {video_output_dir}")
            
            # Split the video into scenes
            workflow_logger.info("Splitting video into scenes...")
            
            # Custom formatter to check stop flag during ffmpeg split
            def custom_formatter(video, scene):
                if self._stop_flag:
                    raise InterruptedError("Operation interrupted by user")
                return f"{video.name}-Scene-{scene.index + 1:03d}.mp4"
            
            try:
                # Monkey patch the invoke_command function to use our custom one
                from scenedetect.platform import invoke_command as original_invoke
                from scenedetect.platform import invoke_command
                invoke_command = self._custom_invoke_command
                
                split_video_ffmpeg(
                    input_video_path=video_path, 
                    scene_list=new_scene_list, 
                    output_dir=video_output_dir, 
                    show_progress=True,
                    formatter=custom_formatter
                )
                
                # Restore original invoke_command
                invoke_command = original_invoke
                
            except InterruptedError as e:
                workflow_logger.info(str(e))
                return {
                    "success": False,
                    "error_message": str(e)
                }
            
            if self._stop_flag:
                return {
                    "success": False,
                    "error_message": "Operation interrupted by user"
                }
            
            # Get all video file paths
            video_paths = []
            for root, dirs, files in os.walk(video_output_dir):
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov')):
                        video_paths.append(os.path.join(root, file))
            
            workflow_logger.info(f"Video slicing completed successfully. Output directory: {video_output_dir}")
            
            return {
                "success": True,
                "video_paths": video_paths
            }
            
        except Exception as e:
            error_msg = f"Video slicing failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg
            }
    
    def _move_scenes_to_subdirectories(self, base_dir):
        """Move video files to individual subdirectories based on their filenames"""
        for file_name in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.mp4', '.avi', '.mov')):
                # Create subdirectory with the same name as the video file (without extension)
                video_subdir = os.path.join(base_dir, os.path.splitext(file_name)[0])
                if not os.path.exists(video_subdir):
                    os.makedirs(video_subdir)
                
                # Move the video file to its subdirectory
                shutil.move(file_path, video_subdir)


if __name__ == "__main__":
    # Setup basic logging
    import logging
    import asyncio
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test VideoSlicingNode
    print("\nTesting VideoSlicingNode:")
    node = VideoSlicingNode()
    result = asyncio.run(node.execute({
        "video_path": "test_video.mp4",
        "threshold": 10,
        "min_clip_frames": 15,
        "skip_end_frames": 5
    }, logger))
    print(f"Result: {result}")
