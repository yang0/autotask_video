try:
    from autotask.nodes import Node, register_node
except ImportError:
    from stub import Node, register_node

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path


@register_node
class KeyFrameExtractor(Node):
    NAME = "Video Key Frame Extractor"
    DESCRIPTION = "Extract evenly distributed key frames from a video file"
    CATEGORY = "Video Processing"
    MAINTAINER = "AutoTask Team"
    ICON = "🎬"

    INPUTS = {
        "video_path": {
            "label": "Video Path",
            "description": "Path to the video file",
            "type": "STRING",
            "widget": "FILE",
            "required": True,
            "default": "",
            "placeholder": "Select video file"
        },
        "num_frames": {
            "label": "Number of Frames",
            "description": "Number of key frames to extract (1-8)",
            "type": "INT",
            "required": True,
            "default": 3,
            "minimum": 1,
            "maximum": 8
        },
        "skip_start": {
            "label": "Skip Start Frames",
            "description": "Number of frames to skip from the start",
            "type": "INT",
            "required": True,
            "default": 5,
            "minimum": 0
        },
        "skip_end": {
            "label": "Skip End Frames",
            "description": "Number of frames to skip from the end",
            "type": "INT",
            "required": True,
            "default": 5,
            "minimum": 0
        },
        "output_dir": {
            "label": "Output Directory",
            "description": "Directory to save extracted frames",
            "type": "STRING",
            "widget": "DIR",
            "required": True,
            "default": "",
            "placeholder": "Select output directory"
        }
    }

    OUTPUTS = {
        "frame1": {
            "label": "Frame 1 Path",
            "description": "Path to the first extracted frame",
            "type": "STRING",
            "default": ""
        },
        "frame2": {
            "label": "Frame 2 Path",
            "description": "Path to the second extracted frame",
            "type": "STRING",
            "default": ""
        },
        "frame3": {
            "label": "Frame 3 Path",
            "description": "Path to the third extracted frame",
            "type": "STRING",
            "default": ""
        },
        "frame4": {
            "label": "Frame 4 Path",
            "description": "Path to the fourth extracted frame",
            "type": "STRING",
            "default": ""
        },
        "frame5": {
            "label": "Frame 5 Path",
            "description": "Path to the fifth extracted frame",
            "type": "STRING",
            "default": ""
        },
        "frame6": {
            "label": "Frame 6 Path",
            "description": "Path to the sixth extracted frame",
            "type": "STRING",
            "default": ""
        },
        "frame7": {
            "label": "Frame 7 Path",
            "description": "Path to the seventh extracted frame",
            "type": "STRING",
            "default": ""
        },
        "frame8": {
            "label": "Frame 8 Path",
            "description": "Path to the eighth extracted frame",
            "type": "STRING",
            "default": ""
        },
        "success": {
            "label": "Success",
            "description": "Whether the operation was successful",
            "type": "BOOL",
            "default": False
        },
        "error_message": {
            "label": "Error Message",
            "description": "Error message if the operation failed",
            "type": "STRING",
            "default": ""
        }
    }

    def _get_frame_positions(self, total_frames: int, num_frames: int, skip_start: int, skip_end: int) -> List[int]:
        """Calculate evenly distributed frame positions for extraction."""
        # Calculate effective frame range
        start_frame = skip_start
        end_frame = total_frames - skip_end
        effective_frames = end_frame - start_frame
        
        if effective_frames < num_frames:
            raise ValueError(f"Not enough frames in video after skipping. Need {num_frames}, but only have {effective_frames}")
        
        # Calculate interval between frames for even distribution
        interval = effective_frames / (num_frames + 1)
        
        # Generate evenly distributed frame positions
        positions = []
        for i in range(1, num_frames + 1):
            # Calculate position using the interval
            position = int(start_frame + (interval * i))
            # Ensure position is within bounds
            position = max(start_frame, min(position, end_frame - 1))
            positions.append(position)
        
        return positions

    def _extract_frames(self, video_path: str, frame_positions: List[int], workflow_logger) -> List[np.ndarray]:
        """Extract frames at specified positions from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame at position {pos}")
            frames.append(frame)
            
        cap.release()
        return frames

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            video_path = node_inputs["video_path"]
            num_frames = node_inputs["num_frames"]
            skip_start = node_inputs["skip_start"]
            skip_end = node_inputs["skip_end"]
            output_dir = node_inputs["output_dir"]
            
            # Validate inputs
            if not os.path.exists(video_path):
                raise ValueError("Video file does not exist")
                
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Open video and get properties
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if total_frames <= 0:
                raise ValueError("Could not read video frame count")
            
            # Calculate frame positions
            workflow_logger.info("Calculating frame positions...")
            frame_positions = self._get_frame_positions(total_frames, num_frames, skip_start, skip_end)
            
            # Extract frames
            workflow_logger.info(f"Extracting {len(frame_positions)} frames...")
            frames = self._extract_frames(video_path, frame_positions, workflow_logger)
            
            # Save frames
            results = {f"frame{i+1}": "" for i in range(8)}
            results.update({"success": True, "error_message": ""})
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            for i, frame in enumerate(frames):
                frame_path = os.path.join(output_dir, f"{video_name}_frame_{i+1}.jpg")
                cv2.imwrite(frame_path, frame)
                results[f"frame{i+1}"] = frame_path
                workflow_logger.info(f"Saved frame {i+1} to {frame_path} (position: {frame_positions[i]})")
            
            workflow_logger.info("Frame extraction completed successfully")
            return results
            
        except Exception as e:
            error_msg = f"Failed to extract frames: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg,
                **{f"frame{i+1}": "" for i in range(8)}
            }
