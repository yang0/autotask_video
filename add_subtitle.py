try:
    from autotask.nodes import Node, register_node
except ImportError:
    from stub import Node, register_node

from typing import Dict, Any
import os
import subprocess
import logging


@register_node
class VideoSubtitleAdder(Node):
    NAME = "Video Subtitle Adder"
    DESCRIPTION = "Add subtitles to video files using ffmpeg"

    INPUTS = {
        "video_file": {
            "label": "Video File",
            "description": "Input video file to add subtitles to",
            "type": "STRING",
            "required": True,
            "widget": "FILE"
        },
        "subtitle_file": {
            "label": "Subtitle File",
            "description": "Subtitle file (.srt, .ass, .ssa, .vtt formats supported)",
            "type": "STRING",
            "required": True,
            "widget": "FILE"
        },
        "output_dir": {
            "label": "Output Directory",
            "description": "Directory to save the output video",
            "type": "STRING",
            "required": True,
            "widget": "DIR"
        }
    }

    OUTPUTS = {
        "output_file": {
            "label": "Output Video File",
            "description": "Path to the video file with embedded subtitles",
            "type": "STRING"
        }
    }

    def _get_output_filename(self, video_path: str) -> str:
        """Generate output filename with _subtitled suffix"""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        ext = os.path.splitext(video_path)[1]
        return f"{base_name}_subtitled{ext}"

    def _detect_subtitle_format(self, subtitle_path: str) -> str:
        """Detect subtitle format from file extension"""
        ext = os.path.splitext(subtitle_path)[1].lower().lstrip('.')
        if ext in ['srt', 'ass', 'ssa', 'vtt']:
            return ext
        raise ValueError(f"Unsupported subtitle format: {ext}")

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            video_path = node_inputs["video_file"]
            subtitle_path = node_inputs["subtitle_file"]
            output_dir = node_inputs["output_dir"]

            # Generate output filename
            output_filename = self._get_output_filename(video_path)
            output_path = os.path.join(output_dir, output_filename)

            workflow_logger.info(f"Adding subtitles to video: {video_path}")
            workflow_logger.info(f"Subtitle file: {subtitle_path}")
            workflow_logger.info(f"Output will be saved as: {output_path}")

            # Detect subtitle format
            subtitle_format = self._detect_subtitle_format(subtitle_path)
            workflow_logger.info(f"Detected subtitle format: {subtitle_format}")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Build ffmpeg command
            # For SRT/VTT: use subtitles filter
            # For ASS/SSA: use ass filter
            if subtitle_format in ['srt', 'vtt']:
                filter_complex = f"subtitles='{subtitle_path}'"
            else:  # ass, ssa
                filter_complex = f"ass='{subtitle_path}'"

            # Construct ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', filter_complex,
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-y',  # Overwrite output file if exists
                output_path
            ]

            # Execute ffmpeg command
            workflow_logger.info(f"Executing ffmpeg command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Monitor process output
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    workflow_logger.debug(output.strip())

            # Check if process was successful
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            workflow_logger.info(f"Successfully added subtitles to video at: {output_path}")
            
            return {
                "success": True,
                "output_file": output_path
            }

        except Exception as e:
            error_msg = f"Failed to add subtitles to video: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg
            }


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test VideoSubtitleAdder
    import asyncio
    
    async def test_subtitle_adder():
        node = VideoSubtitleAdder()
        result = await node.execute({
            "video_file": "test.mp4",
            "subtitle_file": "test.srt",
            "output_dir": "output"
        }, logger)
        print(f"Result: {result}")
    
    asyncio.run(test_subtitle_adder())
