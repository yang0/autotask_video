try:
    from autotask.nodes import Node, register_node
    from autotask.api_keys import get_api_key
except ImportError:
    from stub import Node, register_node

from typing import Dict, Any, List, Tuple
import os
import re
import subprocess
import tempfile
from datetime import datetime
import wave
import contextlib
import asyncio
import dashscope
from dashscope.audio.tts_v2 import *


DASHSCOPE_API_KEY = get_api_key(provider="dashscope.aliyuncs.com", key_name="DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY


# Voice code mapping with detailed specifications
VOICE_MAPPING = {
    "龙婉 (中文女声, 语音助手/导航/数字人)": "longwan",
    "龙橙 (中文女声, 语音助手/导航/数字人)": "longcheng",
    "龙华 (中文女声, 语音助手/导航/数字人)": "longhua",
    "龙小淳 (中英文女声, 语音助手/导航/数字人)": "longxiaochun",
    "龙小夏 (中文女声, 语音助手/数字人)": "longxiaoxia",
    "龙小诚 (中英文男声, 语音助手/导航/数字人)": "longxiaocheng",
    "龙小白 (中文女声, 数字人/有声书/语音助手)": "longxiaobai",
    "龙老铁 (中文东北口音男声, 新闻/有声书/导航/直播)": "longlaotie",
    "龙书 (中文男声, 有声书/导航/新闻/客服)": "longshu",
    "龙硕 (中文男声, 语音助手/导航/新闻/客服)": "longshuo",
    "龙婧 (中文女声, 语音助手/导航/新闻/客服)": "longjing",
    "龙妙 (中文女声, 客服/导航/有声书/语音助手)": "longmiao",
    "龙悦 (中文女声, 语音助手/诗词/有声书/导航/新闻)": "longyue",
    "龙媛 (中文女声, 有声书/语音助手/数字人)": "longyuan",
    "龙飞 (中文男声, 会议/新闻/有声书)": "longfei",
    "龙杰力豆 (中英文男声, 新闻/有声书/聊天助手)": "longjielidou",
    "龙彤 (中文女声, 有声书/导航/数字人)": "longtong",
    "龙祥 (中文男声, 新闻/有声书/导航)": "longxiang",
    "Stella (中英文女声, 语音助手/直播/导航/客服/有声书)": "loongstella",
    "Bella (中文女声, 语音助手/客服/新闻/导航)": "loongbella"
}

# Audio specifications for all voices
AUDIO_SPECS = {
    "model": "cosyvoice-v1",
    "sample_rate": 22050,
    "format": "mp3"
}


class ThreadSafeAsyncioEvent(asyncio.Event):
    def set(self):
        self._loop.call_soon_threadsafe(super().set)


class TtsCallback(ResultCallback):
    def __init__(self, output_file: str, complete_event: ThreadSafeAsyncioEvent):
        self.file = open(output_file, 'wb')
        self.complete_event = complete_event
    
    def on_open(self):
        pass

    def on_complete(self):
        self.complete_event.set()

    def on_error(self, message: str):
        raise Exception(f"TTS synthesis failed: {message}")

    def on_close(self):
        self.file.close()

    def on_event(self, message):
        pass

    def on_data(self, data: bytes) -> None:
        self.file.write(data)


@register_node
class SubtitleToSpeech(Node):
    NAME = "Subtitle to Speech"
    DESCRIPTION = "Convert subtitle text to speech with dynamic speed adjustment to match subtitle timing using Alibaba Cloud TTS"

    INPUTS = {
        "subtitle_file": {
            "label": "Subtitle File",
            "description": "Input subtitle file (.srt format)",
            "type": "STRING",
            "required": True,
            "widget": "FILE"
        },
        "output_dir": {
            "label": "Output Directory",
            "description": "Directory to save the generated audio files",
            "type": "STRING",
            "required": True,
            "widget": "DIR"
        },
        "voice_code": {
            "label": "Voice",
            "description": "Select TTS voice",
            "type": "COMBO",
            "required": True,
            "options": list(VOICE_MAPPING.keys())
        }
    }

    OUTPUTS = {
        "output_file": {
            "label": "Generated Audio File",
            "description": "Path to the generated audio file",
            "type": "STRING"
        }
    }

    def _parse_srt(self, file_path: str) -> List[Dict[str, str]]:
        """Parse SRT file and return list of subtitle blocks with timing"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        subtitles = []
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            subtitles.append({
                'index': match.group(1),
                'start': match.group(2),
                'end': match.group(3),
                'text': match.group(4).strip().replace('\n', ' ')
            })

        return subtitles

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert SRT time format to seconds"""
        h, m, s = time_str.replace(',', '.').split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)

    def _generate_silence(self, duration: float, output_file: str) -> None:
        """Generate silence audio file with specified duration"""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', 'anullsrc=channel_layout=mono:sample_rate=44100',
            '-t', f'{duration:.3f}',
            '-ar', '44100',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            output_file
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _adjust_audio_duration(self, input_file: str, output_file: str, target_duration: float) -> None:
        """Adjust audio duration to match target duration"""
        # Get audio duration using ffprobe
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        audio_duration = float(result.stdout.strip())

        if audio_duration < target_duration:
            # If audio is shorter, add silence
            silence_duration = target_duration - audio_duration
            silence_file = os.path.join(os.path.dirname(output_file), 'silence.wav')
            self._generate_silence(silence_duration, silence_file)

            # Concatenate audio and silence
            concat_file = os.path.join(os.path.dirname(output_file), 'concat.txt')
            with open(concat_file, 'w') as f:
                f.write(f"file '{input_file}'\nfile '{silence_file}'\n")

            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:a', 'pcm_s16le',  # Convert to WAV format
                output_file
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(concat_file)
            os.remove(silence_file)

        elif audio_duration > target_duration:
            # If audio is longer, adjust speed
            speed = audio_duration / target_duration
            cmd = [
                'ffmpeg', '-y',
                '-i', input_file,
                '-filter:a', f'atempo={speed:.2f}',
                '-c:a', 'pcm_s16le',  # Convert to WAV format
                output_file
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            # If durations match, convert to WAV format
            cmd = [
                'ffmpeg', '-y',
                '-i', input_file,
                '-c:a', 'pcm_s16le',  # Convert to WAV format
                output_file
            ]
            subprocess.run(cmd, check=True, capture_output=True)

    def _concatenate_audio_files(self, audio_files: List[str], output_file: str) -> None:
        """Concatenate multiple audio files into one"""
        concat_file = os.path.join(os.path.dirname(output_file), 'concat.txt')
        with open(concat_file, 'w') as f:
            for file in audio_files:
                f.write(f"file '{file}'\n")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            output_file
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(concat_file)

    async def _synthesize_text(self, text: str, output_file: str, voice_code: str) -> None:
        """Synthesize text to speech using Alibaba Cloud TTS"""
        complete_event = ThreadSafeAsyncioEvent()
        synthesizer_callback = TtsCallback(output_file, complete_event)

        # Initialize the speech synthesizer
        speech_synthesizer = SpeechSynthesizer(
            model='cosyvoice-v1',
            voice=voice_code,
            callback=synthesizer_callback
        )

        # Synthesize text
        speech_synthesizer.streaming_call(text)
        speech_synthesizer.async_streaming_complete()
        
        # Wait for synthesis to complete
        await complete_event.wait()

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            subtitle_path = node_inputs["subtitle_file"]
            output_dir = node_inputs["output_dir"]
            voice_name = node_inputs["voice_code"]
            voice_code = VOICE_MAPPING[voice_name]

            workflow_logger.info(f"Processing subtitle file: {subtitle_path}")

            # Parse subtitle file
            subtitles = self._parse_srt(subtitle_path)
            workflow_logger.info(f"Found {len(subtitles)} subtitle blocks")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            audio_files = []
            current_time = 0.0

            # Process each subtitle block
            for i, sub in enumerate(subtitles):
                workflow_logger.info(f"Processing subtitle block {i+1}/{len(subtitles)}")
                
                # Generate TTS audio
                temp_audio = os.path.join(output_dir, f'temp_{i}.wav')
                await self._synthesize_text(sub['text'], temp_audio, voice_code)

                # Calculate target duration
                start_time = self._time_to_seconds(sub['start'])
                end_time = self._time_to_seconds(sub['end'])
                target_duration = end_time - start_time

                # Add silence if needed at the start
                if start_time > current_time:
                    silence_duration = start_time - current_time
                    silence_file = os.path.join(output_dir, f'silence_{i}.wav')
                    self._generate_silence(silence_duration, silence_file)
                    audio_files.append(silence_file)
                    current_time = start_time

                # Adjust audio duration
                adjusted_audio = os.path.join(output_dir, f'adjusted_{i}.wav')
                self._adjust_audio_duration(temp_audio, adjusted_audio, target_duration)
                audio_files.append(adjusted_audio)
                current_time = end_time

                # Clean up temporary files
                os.remove(temp_audio)

            # Concatenate all audio files
            output_file = os.path.join(output_dir, 'final_audio.wav')
            self._concatenate_audio_files(audio_files, output_file)

            # Clean up intermediate files
            for file in audio_files:
                os.remove(file)

            workflow_logger.info(f"Successfully generated audio file: {output_file}")
            return {
                "success": True,
                "output_file": output_file
            }

        except Exception as e:
            error_msg = f"Failed to convert subtitles to speech: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg
            }
