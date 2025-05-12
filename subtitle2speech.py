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
        },
        "use_local_files": {
            "label": "Use Local Files",
            "description": "Use existing intermediate audio files if available",
            "type": "BOOLEAN",
            "required": False,
            "default": False
        }
    }

    OUTPUTS = {
        "output_file": {
            "label": "Generated Audio File",
            "description": "Path to the generated audio file",
            "type": "STRING"
        }
    }

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert SRT time format to seconds"""
        # Handle format like "00:00:02,820"
        h, m, s = time_str.replace(',', '.').split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)

    def _parse_srt(self, file_path: str) -> List[Dict[str, str]]:
        """Parse SRT file and return list of subtitle blocks with timing"""
        print(f"\nReading subtitle file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"File size: {len(content)} bytes")
            print(f"First 200 characters:\n{content[:200]}")
            print("-" * 50)

        subtitles = []
        # Split content into subtitle blocks
        blocks = content.strip().split('\n\n')
        print(f"Found {len(blocks)} blocks in total")
        
        for block_idx, block in enumerate(blocks):
            lines = block.strip().split('\n')
            if len(lines) < 2:  # Skip empty blocks
                continue
                
            try:
                # Parse timing (first line should be timing)
                timing = lines[0].strip()
                if '-->' not in timing:  # If first line is not timing, try second line
                    if len(lines) < 3:
                        continue
                    timing = lines[1].strip()
                    if '-->' not in timing:
                        continue
                
                start_time, end_time = timing.split(' --> ')
                
                # Parse text (all lines after timing)
                text_lines = []
                for line in lines:
                    if '-->' not in line:  # Skip timing line
                        line = line.strip()
                        if line and not line.isdigit():  # Skip empty lines and index numbers
                            text_lines.append(line)
                
                text = ' '.join(text_lines)
                
                if text:  # Only add if we have text
                    subtitle = {
                        'index': str(block_idx + 1),  # Use block index as subtitle index
                        'start': start_time,
                        'end': end_time,
                        'text': text
                    }
                    subtitles.append(subtitle)
                    print(f"\nParsed subtitle block {block_idx + 1}:")
                    print(f"Timing: {start_time} --> {end_time}")
                    print(f"Text: {text}")
                    print("-" * 50)
            except Exception as e:
                print(f"\nError parsing block {block_idx + 1}:")
                print(f"Error: {e}")
                print(f"Block content:\n{block}")
                print("-" * 50)
                continue

        # Sort subtitles by start time to ensure correct order
        subtitles.sort(key=lambda x: self._time_to_seconds(x['start']))
        
        print(f"\nTotal valid subtitles parsed: {len(subtitles)}")
        if subtitles:
            print(f"First subtitle at {subtitles[0]['start']}: {subtitles[0]['text'][:50]}...")
            print(f"Last subtitle at {subtitles[-1]['start']}: {subtitles[-1]['text'][:50]}...")
        return subtitles

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
        """Adjust audio duration to match target duration by adjusting speed or adding silence"""
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
        print(f"[Adjust] {os.path.basename(input_file)}: audio_duration={audio_duration:.3f}s, target_duration={target_duration:.3f}s")

        if abs(audio_duration - target_duration) < 0.1:  # If duration is very close, just copy
            cmd = [
                'ffmpeg', '-y',
                '-i', input_file,
                '-c:a', 'pcm_s16le',  # Convert to WAV format
                output_file
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return

        # If audio is shorter than target duration, add silence
        if audio_duration < target_duration:
            silence_duration = target_duration - audio_duration
            silence_file = os.path.join(os.path.dirname(output_file), 'temp_silence.wav')
            # Generate silence
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', 'anullsrc=channel_layout=mono:sample_rate=22050',
                '-t', f'{silence_duration:.3f}',
                '-ar', '22050',
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                silence_file
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            # Create concat file
            concat_file = os.path.join(os.path.dirname(output_file), 'concat.txt')
            with open(concat_file, 'w') as f:
                f.write(f"file '{input_file}'\n")
                f.write(f"file '{silence_file}'\n")
            # Concatenate audio and silence
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:a', 'pcm_s16le',
                output_file
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            # Clean up temporary files
            os.remove(concat_file)
            os.remove(silence_file)
            return

        # If audio is longer than target duration, adjust speed
        if audio_duration > target_duration:
            speed_factor = target_duration / audio_duration
            # atempo filter only supports 0.5-2.0 per filter, so chain if needed
            filters = []
            sf = speed_factor
            while sf < 0.5:
                filters.append('atempo=0.5')
                sf /= 0.5
            while sf > 2.0:
                filters.append('atempo=2.0')
                sf /= 2.0
            filters.append(f'atempo={sf:.5f}')
            filter_str = ','.join(filters)
            cmd = [
                'ffmpeg', '-y',
                '-i', input_file,
                '-filter:a', filter_str,
                '-c:a', 'pcm_s16le',
                output_file
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return

    def _concatenate_audio_files(self, audio_files: List[str], output_file: str) -> None:
        """Concatenate multiple audio files into one"""
        # Log total duration of input files
        total_input_duration = 0
        for file in audio_files:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            total_input_duration += duration
            print(f"Input file {os.path.basename(file)} duration: {duration:.2f}s")
        print(f"Total input duration: {total_input_duration:.2f}s")

        concat_file = os.path.join(os.path.dirname(output_file), 'concat.txt')
        with open(concat_file, 'w') as f:
            for file in audio_files:
                f.write(f"file '{file}'\n")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',  # Use direct copy for concatenation
            output_file
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(concat_file)

        # Check output file duration
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            output_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_duration = float(result.stdout.strip())
        print(f"Output file duration: {output_duration:.2f}s")
        if abs(output_duration - total_input_duration) > 0.1:
            print(f"Warning: Duration mismatch! Difference: {abs(output_duration - total_input_duration):.2f}s")

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

    def _check_audio_duration(self, audio_file: str, logger, target_duration: float = None) -> float:
        """Check the duration of an audio file and log it, optionally with target duration"""
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        if target_duration is not None:
            logger.info(f"File: {os.path.basename(audio_file)}, Duration: {duration:.2f}s, Target: {target_duration:.2f}s")
        else:
            logger.info(f"File: {os.path.basename(audio_file)}, Duration: {duration:.2f}s")
        return duration

    def _check_audio_segments(self, audio_files: List[str], subtitles: List[Dict[str, str]], logger) -> None:
        """Check and log the duration of all audio segments"""
        total_duration = 0
        logger.info("\nChecking duration of all audio segments:")
        for audio_file in audio_files:
            duration = self._check_audio_duration(audio_file, logger)
            total_duration += duration
        
        logger.info(f"\nTotal duration of all segments: {total_duration:.2f}s")
        if subtitles:
            expected_duration = self._time_to_seconds(subtitles[-1]['end'])
            logger.info(f"Expected duration from subtitles: {expected_duration:.2f}s")
            if abs(total_duration - expected_duration) > 0.1:
                logger.warning(f"Duration mismatch! Difference: {abs(total_duration - expected_duration):.2f}s")

    def _collect_debug_audio(self, audio_files: List[str], output_dir: str, subtitles: List[Dict[str, str]]) -> None:
        """Collect and save intermediate audio files for debugging purposes"""
        debug_dir = os.path.join(output_dir, 'debug_audio')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save audio files with detailed information
        for i, file in enumerate(audio_files):
            # Get file duration
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            
            # Copy file to debug directory with detailed name
            base_name = os.path.basename(file)
            new_name = f"{i:03d}_{base_name}"
            new_path = os.path.join(debug_dir, new_name)
            
            # Copy the file
            cmd = [
                'ffmpeg', '-y',
                '-i', file,
                '-c:a', 'pcm_s16le',
                new_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Create a text file with information about this audio segment
            info_file = os.path.join(debug_dir, f"{i:03d}_info.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"Original file: {base_name}\n")
                f.write(f"Duration: {duration:.2f}s\n")
                if 'temp_' in base_name:
                    idx = int(base_name.split('_')[1].split('.')[0])
                    if idx < len(subtitles):
                        f.write(f"Subtitle text: {subtitles[idx]['text']}\n")
                        f.write(f"Subtitle timing: {subtitles[idx]['start']} --> {subtitles[idx]['end']}\n")
        
        print(f"\nDebug audio files have been saved to: {debug_dir}")
        print("Each audio file has a corresponding info.txt file with details")

    def _find_local_audio_files(self, output_dir: str) -> List[str]:
        """Find existing intermediate audio files in the output directory"""
        print(f"\nSearching for audio files in: {output_dir}")
        
        # Dictionary to store files by index
        files_by_index = {}
        
        # First check the debug_audio subdirectory if it exists
        debug_dir = os.path.join(output_dir, 'debug_audio')
        if os.path.exists(debug_dir):
            print(f"\nSearching in debug directory: {debug_dir}")
            for file in os.listdir(debug_dir):
                if file.endswith('.wav'):
                    try:
                        # Extract the numeric prefix and base name
                        prefix = int(file.split('_')[0])
                        base_name = file.split('_', 1)[1]  # Get the part after the prefix
                        full_path = os.path.join(debug_dir, file)
                        
                        # Check if the file matches our naming patterns
                        if base_name.startswith('temp_') or base_name.startswith('adjusted_'):
                            idx = int(base_name.split('_')[1].split('.')[0])
                            if idx not in files_by_index:
                                files_by_index[idx] = {'tts': None, 'start': None, 'end': None}
                            if base_name.startswith('adjusted_'):
                                files_by_index[idx]['tts'] = (idx, 0, full_path)
                            elif base_name.startswith('temp_') and files_by_index[idx]['tts'] is None:
                                files_by_index[idx]['tts'] = (idx, 0, full_path)
                        elif base_name.startswith('silence_start_'):
                            idx = int(base_name.split('_')[2].split('.')[0])
                            if idx not in files_by_index:
                                files_by_index[idx] = {'tts': None, 'start': None, 'end': None}
                            files_by_index[idx]['start'] = (idx, -1, full_path)
                        elif base_name.startswith('silence_end_'):
                            idx = int(base_name.split('_')[2].split('.')[0])
                            if idx not in files_by_index:
                                files_by_index[idx] = {'tts': None, 'start': None, 'end': None}
                            files_by_index[idx]['end'] = (idx, 1, full_path)
                    except (ValueError, IndexError) as e:
                        print(f"Error processing debug file {file}: {e}")
                        continue
        
        # If no files found in debug directory, check the main output directory
        if not files_by_index:
            print(f"\nNo files found in debug directory, checking main directory: {output_dir}")
            for file in os.listdir(output_dir):
                if file.endswith('.wav'):
                    try:
                        if (file.startswith('temp_') or 
                            file.startswith('adjusted_') or 
                            file.startswith('silence_start_') or 
                            file.startswith('silence_end_')):
                            full_path = os.path.join(output_dir, file)
                            print(f"Found audio file: {file}")
                            
                            if file.startswith('temp_') or file.startswith('adjusted_'):
                                idx = int(file.split('_')[1].split('.')[0])
                                if idx not in files_by_index:
                                    files_by_index[idx] = {'tts': None, 'start': None, 'end': None}
                                if file.startswith('adjusted_'):
                                    files_by_index[idx]['tts'] = (idx, 0, full_path)
                                elif file.startswith('temp_') and files_by_index[idx]['tts'] is None:
                                    files_by_index[idx]['tts'] = (idx, 0, full_path)
                            elif file.startswith('silence_start_'):
                                idx = int(file.split('_')[2].split('.')[0])
                                if idx not in files_by_index:
                                    files_by_index[idx] = {'tts': None, 'start': None, 'end': None}
                                files_by_index[idx]['start'] = (idx, -1, full_path)
                            elif file.startswith('silence_end_'):
                                idx = int(file.split('_')[2].split('.')[0])
                                if idx not in files_by_index:
                                    files_by_index[idx] = {'tts': None, 'start': None, 'end': None}
                                files_by_index[idx]['end'] = (idx, 1, full_path)
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue
        
        # Convert dictionary to sorted list
        audio_files = []
        for idx in sorted(files_by_index.keys()):
            files = files_by_index[idx]
            # Add files in the correct order: start silence -> TTS -> end silence
            if files['start']:
                audio_files.append(files['start'])
            if files['tts']:
                audio_files.append(files['tts'])
            if files['end']:
                audio_files.append(files['end'])
        
        # Print the final sorted list
        print("\nSorted audio files:")
        for _, _, file in audio_files:
            print(f"- {os.path.basename(file)}")
        
        # Return only the file paths
        return [f[2] for f in audio_files]

    def _ensure_wav_pcm(self, input_file: str, output_file: str, sample_rate: int = 22050) -> None:
        """Ensure the audio file is wav (pcm_s16le), specified sample rate, mono."""
        cmd = [
            'ffmpeg', '-y',
            '-i', input_file,
            '-ar', str(sample_rate),
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            output_file
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            subtitle_path = node_inputs["subtitle_file"]
            output_dir = node_inputs["output_dir"]
            voice_name = node_inputs["voice_code"]
            use_local_files = node_inputs.get("use_local_files", False)
            voice_code = VOICE_MAPPING[voice_name]

            workflow_logger.info(f"Processing subtitle file: {subtitle_path}")

            # Parse subtitle file
            subtitles = self._parse_srt(subtitle_path)
            workflow_logger.info(f"Found {len(subtitles)} subtitle blocks")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Check if we should use local files
            if use_local_files:
                local_files = self._find_local_audio_files(output_dir)
                if local_files:
                    workflow_logger.info(f"Found {len(local_files)} local audio files")
                    # Collect debug audio files for verification
                    self._collect_debug_audio(local_files, output_dir, subtitles)
                    
                    # Concatenate all audio files
                    output_file = os.path.join(output_dir, 'final_audio.wav')
                    self._concatenate_audio_files(local_files, output_file)
                    
                    # Check final audio duration
                    workflow_logger.info("\nChecking final audio duration:")
                    final_duration = self._check_audio_duration(output_file, workflow_logger)
                    expected_duration = self._time_to_seconds(subtitles[-1]['end'])
                    if abs(final_duration - expected_duration) > 0.1:
                        workflow_logger.warning(f"Final duration mismatch! Difference: {abs(final_duration - expected_duration):.2f}s")
                    
                    workflow_logger.info(f"Successfully generated audio file using local files: {output_file}")
                    return {
                        "success": True,
                        "output_file": output_file
                    }
                else:
                    workflow_logger.info("No local audio files found, proceeding with normal processing")

            audio_files = []
            current_time = 0.0

            # Process each subtitle block
            for i, sub in enumerate(subtitles):
                start_time = self._time_to_seconds(sub['start'])
                end_time = self._time_to_seconds(sub['end'])
                target_duration = end_time - start_time
                workflow_logger.info(f"Processing subtitle block {i+1}/{len(subtitles)} | Duration: {target_duration:.2f}s ({sub['start']} --> {sub['end']})")

                # Generate TTS audio
                temp_audio = os.path.join(output_dir, f'temp_{i}.wav')
                await self._synthesize_text(sub['text'], temp_audio, voice_code)
                # Ensure TTS output is in correct format
                temp_audio_fixed = os.path.join(output_dir, f'temp_{i}_fixed.wav')
                self._ensure_wav_pcm(temp_audio, temp_audio_fixed)
                os.replace(temp_audio_fixed, temp_audio)
                workflow_logger.info(f"Generated TTS audio for block {i+1}")
                # 获取TTS音频实际时长
                tts_duration = self._check_audio_duration(temp_audio, workflow_logger, target_duration)

                # 计算需要的静音时长
                silence_duration = start_time - current_time
                # 优先裁剪静音
                if silence_duration + tts_duration > target_duration:
                    silence_duration = max(0, target_duration - tts_duration)
                    workflow_logger.info(f"[Adjust] Silence for block {i+1} trimmed to {silence_duration:.2f}s to fit target duration.")
                # 如果静音>0，生成静音
                if silence_duration > 0.1:
                    silence_file = os.path.join(output_dir, f'silence_start_{i}.wav')
                    self._generate_silence(silence_duration, silence_file)
                    silence_file_fixed = os.path.join(output_dir, f'silence_start_{i}_fixed.wav')
                    self._ensure_wav_pcm(silence_file, silence_file_fixed)
                    os.replace(silence_file_fixed, silence_file)
                    audio_files.append(silence_file)
                    workflow_logger.info(f"Added start silence for block {i+1}")
                    self._check_audio_duration(silence_file, workflow_logger, silence_duration)
                current_time = start_time + silence_duration

                # 再次获取TTS音频时长（防止静音裁剪后current_time变化）
                tts_duration = self._check_audio_duration(temp_audio, workflow_logger, target_duration)
                # 如果TTS音频比剩余目标时长还长，需要压缩TTS
                tts_target = end_time - current_time
                if tts_duration > tts_target + 0.01:
                    adjusted_audio = os.path.join(output_dir, f'adjusted_{i}.wav')
                    # 修正 atempo 逻辑
                    speed_factor = tts_duration / tts_target
                    filters = []
                    sf = speed_factor
                    while sf > 2.0:
                        filters.append('atempo=2.0')
                        sf /= 2.0
                    while sf < 0.5:
                        filters.append('atempo=0.5')
                        sf /= 0.5
                    filters.append(f'atempo={sf:.5f}')
                    filter_str = ','.join(filters)
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_audio,
                        '-filter:a', filter_str,
                        '-c:a', 'pcm_s16le',
                        adjusted_audio
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    adjusted_audio_fixed = os.path.join(output_dir, f'adjusted_{i}_fixed.wav')
                    self._ensure_wav_pcm(adjusted_audio, adjusted_audio_fixed)
                    os.replace(adjusted_audio_fixed, adjusted_audio)
                    audio_files.append(adjusted_audio)
                    workflow_logger.info(f"Adjusted audio speed for block {i+1}")
                    self._check_audio_duration(adjusted_audio, workflow_logger, tts_target)
                    current_time = end_time
                else:
                    audio_files.append(temp_audio)
                    current_time = current_time + tts_duration

                # Add silence at the end if needed
                if current_time < end_time:
                    silence_duration = end_time - current_time
                    if silence_duration > 0.1:
                        silence_file = os.path.join(output_dir, f'silence_end_{i}.wav')
                        self._generate_silence(silence_duration, silence_file)
                        silence_file_fixed = os.path.join(output_dir, f'silence_end_{i}_fixed.wav')
                        self._ensure_wav_pcm(silence_file, silence_file_fixed)
                        os.replace(silence_file_fixed, silence_file)
                        audio_files.append(silence_file)
                        workflow_logger.info(f"Added end silence for block {i+1}")
                        self._check_audio_duration(silence_file, workflow_logger, silence_duration)
                    current_time = end_time

            # Collect debug audio files
            self._collect_debug_audio(audio_files, output_dir, subtitles)

            # Check all audio segments before concatenation
            workflow_logger.info("\nChecking all audio segments before concatenation:")
            self._check_audio_segments(audio_files, subtitles, workflow_logger)

            # Concatenate all audio files
            output_file = os.path.join(output_dir, 'final_audio.wav')
            self._concatenate_audio_files(audio_files, output_file)

            # Check final audio duration
            workflow_logger.info("\nChecking final audio duration:")
            final_duration = self._check_audio_duration(output_file, workflow_logger)
            expected_duration = self._time_to_seconds(subtitles[-1]['end'])
            if abs(final_duration - expected_duration) > 0.1:
                workflow_logger.warning(f"Final duration mismatch! Difference: {abs(final_duration - expected_duration):.2f}s")

            # Clean up intermediate files
            for file in audio_files:
                pass
                # os.remove(file)

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
