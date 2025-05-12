from typing import Dict, Any, List
try:
    from autotask.nodes import Node, register_node
    from autotask.api_keys import get_api_key
except:
    from .stub import Node, register_node, get_api_key
import os
import json
from datetime import datetime
from .make_wav import AudioToWavNode
import dashscope
from dashscope.audio.asr import Recognition
import subprocess
import asyncio
import re

# Initialize DashScope API key
DASHSCOPE_API_KEY = get_api_key(provider="dashscope.aliyuncs.com", key_name="DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

def detect_silence(input_file: str, silence_threshold: str = '-30dB', min_silence_duration: float = 0.5) -> list:
    """
    Use ffmpeg's silencedetect to find silence points in the audio.
    Returns a list of (silence_start, silence_end) tuples in seconds.
    """
    cmd = [
        'ffmpeg', '-i', input_file, '-af', f'silencedetect=noise={silence_threshold}:d={min_silence_duration}',
        '-f', 'null', '-'
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    output = result.stderr
    silence_starts = [float(m.group(1)) for m in re.finditer(r'silence_start: (\d+\.?\d*)', output)]
    silence_ends = [float(m.group(1)) for m in re.finditer(r'silence_end: (\d+\.?\d*)', output)]
    # Pair up starts and ends
    silences = []
    for i in range(min(len(silence_starts), len(silence_ends))):
        silences.append((silence_starts[i], silence_ends[i]))
    return silences

def get_split_points(silences: list, audio_duration: float, min_segment: float = 30, max_segment: float = 300) -> list:
    """
    Decide split points based on silence intervals and segment duration constraints.
    Returns a list of (start, end) tuples for each segment.
    """
    split_points = [0.0]
    for _, silence_end in silences:
        if silence_end - split_points[-1] >= min_segment:
            split_points.append(silence_end)
    if audio_duration - split_points[-1] > 1:
        split_points.append(audio_duration)
    # Enforce max_segment
    segments = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i+1]
        while end - start > max_segment:
            segments.append((start, start + max_segment))
            start += max_segment
        segments.append((start, end))
    return segments

def get_audio_duration(input_file: str) -> float:
    """
    Get the duration of the audio file in seconds using ffprobe.
    """
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def split_audio_by_silence(input_file: str, output_dir: str, min_segment: int = 30, max_segment: int = 300, silence_threshold: str = '-30dB', min_silence_duration: float = 0.5) -> list:
    """
    Split audio at silence points using ffmpeg's silencedetect.
    Returns a list of tuples containing (output_file_path, start_time, duration).
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_duration = get_audio_duration(input_file)
    silences = detect_silence(input_file, silence_threshold, min_silence_duration)
    segments = get_split_points(silences, audio_duration, min_segment, max_segment)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_files = []
    for idx, (start, end) in enumerate(segments):
        duration = end - start
        output_path = os.path.join(output_dir, f"{base_name}_segment_{idx+1:03d}_{start:.2f}_{duration:.2f}.wav")
        cmd = [
            'ffmpeg', '-y', '-i', input_file, '-ss', str(start), '-to', str(end),
            '-c', 'copy', output_path
        ]
        subprocess.run(cmd, check=True)
        output_files.append((output_path, start, duration))
    return output_files

def merge_adjacent_sentences(sentences, time_threshold=1000, text_overlap=5):
    """
    Merge adjacent sentences if their time overlaps or text overlaps.
    time_threshold: max allowed time overlap in ms
    text_overlap: max allowed text overlap in chars
    """
    if not sentences:
        return []
    merged = [sentences[0]]
    for curr in sentences[1:]:
        prev = merged[-1]
        # 时间重叠
        if curr.get('begin_time', 0) <= prev.get('end_time', 0) + time_threshold:
            # 文本重叠
            prev_text = prev.get('text', '')
            curr_text = curr.get('text', '')
            if prev_text and curr_text and prev_text[-text_overlap:] in curr_text[:text_overlap]:
                prev['end_time'] = max(prev.get('end_time', 0), curr.get('end_time', 0))
                prev['text'] += curr_text[text_overlap:]
                continue
        merged.append(curr)
    return merged

@register_node
class AudioRecognitionNode(Node):
    NAME = "Audio Recognition"
    DESCRIPTION = "Convert audio to text using Alibaba Cloud's speech recognition service"

    INPUTS = {
        "input_file": {
            "label": "Input Audio File",
            "description": "Path to the input audio file (supports various formats)",
            "type": "STRING",
            "required": True,
            "widget": "FILE"
        },
        "output_dir": {
            "label": "Output Directory",
            "description": "Directory to save the recognition results",
            "type": "STRING",
            "required": True,
            "widget": "DIR"
        },
        "model": {
            "label": "Recognition Model",
            "description": "Speech recognition model to use",
            "type": "STRING",
            "required": False,
            "default": "paraformer-realtime-v2"
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
        "segment_duration": {
            "label": "Segment Duration",
            "description": "Duration of each audio segment in seconds",
            "type": "INT",
            "required": False,
            "default": 300
        },
        "concurrency": {
            "label": "Concurrency",
            "description": "Number of concurrent audio recognition tasks",
            "type": "INT",
            "required": False,
            "default": 5
        }
    }

    OUTPUTS = {
        "text": {
            "label": "Recognized Text",
            "description": "The recognized text from the audio",
            "type": "STRING"
        },
        "json_file": {
            "label": "JSON Result File",
            "description": "Path to the detailed recognition results in JSON format",
            "type": "STRING"
        }
    }

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            input_file = node_inputs["input_file"]
            output_dir = node_inputs["output_dir"]
            model = node_inputs.get("model", "paraformer-realtime-v2")
            sample_rate = node_inputs.get("sample_rate", 16000)
            channels = node_inputs.get("channels", 1)
            bit_depth = node_inputs.get("bit_depth", 16)
            segment_duration = node_inputs.get("segment_duration", 300)
            concurrency = node_inputs.get("concurrency", 5)

            workflow_logger.info(f"Starting audio recognition for: {input_file}")

            # First convert audio to WAV format if needed
            input_ext = os.path.splitext(input_file)[1].lower()
            if input_ext != '.wav':
                wav_node = AudioToWavNode()
                wav_result = await wav_node.execute({
                    "input_file": input_file,
                    "output_dir": output_dir,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "bit_depth": bit_depth,
                    "overwrite": "true"
                }, workflow_logger)

                if not wav_result["success"]:
                    return wav_result

                wav_file = wav_result["output_file"]
                workflow_logger.info(f"Audio converted to WAV format: {wav_file}")
            else:
                wav_file = input_file
                workflow_logger.info(f"Input file is already in WAV format: {wav_file}")

            # Split audio into segments using silence detection
            workflow_logger.info(f"Splitting audio using silence detection")
            split_files = split_audio_by_silence(wav_file, output_dir, min_segment=30, max_segment=segment_duration)
            workflow_logger.info(f"Audio split into {len(split_files)} segments")

            # Process segments concurrently
            all_results = []
            semaphore = asyncio.Semaphore(concurrency)

            async def process_segment(segment_info: tuple, index: int):
                segment_file, start_time, duration = segment_info
                async with semaphore:
                    workflow_logger.info(f"Processing segment {index+1}/{len(split_files)}: {segment_file}")
                    
                    # Use the actual start time from the segment info
                    segment_start = start_time
                    
                    # Call Alibaba Cloud speech recognition API
                    recognition = Recognition(
                        model=model,
                        format='wav',
                        sample_rate=sample_rate,
                        callback=None,
                    )
                    
                    result = recognition.call(segment_file)
                    if result is None:
                        workflow_logger.warning(f"No recognition result for segment {index+1}")
                        return None
                        
                    sentence_list = result.get_sentence()
                    if sentence_list is None:
                        workflow_logger.warning(f"No sentences in segment {index+1}")
                        return None
                    
                    # Adjust timestamps based on actual segment start time
                    for sentence in sentence_list:
                        if 'begin_time' in sentence:
                            sentence['begin_time'] += segment_start * 1000  # Convert to milliseconds
                        if 'end_time' in sentence:
                            sentence['end_time'] += segment_start * 1000  # Convert to milliseconds
                        
                        # Adjust timestamps for words in the sentence
                        if 'words' in sentence:
                            for word in sentence['words']:
                                if 'begin_time' in word:
                                    word['begin_time'] += segment_start * 1000
                                if 'end_time' in word:
                                    word['end_time'] += segment_start * 1000
                    
                    # Save intermediate result for this segment
                    intermediate_json = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(segment_file))[0]}_recognition.json")
                    with open(intermediate_json, 'w', encoding='utf-8') as f:
                        json.dump(sentence_list, f, ensure_ascii=False, indent=2)
                    workflow_logger.info(f"Saved intermediate result for segment {index+1} to: {intermediate_json}")
                    
                    return sentence_list

            # Create tasks for all segments
            tasks = [process_segment(segment_info, i) for i, segment_info in enumerate(split_files)]
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            
            # Filter out None results and extend all_results
            for result in results:
                if result is not None:
                    all_results.extend(result)

            if not all_results:
                error_msg = "No recognition results from any segment"
                workflow_logger.error(error_msg)
                return {"success": False, "error_message": error_msg}

            # Sort results by begin_time to ensure correct order
            all_results.sort(key=lambda x: x.get('begin_time', 0))
            
            # Merge adjacent/overlapping or highly similar sentences
            all_results = merge_adjacent_sentences(all_results)

            # Combine all sentences into one text, preserving timestamps
            text = " ".join(sentence['text'] for sentence in all_results)
            
            # Save detailed results to JSON file
            input_filename = os.path.basename(input_file)
            name_without_ext = os.path.splitext(input_filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"{name_without_ext}_{timestamp}_recognition.json"
            json_file = os.path.join(output_dir, json_filename)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            workflow_logger.info(f"Recognition completed successfully. Results saved to: {json_file}")
            return {
                "success": True,
                "text": text,
                "json_file": json_file
            }

        except Exception as e:
            error_msg = f"Audio recognition failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {"success": False, "error_message": error_msg}


if __name__ == "__main__":
    # Setup basic logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test AudioRecognitionNode
    print("\nTesting AudioRecognitionNode:")
    node = AudioRecognitionNode()
    test_inputs = {
        "input_file": "test.mp3",
        "output_dir": "output",
        "model": "paraformer-realtime-v2",
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16
    }
    result = asyncio.run(node.execute(test_inputs, logger))
    print(f"Result: {result}")




