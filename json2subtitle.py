from typing import Dict, Any, List
try:
    from autotask.nodes import Node, register_node
except:
    from .stub import Node, register_node
import os
import json
from datetime import datetime, timedelta
import re


@register_node
class Json2SubtitleNode(Node):
    NAME = "Json2Subtitle"
    DESCRIPTION = "Convert speech recognition results to subtitle files"

    INPUTS = {
        "json_file": {
            "label": "Recognition Result File",
            "description": "Path to the JSON file containing speech recognition results",
            "type": "STRING",
            "required": True,
            "widget": "FILE"
        },
        "output_dir": {
            "label": "Output Directory",
            "description": "Directory to save the subtitle files",
            "type": "STRING",
            "required": True,
            "widget": "DIR"
        },
        "format": {
            "label": "Subtitle Format",
            "description": "Format of the subtitle file",
            "type": "COMBO",
            "required": False,
            "default": "srt",
            "options": ["srt", "vtt", "ass"]
        },
        "max_length": {
            "label": "Max Subtitle Length",
            "description": "Maximum number of characters per subtitle line",
            "type": "INT",
            "required": False,
            "default": 100
        }
    }

    OUTPUTS = {
        "subtitle_file": {
            "label": "Subtitle File",
            "description": "Path to the generated subtitle file",
            "type": "STRING"
        }
    }

    def _format_time(self, milliseconds: int) -> str:
        """Convert milliseconds to time format"""
        time = timedelta(milliseconds=milliseconds)
        hours = time.seconds // 3600
        minutes = (time.seconds % 3600) // 60
        seconds = time.seconds % 60
        milliseconds = time.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _generate_srt(self, sentences: List[Dict], output_file: str) -> None:
        """Generate SRT format subtitle file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(sentences, 1):
                start_time = self._format_time(sentence['begin_time'])
                end_time = self._format_time(sentence['end_time'])
                text = sentence['text'].strip()
                f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

    def _generate_vtt(self, sentences: List[Dict], output_file: str) -> None:
        """Generate VTT format subtitle file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for i, sentence in enumerate(sentences, 1):
                start_time = self._format_time(sentence['begin_time']).replace(',', '.')
                end_time = self._format_time(sentence['end_time']).replace(',', '.')
                text = sentence['text'].strip()
                f.write(f"{start_time} --> {end_time}\n{text}\n\n")

    def _generate_ass(self, sentences: List[Dict], output_file: str) -> None:
        """Generate ASS format subtitle file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write ASS header
            f.write("""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,54,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
            # Write subtitles
            for sentence in sentences:
                start_time = self._format_time(sentence['begin_time']).replace(',', '.')
                end_time = self._format_time(sentence['end_time']).replace(',', '.')
                text = sentence['text'].strip()
                f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")

    def is_chinese(self, text):
        return any('\u4e00' <= c <= '\u9fff' for c in text)

    def split_long_sentence(self, sentence, max_length=100):
        text = sentence['text']
        words = sentence.get('words', None)
        result = []
        # 1. 先按标点切分为自然句
        segs = []
        last = 0
        for m in re.finditer(r'[。！？!?.,;:]', text):
            end = m.end()
            segs.append((last, end))
            last = end
        if last < len(text):
            segs.append((last, len(text)))
        # 2. 贪心合并自然句，直到max_length
        buf = ''
        buf_start = segs[0][0] if segs else 0
        buf_end = segs[0][1] if segs else 0
        buf_words = []
        char_pos = 0
        for i, (start, end) in enumerate(segs):
            seg = text[start:end]
            # 计算合并后长度
            if len(buf) + len(seg) <= max_length or not buf:
                if not buf:
                    buf_start = start
                buf += seg
                buf_end = end
            else:
                # 输出一条
                s = dict(sentence)
                s['text'] = text[buf_start:buf_end]
                # 英文用words时间
                if words and not self.is_chinese(text):
                    seg_words = []
                    char_pos = 0
                    for w in words:
                        w_text = w['text']
                        idx = text.find(w_text, char_pos)
                        if idx == -1:
                            idx = char_pos
                        w_start = idx
                        w_end = idx + len(w_text)
                        char_pos = w_end
                        if w_start >= buf_start and w_end <= buf_end:
                            seg_words.append(w)
                    if seg_words:
                        s['begin_time'] = seg_words[0].get('begin_time', sentence['begin_time'])
                        s['end_time'] = seg_words[-1].get('end_time', sentence['end_time'])
                else:
                    # For Chinese text, calculate proportional timestamps
                    total_chars = len(text)
                    start_ratio = buf_start / total_chars
                    end_ratio = buf_end / total_chars
                    duration = sentence['end_time'] - sentence['begin_time']
                    s['begin_time'] = sentence['begin_time'] + int(duration * start_ratio)
                    s['end_time'] = sentence['begin_time'] + int(duration * end_ratio)
                result.append(s)
                # 新起一条
                buf = seg
                buf_start = start
                buf_end = end
        # 最后一条
        if buf:
            s = dict(sentence)
            s['text'] = text[buf_start:buf_end]
            if words and not self.is_chinese(text):
                seg_words = []
                char_pos = 0
                for w in words:
                    w_text = w['text']
                    idx = text.find(w_text, char_pos)
                    if idx == -1:
                        idx = char_pos
                    w_start = idx
                    w_end = idx + len(w_text)
                    char_pos = w_end
                    if w_start >= buf_start and w_end <= buf_end:
                        seg_words.append(w)
                if seg_words:
                    s['begin_time'] = seg_words[0].get('begin_time', sentence['begin_time'])
                    s['end_time'] = seg_words[-1].get('end_time', sentence['end_time'])
            else:
                # For Chinese text, calculate proportional timestamps
                total_chars = len(text)
                start_ratio = buf_start / total_chars
                end_ratio = buf_end / total_chars
                duration = sentence['end_time'] - sentence['begin_time']
                s['begin_time'] = sentence['begin_time'] + int(duration * start_ratio)
                s['end_time'] = sentence['begin_time'] + int(duration * end_ratio)
            result.append(s)
        return result

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            json_file = node_inputs["json_file"]
            output_dir = node_inputs["output_dir"]
            format_type = node_inputs.get("format", "srt").lower()

            workflow_logger.info(f"Starting subtitle generation for: {json_file}")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Read recognition results
            with open(json_file, 'r', encoding='utf-8') as f:
                sentences = json.load(f)

            # Split long sentences for better subtitle readability
            max_length = int(node_inputs.get("max_length", 50))
            split_sentences = []
            for s in sentences:
                split_sentences.extend(self.split_long_sentence(s, max_length=max_length))

            # Prepare output filename
            input_filename = os.path.basename(json_file)
            name_without_ext = os.path.splitext(input_filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{name_without_ext}_{timestamp}.{format_type}"
            output_file = os.path.join(output_dir, output_filename)

            # Generate subtitle file
            if format_type == "srt":
                self._generate_srt(split_sentences, output_file)
            elif format_type == "vtt":
                self._generate_vtt(split_sentences, output_file)
            elif format_type == "ass":
                self._generate_ass(split_sentences, output_file)
            else:
                error_msg = f"Unsupported subtitle format: {format_type}"
                workflow_logger.error(error_msg)
                return {"success": False, "error_message": error_msg}

            workflow_logger.info(f"Subtitle file generated successfully: {output_file}")
            return {
                "success": True,
                "subtitle_file": output_file
            }

        except Exception as e:
            error_msg = f"Subtitle generation failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {"success": False, "error_message": error_msg}


if __name__ == "__main__":
    # Setup basic logging
    import logging
    import asyncio
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test SubtitleGeneratorNode
    print("\nTesting SubtitleGeneratorNode:")
    node = SubtitleGeneratorNode()
    test_inputs = {
        "json_file": "result.json",
        "output_dir": "output",
        "format": "srt"
    }
    result = asyncio.run(node.execute(test_inputs, logger))
    print(f"Result: {result}")
