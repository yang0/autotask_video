try:
    from autotask.nodes import Node, GeneratorNode, register_node
except ImportError:
    from stub import Node, GeneratorNode, register_node

from typing import Dict, Any, List, AsyncGenerator
import re
import os
from datetime import datetime


class SubtitleFormat:
    SRT = "srt"
    ASS = "ass"
    SSA = "ssa"
    VTT = "vtt"


@register_node
class SubtitleSplitter(GeneratorNode):
    NAME = "Subtitle Splitter"
    DESCRIPTION = "Split subtitle files into timestamp blocks (supports .srt, .ass, .ssa, .vtt) and output them in configurable batches"

    INPUTS = {
        "subtitle_file": {
            "label": "Subtitle File",
            "description": "Input subtitle file (.srt, .ass, .ssa, .vtt formats supported)",
            "type": "STRING",
            "required": True,
            "widget": "FILE"
        },
        "lines_per_output": {
            "label": "Timestamp Blocks Per Output",
            "description": "Number of timestamp blocks to output at once (default: 10)",
            "type": "INT",
            "required": False,
            "default": 10
        }
    }

    OUTPUTS = {
        "extracted_text": {
            "label": "Extracted Text",
            "description": "Text content extracted from subtitles with timestamps (10 timestamp blocks per output)",
            "type": "STRING"
        }
    }

    def _detect_format(self, file_path: str) -> str:
        """Detect subtitle format from file extension"""
        ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        if ext in [SubtitleFormat.SRT, SubtitleFormat.ASS, SubtitleFormat.SSA, SubtitleFormat.VTT]:
            return ext
        raise ValueError(f"Unsupported subtitle format: {ext}")

    def _extract_srt(self, content: str) -> List[str]:
        """Extract text from SRT format"""
        text_blocks = []
        pattern = r'(\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n)(.*?)(?=\n\n|\Z)'
        matches = re.finditer(pattern, content, re.DOTALL)
        for match in matches:
            timestamp = match.group(1).strip()
            text = match.group(2).strip()
            if text:
                text_blocks.append(f"{timestamp}{text}")
        return text_blocks

    def _extract_ass_ssa(self, content: str) -> List[str]:
        """Extract text from ASS/SSA format"""
        text_blocks = []
        # Skip header section
        content = content.split('[Events]')[-1]
        # Extract text from Dialogue lines
        pattern = r'Dialogue:.*?,(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?),(.*?)$'
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            start_time = match.group(1)
            end_time = match.group(2)
            text = match.group(10).strip()
            if text:
                text_blocks.append(f"{start_time} --> {end_time}\n{text}")
        return text_blocks

    def _extract_vtt(self, content: str) -> List[str]:
        """Extract text from VTT format"""
        text_blocks = []
        # Skip header
        content = content.split('\n\n', 1)[-1]
        # Extract text blocks
        pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\n)(.*?)(?=\n\n|\Z)'
        matches = re.finditer(pattern, content, re.DOTALL)
        for match in matches:
            timestamp = match.group(1).strip()
            text = match.group(2).strip()
            if text:
                text_blocks.append(f"{timestamp}{text}")
        return text_blocks

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            subtitle_path = node_inputs["subtitle_file"]
            blocks_per_output = node_inputs.get("lines_per_output", 10)
            workflow_logger.info(f"Processing subtitle file: {subtitle_path}")

            # Detect format
            format_type = self._detect_format(subtitle_path)
            workflow_logger.info(f"Detected subtitle format: {format_type}")

            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract text based on format
            if format_type == SubtitleFormat.SRT:
                text_blocks = self._extract_srt(content)
            elif format_type in [SubtitleFormat.ASS, SubtitleFormat.SSA]:
                text_blocks = self._extract_ass_ssa(content)
            elif format_type == SubtitleFormat.VTT:
                text_blocks = self._extract_vtt(content)
            else:
                raise ValueError(f"Unsupported subtitle format: {format_type}")

            # Output text blocks in chunks of blocks_per_output
            for i in range(0, len(text_blocks), blocks_per_output):
                chunk = text_blocks[i:i+blocks_per_output]
                # Join with double newlines to separate blocks
                extracted_text = '\n\n'.join(chunk)
                workflow_logger.info(f"Outputting timestamp blocks {i+1} to {min(i+blocks_per_output, len(text_blocks))}")
                yield {
                    "success": True,
                    "extracted_text": extracted_text
                }

            workflow_logger.info(f"Successfully extracted all text blocks from subtitles")

        except Exception as e:
            error_msg = f"Failed to extract subtitle text: {str(e)}"
            workflow_logger.error(error_msg)
            yield {
                "success": False,
                "error_message": error_msg
            }


@register_node
class SubtitleGenerator(Node):
    NAME = "Subtitle Generator"
    DESCRIPTION = "Generate new subtitle file with translated text (supports .srt, .ass, .ssa, .vtt) and bilingual text format"

    INPUTS = {
        "original_subtitle_file": {
            "label": "Original Subtitle File",
            "description": "Original subtitle file for reference (.srt, .ass, .ssa, .vtt formats supported)",
            "type": "STRING",
            "required": True,
            "widget": "FILE"
        },
        "translated_text": {
            "label": "Translated Text",
            "description": "Translated text content with timestamps",
            "type": "STRING",
            "required": True
        },
        "output_dir": {
            "label": "Output Directory",
            "description": "Directory to save the new subtitle file",
            "type": "STRING",
            "required": True,
            "widget": "DIR"
        },
        "keep_original": {
            "label": "Keep Original Text",
            "description": "Keep original text below translation when enabled",
            "type": "BOOLEAN",
            "required": False,
            "default": False,
            "widget": "SWITCH"
        }
    }

    OUTPUTS = {
        "output_file": {
            "label": "Generated Subtitle File",
            "description": "Path to the generated subtitle file",
            "type": "STRING"
        }
    }

    def _detect_format(self, file_path: str) -> str:
        """Detect subtitle format from file extension"""
        ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        if ext in [SubtitleFormat.SRT, SubtitleFormat.ASS, SubtitleFormat.SSA, SubtitleFormat.VTT]:
            return ext
        raise ValueError(f"Unsupported subtitle format: {ext}")

    def _process_translated_text(self, text: str) -> List[tuple]:
        """Process translated text into a list of (timestamp, text) tuples"""
        translations = []
        current_timestamp = None
        current_text = []
        
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains timestamp
            if '-->' in line:
                if current_timestamp and current_text:
                    # Clean up timestamp by removing any newlines
                    clean_timestamp = current_timestamp.strip()
                    translations.append((clean_timestamp, ' '.join(current_text)))
                current_timestamp = line
                current_text = []
            else:
                # Remove HTML entities and clean up text
                clean_text = line.replace('&nbsp;', ' ').strip()
                if clean_text:
                    current_text.append(clean_text)
        
        # Add the last block
        if current_timestamp and current_text:
            # Clean up timestamp by removing any newlines
            clean_timestamp = current_timestamp.strip()
            translations.append((clean_timestamp, ' '.join(current_text)))
            
        return translations

    def _generate_vtt(self, original_content: str, translations: List[tuple], keep_original: bool) -> str:
        """Generate VTT format subtitle"""
        # Keep VTT header and metadata
        header_end = original_content.find('\n\n') + 2
        header = original_content[:header_end]
        
        # Create translation lookup dictionary
        translation_dict = {ts.strip(): text for ts, text in translations}
        
        # Extract timestamps and original text
        pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3})\n(.*?)(?=\n\n|\Z)'
        matches = list(re.finditer(pattern, original_content[header_end:], re.DOTALL))
        
        # Generate new content
        new_content = [header]
        
        for match in matches:
            timestamp = match.group(1)
            original_text = match.group(2).strip()
            # Get translation if available
            if timestamp in translation_dict:
                translation = translation_dict[timestamp]
                if keep_original:
                    new_content.append(f"{timestamp}\n{translation}\n{original_text}\n\n")
                else:
                    new_content.append(f"{timestamp}\n{translation}\n\n")
            else:
                new_content.append(f"{timestamp}\n{original_text}\n\n")
        
        return ''.join(new_content)

    def _generate_srt(self, original_content: str, translations: List[tuple], keep_original: bool) -> str:
        """Generate SRT format subtitle"""
        new_content = []
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)'
        matches = list(re.finditer(pattern, original_content, re.DOTALL))
        
        # Create translation lookup dictionary
        translation_dict = {ts.strip(): text for ts, text in translations}
        
        for match in matches:
            index = match.group(1)
            timestamp = match.group(2)
            original_text = match.group(3).strip()
            # Get translation if available
            if timestamp in translation_dict:
                translation = translation_dict[timestamp]
                if keep_original:
                    new_content.append(f"{index}\n{timestamp}\n{translation}\n{original_text}\n\n")
                else:
                    new_content.append(f"{index}\n{timestamp}\n{translation}\n\n")
            else:
                new_content.append(f"{index}\n{timestamp}\n{original_text}\n\n")
        
        return ''.join(new_content)

    def _generate_ass_ssa(self, original_content: str, translations: List[tuple], keep_original: bool) -> str:
        """Generate ASS/SSA format subtitle"""
        # Split content into header and events
        parts = original_content.split('[Events]')
        if len(parts) != 2:
            raise ValueError("Invalid ASS/SSA format: missing [Events] section")
        
        header = parts[0] + '[Events]'
        events = parts[1]
        
        # Extract format line and dialogue lines
        format_line = re.search(r'Format:.*?\n', events).group(0)
        dialogue_lines = re.findall(r'Dialogue:.*?\n', events)
        
        # Create translation lookup dictionary
        translation_dict = {ts.strip(): text for ts, text in translations}
        
        # Generate new dialogue lines with translated text
        new_dialogues = []
        for line in dialogue_lines:
            # Split dialogue line and get text
            parts = line.split(',')
            if len(parts) >= 10:
                timestamp = f"{parts[1]} --> {parts[2]}"
                original_text = parts[9].strip()
                # Get translation if available
                if timestamp in translation_dict:
                    translation = translation_dict[timestamp]
                    if keep_original:
                        parts[9] = f"{translation}\n{original_text}"
                    else:
                        parts[9] = translation
                new_dialogues.append(','.join(parts))
        
        return header + format_line + ''.join(new_dialogues)

    def _generate_output_filename(self, original_path: str) -> str:
        """Generate output filename with date suffix"""
        # Get original filename without extension
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        # Get original extension
        ext = os.path.splitext(original_path)[1]
        # Generate date suffix
        date_suffix = datetime.now().strftime("%Y%m%d")
        # Combine to create new filename
        return f"{base_name}_{date_suffix}{ext}"

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            original_path = node_inputs["original_subtitle_file"]
            translated_text = node_inputs["translated_text"]
            output_dir = node_inputs["output_dir"]
            keep_original = node_inputs.get("keep_original", False)
            
            # Generate output filename
            output_filename = self._generate_output_filename(original_path)
            output_path = os.path.join(output_dir, output_filename)
            
            workflow_logger.info(f"Generating new subtitle file from: {original_path}")
            workflow_logger.info(f"Output will be saved as: {output_path}")

            # Detect format
            format_type = self._detect_format(original_path)
            workflow_logger.info(f"Detected subtitle format: {format_type}")

            # Read original subtitle file
            with open(original_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Process translated text
            translations = self._process_translated_text(translated_text)
            
            # Generate new subtitle content based on format
            if format_type == SubtitleFormat.SRT:
                new_content = self._generate_srt(original_content, translations, keep_original)
            elif format_type in [SubtitleFormat.ASS, SubtitleFormat.SSA]:
                new_content = self._generate_ass_ssa(original_content, translations, keep_original)
            elif format_type == SubtitleFormat.VTT:
                new_content = self._generate_vtt(original_content, translations, keep_original)
            else:
                raise ValueError(f"Unsupported subtitle format: {format_type}")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Write new subtitle file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            workflow_logger.info(f"Successfully generated new subtitle file at: {output_path}")
            
            return {
                "success": True,
                "output_file": output_path
            }

        except Exception as e:
            error_msg = f"Failed to generate subtitle file: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg
            }
