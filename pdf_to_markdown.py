import os
import sys
import logging
import base64
import requests
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import tempfile
import io
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

import pdfplumber
import fitz
from PIL import Image
import markdownify

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SILICONFLOW_API_KEY = '****'
SILICONFLOW_API_URL = 'https://api.siliconflow.cn/v1/chat/completions'

class PDFElement:
    def __init__(self, element_type: str, content: str, x0: float, y0: float, x1: float, y1: float, page_num: int):
        self.element_type = element_type
        self.content = content
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.page_num = page_num
        self.center_y = (y0 + y1) / 2
        self.center_x = (x0 + x1) / 2

    def __repr__(self):
        return f"PDFElement(type={self.element_type}, y={self.y0:.2f}, content={self.content[:50]})"

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ocr_image(image_path: str, max_retries: int = 3) -> Optional[str]:
    if not SILICONFLOW_API_KEY:
        logger.warning("SILICONFLOW_API_KEY not set, skipping OCR")
        return None
    
    # 根据文件扩展名设置正确的 MIME 类型
    ext = os.path.splitext(image_path)[1].lower().lstrip('.')
    mime_type_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'bmp': 'image/bmp'
    }
    mime_type = mime_type_map.get(ext, 'image/jpeg')  # 默认使用 jpeg
    
    headers = {
        'Authorization': f'Bearer {SILICONFLOW_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None
    
    payload = {
        'model': 'zai-org/GLM-4.5V',
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:{mime_type};base64,{base64_image}'
                        }
                    },
                    {
                        'type': 'text',
                        'text': '梳理出图片中的内容'
                    }
                ]
            }
        ],
        'max_tokens': 8192,
        'temperature': 0.6,
        'top_p': 0.95,
        'top_k': 20,
        'enable_thinking': False,
        'thinking_budget': 4096
    }
    
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = requests.post(SILICONFLOW_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            extracted_text = result['choices'][0]['message']['content']
            
            logger.info(f"Successfully OCR'd image: {image_path}")
            return extracted_text
            
        except requests.exceptions.RequestException as e:
            last_exception = e
            import json
            logger.warning(f"API request failed for {image_path} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # 指数退避
            else:
                logger.error(f"API request failed after {max_retries} attempts for {image_path}")
                logger.error(f"Request body: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse API response for {image_path}: {e}")
            return None
        except Exception as e:
            import json
            logger.error(f"Unexpected error during OCR for {image_path}: {e}")
            logger.error(f"Request body: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            return None
    
    return None

def extract_text_blocks(page, page_num: int, table_bboxes: List[Tuple[float, float, float, float]] = None) -> List[PDFElement]:
    elements = []
    
    # 获取表格区域，用于过滤掉表格内的文本
    table_bboxes = table_bboxes or []
    
    # 使用 extract_words 获取合并的文本块
    words = page.extract_words(
        keep_blank_chars=True,
        x_tolerance=3,
        y_tolerance=3
    )
    
    # 过滤掉在表格区域内的单词
    filtered_words = []
    for word in words:
        word_center_x = (word['x0'] + word['x1']) / 2
        word_center_y = (word['top'] + word['bottom']) / 2
        
        # 检查单词是否在表格区域内
        in_table = False
        for bbox in table_bboxes:
            if (bbox[0] <= word_center_x <= bbox[2] and 
                bbox[1] <= word_center_y <= bbox[3]):
                in_table = True
                break
        
        if not in_table:
            filtered_words.append(word)
    
    # 按行分组
    lines = {}
    for word in filtered_words:
        y_key = round(word['top'] / 5) * 5  # 按 y 坐标分组，容差 5
        if y_key not in lines:
            lines[y_key] = []
        lines[y_key].append(word)
    
    # 处理每一行
    for y_key in sorted(lines.keys()):
        line_words = sorted(lines[y_key], key=lambda w: w['x0'])
        
        if not line_words:
            continue
        
        # 合并文本
        text = ' '.join(w['text'] for w in line_words)
        if not text.strip():
            continue
        
        # 判断是否为标题（基于字体大小）
        font_sizes = [w.get('size', 12) for w in line_words if 'size' in w]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        
        # 检测粗体
        fontnames = [w.get('fontname', '') for w in line_words if 'fontname' in w]
        is_bold = any('bold' in fn.lower() for fn in fontnames)
        
        element_type = 'heading' if avg_font_size >= 14 or is_bold else 'text'
        
        # 计算边界框
        x0 = min(w['x0'] for w in line_words)
        y0 = min(w['top'] for w in line_words)
        x1 = max(w['x1'] for w in line_words)
        y1 = max(w['bottom'] for w in line_words)
        
        elements.append(PDFElement(
            element_type=element_type,
            content=text,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            page_num=page_num
        ))
    
    return elements

def extract_tables(page, page_num: int) -> Tuple[List[PDFElement], List[PDFElement]]:
    table_elements = []
    failed_tables = []
    
    tables = page.extract_tables()
    
    if tables:
        for table_idx, table in enumerate(tables):
            if table and len(table) > 0:
                markdown_table = convert_table_to_markdown(table)
                
                try:
                    bbox = page.find_tables()[table_idx].bbox
                    table_elements.append(PDFElement(
                        element_type='table',
                        content=markdown_table,
                        x0=bbox[0],
                        y0=bbox[1],
                        x1=bbox[2],
                        y1=bbox[3],
                        page_num=page_num
                    ))
                    logger.info(f"Successfully extracted table {table_idx} on page {page_num}")
                except Exception as e:
                    logger.warning(f"Could not get bbox for table {table_idx} on page {page_num}: {e}")
                    failed_tables.append(PDFElement(
                        element_type='failed_table',
                        content='',
                        x0=0,
                        y0=0,
                        x1=page.width,
                        y1=page.height,
                        page_num=page_num
                    ))
    else:
        logger.info(f"No tables found on page {page_num}")
    
    return table_elements, failed_tables

def convert_table_to_markdown(table: List[List[str]]) -> str:
    if not table or not table[0]:
        return ''
    
    max_cols = max(len(row) for row in table)
    normalized_table = []
    
    for row in table:
        normalized_row = [str(cell) if cell is not None else '' for cell in row]
        while len(normalized_row) < max_cols:
            normalized_row.append('')
        normalized_table.append(normalized_row)
    
    header = normalized_table[0]
    separator = ['---'] * max_cols
    rows = normalized_table[1:]
    
    markdown_lines = []
    markdown_lines.append('| ' + ' | '.join(header) + ' |')
    markdown_lines.append('| ' + ' | '.join(separator) + ' |')
    
    for row in rows:
        markdown_lines.append('| ' + ' | '.join(row) + ' |')
    
    return '\n'.join(markdown_lines)

def extract_images(page: fitz.Page, page_num: int, temp_dir: str) -> List[PDFElement]:
    image_elements = []
    image_list = page.get_images(full=True)
    
    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        image_path = os.path.join(temp_dir, f'page_{page_num}_img_{img_index}.{image_ext}')
        
        with open(image_path, 'wb') as img_file:
            img_file.write(image_bytes)
        
        try:
            img_rects = page.get_image_rects(xref)
            if img_rects:
                rect = img_rects[0]
                image_elements.append(PDFElement(
                    element_type='image',
                    content=image_path,
                    x0=rect.x0,
                    y0=rect.y0,
                    x1=rect.x1,
                    y1=rect.y1,
                    page_num=page_num
                ))
                logger.info(f"Extracted image {img_index} on page {page_num}")
        except Exception as e:
            logger.warning(f"Could not get rect for image {img_index} on page {page_num}: {e}")
            image_elements.append(PDFElement(
                element_type='image',
                content=image_path,
                x0=0,
                y0=0,
                x1=page.rect.width,
                y1=page.rect.height,
                page_num=page_num
            ))
    
    return image_elements

def sort_elements(elements: List[PDFElement]) -> List[PDFElement]:
    return sorted(elements, key=lambda x: (x.page_num, x.center_y, x.center_x))

def determine_heading_level(element: PDFElement, base_font_size: float = 12) -> int:
    if element.element_type == 'heading':
        return 1
    return 0

def format_element_as_markdown(element: PDFElement) -> str:
    if element.element_type == 'text':
        return element.content + '\n'
    
    elif element.element_type == 'heading':
        level = determine_heading_level(element)
        heading_prefix = '#' * (level + 1)
        return f'{heading_prefix} {element.content.strip()}\n\n'
    
    elif element.element_type == 'table':
        return '\n' + element.content + '\n\n'
    
    elif element.element_type == 'image':
        # 图片不再显示本地路径，而是使用占位符
        return f'\n> 📷 [图片内容见下方 OCR 结果]\n\n'
    
    elif element.element_type == 'failed_table':
        return '\n' + element.content + '\n\n'
    
    elif element.element_type == 'ocr_result':
        # 清理特殊标记，格式化 OCR 结果
        content = element.content
        content = content.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
        content = content.strip()
        if content:
            return f'\n> **图片内容：**\n>\n> {content.replace(chr(10), chr(10) + "> ")}\n\n'
        return ''
    
    return ''

def convert_pdf_to_markdown(pdf_path: str, output_path: Optional[str] = None) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if output_path is None:
        output_path = Path(pdf_path).with_suffix('.md')
    
    logger.info(f"Starting conversion: {pdf_path} -> {output_path}")
    
    all_elements = []
    temp_dir = tempfile.mkdtemp(prefix='pdf_conversion_')
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"PDF has {len(pdf.pages)} pages")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                logger.info(f"Processing page {page_num}")
                
                text_elements = extract_text_blocks(page, page_num)
                all_elements.extend(text_elements)
                
                table_elements, failed_tables = extract_tables(page, page_num)
                all_elements.extend(table_elements)
                all_elements.extend(failed_tables)
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            logger.info(f"Extracting images from page {page_num + 1}")
            
            image_elements = extract_images(page, page_num + 1, temp_dir)
            all_elements.extend(image_elements)
        
        doc.close()
        
        sorted_elements = sort_elements(all_elements)
        
        # 去重：移除内容完全相同的元素
        unique_elements = []
        seen_contents = set()
        for element in sorted_elements:
            content_hash = hash(element.content.strip())
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_elements.append(element)
        
        # 进一步去重：移除被表格包含的文本内容
        filtered_elements = []
        table_bboxes = []
        for elem in unique_elements:
            if elem.element_type == 'table':
                table_bboxes.append((elem.x0, elem.y0, elem.x1, elem.y1, elem.page_num))
        
        for elem in unique_elements:
            if elem.element_type == 'text':
                # 检查文本是否被表格包含
                elem_center_x = (elem.x0 + elem.x1) / 2
                elem_center_y = (elem.y0 + elem.y1) / 2
                
                in_table = False
                for bbox in table_bboxes:
                    if (bbox[4] == elem.page_num and
                        bbox[0] <= elem_center_x <= bbox[2] and 
                        bbox[1] <= elem_center_y <= bbox[3]):
                        in_table = True
                        break
                
                if not in_table:
                    filtered_elements.append(elem)
            else:
                filtered_elements.append(elem)
        
        markdown_content = []
        markdown_content.append(f"# Document: {Path(pdf_path).name}\n\n")
        
        for element in filtered_elements:
            if element.element_type == 'failed_table':
                logger.info(f"Performing OCR on failed table on page {element.page_num}")
                ocr_text = ocr_image(element.content)
                if ocr_text:
                    element.content = f"```\n{ocr_text}\n```"
                    element.element_type = 'ocr_result'
            
            elif element.element_type == 'image':
                logger.info(f"Performing OCR on image on page {element.page_num}")
                ocr_text = ocr_image(element.content)
                if ocr_text:
                    element.content = ocr_text
                    element.element_type = 'ocr_result'
                else:
                    # OCR 失败时跳过该图片
                    continue
            
            formatted = format_element_as_markdown(element)
            if formatted:  # 只添加非空内容
                markdown_content.append(formatted)
        
        final_markdown = ''.join(markdown_content)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_markdown)
        
        logger.info(f"Conversion completed successfully: {output_path}")
        
        return final_markdown
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise
    finally:
        import shutil
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up temporary directory: {e}")

class PDFToMarkdownApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF 转 Markdown 工具")
        self.root.geometry("600x400")
        self.root.resizable(True, True)
        
        # 设置字体和颜色
        self.font = ("微软雅黑", 10)
        self.bg_color = "#f0f0f0"
        self.button_color = "#4CAF50"
        self.text_color = "#333333"
        
        # 创建主框架
        self.main_frame = tk.Frame(root, bg=self.bg_color, padx=20, pady=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 输入文件选择
        self.input_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.input_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(self.input_frame, text="选择 PDF 文件:", font=self.font, bg=self.bg_color, fg=self.text_color).pack(side=tk.LEFT, padx=5)
        
        self.input_entry = tk.Entry(self.input_frame, width=50, font=self.font)
        self.input_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.browse_input_button = tk.Button(self.input_frame, text="浏览", font=self.font, bg=self.button_color, fg="white", command=self.browse_input)
        self.browse_input_button.pack(side=tk.RIGHT, padx=5)
        
        # 输出目录选择
        self.output_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.output_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(self.output_frame, text="选择输出目录:", font=self.font, bg=self.bg_color, fg=self.text_color).pack(side=tk.LEFT, padx=5)
        
        self.output_entry = tk.Entry(self.output_frame, width=50, font=self.font)
        self.output_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.browse_output_button = tk.Button(self.output_frame, text="浏览", font=self.font, bg=self.button_color, fg="white", command=self.browse_output)
        self.browse_output_button.pack(side=tk.RIGHT, padx=5)
        
        # 转换按钮
        self.convert_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.convert_frame.pack(fill=tk.X, pady=20)
        
        self.convert_button = tk.Button(self.convert_frame, text="开始转换", font=("微软雅黑", 12, "bold"), bg=self.button_color, fg="white", command=self.convert)
        self.convert_button.pack(fill=tk.X, padx=50, pady=10)
        
        # 状态显示
        self.status_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.status_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = tk.Label(self.status_frame, text="就绪", font=self.font, bg=self.bg_color, fg=self.text_color)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
    def browse_input(self):
        file_path = filedialog.askopenfilename(
            title="选择 PDF 文件",
            filetypes=[("PDF 文件", "*.pdf"), ("所有文件", "*.*")]
        )
        if file_path:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, file_path)
    
    def browse_output(self):
        directory = filedialog.askdirectory(
            title="选择输出目录"
        )
        if directory:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, directory)
    
    def convert(self):
        input_pdf = self.input_entry.get().strip()
        output_dir = self.output_entry.get().strip()
        
        if not input_pdf:
            messagebox.showerror("错误", "请选择 PDF 文件")
            return
        
        if not output_dir:
            messagebox.showerror("错误", "请选择输出目录")
            return
        
        if not os.path.exists(input_pdf):
            messagebox.showerror("错误", f"PDF 文件不存在: {input_pdf}")
            return
        
        if not os.path.exists(output_dir):
            messagebox.showerror("错误", f"输出目录不存在: {output_dir}")
            return
        
        try:
            self.status_label.config(text="转换中...")
            self.root.update()
            
            # 生成带时间戳的输出文件名
            pdf_name = Path(input_pdf).stem
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            output_md = os.path.join(output_dir, f"{pdf_name}_{timestamp}.md")
            
            convert_pdf_to_markdown(input_pdf, output_md)
            
            self.status_label.config(text="转换完成")
            messagebox.showinfo("成功", f"转换完成！\n输出文件: {output_md}")
        except Exception as e:
            self.status_label.config(text="转换失败")
            messagebox.showerror("错误", f"转换失败: {e}")

def main():
    if len(sys.argv) < 2:
        # 没有命令行参数，启动 GUI
        root = tk.Tk()
        app = PDFToMarkdownApp(root)
        root.mainloop()
    else:
        # 命令行模式
        input_pdf = sys.argv[1]
        output_md = sys.argv[2] if len(sys.argv) > 2 else None
        
        try:
            convert_pdf_to_markdown(input_pdf, output_md)
            print(f"\n✓ Conversion completed successfully!")
            print(f"Output: {output_md if output_md else Path(input_pdf).with_suffix('.md')}")
        except Exception as e:
            print(f"\n✗ Conversion failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
