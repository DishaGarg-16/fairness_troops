from fpdf import FPDF
import base64
import tempfile
import os
from datetime import datetime

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Fairness Troops - Audit Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

class ReportGenerator:
    def __init__(self, metrics: dict, config: dict):
        self.metrics = metrics
        self.config = config
        self.pdf = PDFReport()
        self.pdf.add_page()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def _add_section_title(self, title):
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.set_fill_color(200, 220, 255)
        self.pdf.cell(0, 10, title, 0, 1, 'L', fill=True)
        self.pdf.ln(5)

    def _add_key_value(self, key, value):
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.cell(60, 8, f"{key}:", 0)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.cell(0, 8, str(value), 0, 1)

    def generate(self, plot_images: dict = None) -> bytes:
        # 1. Configuration Info
        self._add_section_title("Audit Configuration")
        self._add_key_value("Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        for k, v in self.config.items():
            self._add_key_value(k.replace('_', ' ').title(), v)
        self.pdf.ln(10)

        # 2. Metrics
        self._add_section_title("Fairness Metrics")
        for k, v in self.metrics.items():
            # Format float if possible
             val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
             self._add_key_value(k.replace('_', ' ').title(), val_str)
        self.pdf.ln(10)

        # 3. Plots
        if plot_images:
            self._add_section_title("Visualizations")
            for title, b64_str in plot_images.items():
                if not b64_str:
                    continue
                    
                self.pdf.add_page() # New page for plots
                self.pdf.set_font('Arial', 'B', 11)
                self.pdf.cell(0, 10, title, 0, 1, 'C')
                
                try:
                    # Save temp file for FPDF to read
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        img_data = base64.b64decode(b64_str)
                        tmp.write(img_data)
                        tmp_path = tmp.name
                    
                    # Add image (width 180mm)
                    self.pdf.image(tmp_path, x=15, w=180)
                    os.unlink(tmp_path)
                    self.pdf.ln(5)
                except Exception as e:
                    self.pdf.cell(0, 10, f"Error processing image: {e}", 0, 1)

        # Return Bytes
        try:
             # output() with dest='S' returns input string (latin1), we need bytes
             # recent FPDF versions prefer:
             return self.pdf.output(dest='S').encode('latin-1')
        except Exception:
             # Fallback or different version
             # For some versions, output() returns str.
             return self.pdf.output(dest='S').encode('latin-1')
