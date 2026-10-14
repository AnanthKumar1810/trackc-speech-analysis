from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from xml.sax.saxutils import escape
r=Path(__file__).resolve().parents[1]
text=(r/'TECHNICAL_REPORT.md').read_text(encoding='utf-8')
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8,textColor=colors.HexColor('#23334a')));styles['Title'].fontSize=22;styles['Title'].leading=27;styles['Title'].textColor=colors.HexColor('#123d51');styles['Heading2'].fontSize=12;styles['Heading2'].leading=16;styles['Heading2'].spaceBefore=12;styles['Heading2'].keepWithNext=True
story=[]
for p in text.split('\n\n'):
 p=p.strip()
 if not p:continue
 style=styles['BodyCustom']
 if p.startswith('# '):p=p[2:].replace(' — ',' - ');style=styles['Title']
 elif p.startswith('## '):p=p[3:];style=styles['Heading2']
 p=p.replace('—','-').replace('–','-').replace('×','x').replace('“','"').replace('”','"').replace('’',"'")
 story.append(Paragraph(escape(p).replace('\n','<br/>'),style))
def footer(c,d):
 c.setStrokeColor(colors.HexColor('#c6d6df'));c.line(42,39,553,39);c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#516578'));c.drawString(42,26,'TRACK C | V2 | Machine-validated research pilot; human review pending');c.drawRightString(553,26,str(d.page))
SimpleDocTemplate(str(r/'TECHNICAL_REPORT.pdf'),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=52,title='Track C contrastive speech pilot v2').build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
pdf=PdfReader(r/'TECHNICAL_REPORT.pdf');assert len(pdf.pages)<=6;print('PDF pages',len(pdf.pages))
