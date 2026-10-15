"""Render the concise HTML technical report as a dependency-free PDF."""
from html.parser import HTMLParser
from html import unescape
from pathlib import Path
import re,textwrap

ROOT=Path(__file__).resolve().parents[1]

class Extract(HTMLParser):
    def __init__(self):
        super().__init__();self.lines=[];self.buf='';self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ('style','script'):self.skip+=1
        if not self.skip and tag in ('h1','h2','h3','p','tr','li'):
            self.flush()
        if not self.skip and tag in ('td','th') and self.buf:self.buf+=' | '
    def handle_endtag(self,tag):
        if tag in ('style','script'):self.skip-=1
        if not self.skip and tag in ('h1','h2','h3','p','tr','li'):self.flush()
    def handle_data(self,data):
        if not self.skip:self.buf+=data
    def flush(self):
        t=' '.join(unescape(self.buf).split());self.buf=''
        if t:self.lines.append(t)

def pdf_literal(text):
    text=text.translate(str.maketrans({'≥':'>=','≤':'<=','−':'-','–':'-','—':'-','×':'x','≥':'>=','’':"'"}))
    text=text.encode('latin-1','replace').decode('latin-1')
    return text.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')

def main():
    p=Extract();p.feed((ROOT/'TECHNICAL_REPORT_2026-10-15.html').read_text(encoding='utf-8'));p.flush()
    lines=[]
    for paragraph in p.lines:
        lines.extend(textwrap.wrap(paragraph,width=92,break_long_words=False,break_on_hyphens=False))
        lines.append('')
    while lines and not lines[-1]:lines.pop()
    page_lines=52
    pages=[lines[i:i+page_lines] for i in range(0,len(lines),page_lines)]
    if len(pages)>6:raise RuntimeError(f'Technical report exceeds six pages: {len(pages)}')
    objects=[]
    def add(data):objects.append(data);return len(objects)
    catalog=add(None);pages_id=add(None);font=add(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')
    page_ids=[]
    for number,page in enumerate(pages,1):
        commands=['BT /F1 10.5 Tf 50 790 Td 14 TL']
        for line in page:
            commands.append('('+pdf_literal(line)+') Tj T*')
        commands.extend(['ET','BT /F1 8 Tf 50 35 Td (Track C technical report | 15 October 2026 | page '+str(number)+') Tj ET'])
        stream='\n'.join(commands).encode('latin-1')
        contents=add(b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream')
        page_id=add(b'<< /Type /Page /Parent '+str(pages_id).encode()+b' 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 '+str(font).encode()+b' 0 R >> >> /Contents '+str(contents).encode()+b' 0 R >>')
        page_ids.append(page_id)
    objects[catalog-1]=b'<< /Type /Catalog /Pages '+str(pages_id).encode()+b' 0 R >>'
    objects[pages_id-1]=b'<< /Type /Pages /Count '+str(len(page_ids)).encode()+b' /Kids ['+b' '.join(str(n).encode()+b' 0 R' for n in page_ids)+b'] >>'
    output=bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n');offsets=[0]
    for i,obj in enumerate(objects,1):
        offsets.append(len(output));output+=str(i).encode()+b' 0 obj\n'+obj+b'\nendobj\n'
    start=len(output);output+=b'xref\n0 '+str(len(objects)+1).encode()+b'\n0000000000 65535 f \n'
    for pos in offsets[1:]:output+=f'{pos:010d} 00000 n \n'.encode()
    output+=b'trailer\n<< /Size '+str(len(objects)+1).encode()+b' /Root '+str(catalog).encode()+b' 0 R >>\nstartxref\n'+str(start).encode()+b'\n%%EOF\n'
    dest=ROOT/'TECHNICAL_REPORT_2026-10-15.pdf';dest.write_bytes(output)
    print(f'PDF pages: {len(pages)}, bytes: {len(output)}')

if __name__=='__main__':main()
