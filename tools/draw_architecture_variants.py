"""Generate editable architecture candidates using the original spectrogram assets."""
from pathlib import Path
import html
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures/architecture_candidates'
BLUE, ORANGE, INK, GRAY = '#28648c', '#bf7b2d', '#233544', '#75828e'
parts = []
images = list(ET.parse(ROOT / '简化视觉架构图.svg').getroot().iter('{http://www.w3.org/2000/svg}image'))
assets = [e.attrib['{http://www.w3.org/1999/xlink}href'].replace('\n', '').replace('\t', '').replace(' ', '') for e in images]

def text(x, y, value, size=21, color=INK, anchor='middle', bold=False):
    for i, line in enumerate(value.split('\n')):
        parts.append(f'<text x="{x}" y="{y+i*(size+5)}" text-anchor="{anchor}" font-size="{size}" fill="{color}" font-weight="{600 if bold else 400}">{html.escape(line)}</text>')

def box(x, y, w, h, label='', color=BLUE, fill='#f0f5fa', size=21):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{color}" stroke-width="1.6"/>')
    if label:
        n = len(label.split('\n'))
        text(x+w/2, y+h/2-(n-1)*(size+5)/2+size*.34, label, size)

def arrow(points, color=BLUE, dash=False):
    pts = ' '.join(f'{x},{y}' for x,y in points)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" {"stroke-dasharray=\"6 5\"" if dash else ""} marker-end="url(#{"orange" if color==ORANGE else "gray" if color==GRAY else "blue"})"/>')

def photo(x, y, normal=True, size=110):
    parts.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="{assets[1 if normal else 0]}"/>')

def features(x, y, cnn=False, selected=False):
    for i in range(7):
        c = ORANGE if cnn else BLUE
        fill = c if selected or i in (1,4) else '#ffffff'
        if cnn:
            parts.append(f'<circle cx="{x+i*18}" cy="{y+(i%2)*12}" r="5" fill="{fill}" stroke="{c}" stroke-width="1.5"/>')
        else:
            parts.append(f'<rect x="{x+i*18-5}" y="{y-5+(i%2)*12}" width="10" height="10" fill="{fill}" stroke="{c}" stroke-width="1.5"/>')

def begin(title, subtitle, height=760):
    parts.clear()
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height}" viewBox="0 0 1400 {height}"><defs>')
    for name,c in [('blue',BLUE),('orange',ORANGE),('gray',GRAY)]:
        parts.append(f'<marker id="{name}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{c}"/></marker>')
    parts.append('</defs><rect width="100%" height="100%" fill="white"/><g font-family="Times New Roman,serif">')
    text(40,42,title,25,anchor='start',bold=True)
    text(40,72,subtitle,18,GRAY,anchor='start')

def save(name, height=760):
    parts.append('</g></svg>')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/f'{name}.svg'
    path.write_text(''.join(parts),encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='spectramemad-figure-') as profile:
        subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--hide-scrollbars',f'--user-data-dir={profile}','--force-device-scale-factor=2',f'--window-size=1400,{height+140}',f'--screenshot={OUT/name}.png',path.as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        with Image.open(OUT/f'{name}.png') as raster:
            raster.crop((0,0,2800,height*2)).save(OUT/f'{name}.png')

def fusion_panel(x,y,w=300,h=220):
    box(x,y,w,h,color=INK,fill='#fafbfc')
    text(x+w/2,y+34,'Reference-Calibrated Fusion',20,bold=True)
    box(x+20,y+58,w-40,44,'Global base  V(x)',BLUE,'#edf4fa',20)
    box(x+20,y+130,w-40,44,'Gated local correction',ORANGE,'#fff4e6',20)
    text(x+w/2,y+122,'+',23)
    text(x+w/2,y+h-15,'Final anomaly score  s(x)',20)

def version_a():
    begin('A  /  Refined two-path layout','Separate global and local memories; normal references calibrate fusion.')
    text(45,123,'NORMAL REFERENCE PATH',19,anchor='start',bold=True)
    photo(50,172)
    text(105,312,'Normal spectrogram',18)
    for yy,cnn in [(160,False),(275,True)]:
        c=ORANGE if cnn else BLUE
        fill='#fff4e6' if cnn else '#edf4fa'
        arrow([(160,227),(196,227),(196,yy+35),(230,yy+35)],c)
        box(230,yy,220,70,'Frozen ResNet-18' if cnn else 'Frozen ViT',c,fill)
        arrow([(450,yy+35),(510,yy+35)],c)
        features(533,yy+28,cnn)
        arrow([(666,yy+35),(700,yy+35)],c)
        box(700,yy,195,70,'Farthest-point\nselection',c,fill,20)
        arrow([(895,yy+35),(950,yy+35)],c)
        box(950,yy,185,70,'Local Memory' if cnn else 'Global Memory',c,fill,20)
    box(1180,183,180,130,'Normal-score\ncalibration',GRAY,'#f7f8fa',20)
    arrow([(1135,195),(1157,195),(1157,217),(1180,217)],GRAY,True)
    arrow([(1135,310),(1157,310),(1157,275),(1180,275)],GRAY,True)
    parts.append('<path d="M40 375H1360" stroke="#d9e0e6"/>')
    text(45,412,'TEST PATH',19,anchor='start',bold=True)
    photo(50,482,False)
    text(105,622,'Test spectrogram',18)
    for yy,cnn in [(465,False),(580,True)]:
        c=ORANGE if cnn else BLUE
        fill='#fff4e6' if cnn else '#edf4fa'
        arrow([(160,537),(196,537),(196,yy+35),(230,yy+35)],c)
        box(230,yy,220,70,'Frozen ResNet-18' if cnn else 'Frozen ViT',c,fill)
        arrow([(450,yy+35),(510,yy+35)],c)
        box(510,yy,225,70,'1-NN matching\n+ score aggregation',c,fill,19)
        arrow([(735,yy+35),(940,yy+35)],c)
        text(830,yy+22,'Local score C(x)' if cnn else 'Global score V(x)',19,c)
    arrow([(1135,210),(1148,210),(1148,355),(620,355),(620,465)],BLUE,True)
    arrow([(950,330),(920,330),(920,435),(755,435),(755,564),(620,564),(620,580)],ORANGE,True)
    fusion_panel(950,446,380,230)
    arrow([(1270,313),(1270,446)],GRAY,True)
    text(40,725,'Blue: ViT branch     Orange: CNN branch     Dashed arrows: reference information',18,GRAY,anchor='start')
    save('A_refined_two_path')

def version_b():
    begin('B  /  Fusion-focused layout','Memory construction and normal-score calibration are shown as distinct operations.')
    text(40,121,'NORMAL REFERENCES',19,anchor='start',bold=True)
    photo(60,151,True,110)
    box(230,139,210,65,'Frozen ViT',BLUE,'#edf4fa')
    box(230,231,210,65,'Frozen ResNet-18',ORANGE,'#fff4e6')
    for yy,cnn in [(139,False),(231,True)]:
        c=ORANGE if cnn else BLUE
        arrow([(170,206),(200,206),(200,yy+32),(230,yy+32)],c)
        arrow([(440,yy+32),(500,yy+32)],c)
        box(500,yy,225,65,'Farthest-point coreset',c,'#fff4e6' if cnn else '#edf4fa',19)
        arrow([(725,yy+32),(785,yy+32)],c)
        box(785,yy,195,65,'Local Memory' if cnn else 'Global Memory',c,'#fff4e6' if cnn else '#edf4fa',20)
    box(1030,139,325,157,color=GRAY,fill='#f7f8fa')
    text(1192,174,'Normal-score calibration',22,bold=True)
    text(1192,214,'Median · IQR · Local rank',20)
    text(1192,257,'Normal references only',18,GRAY)
    arrow([(465,171),(465,217),(1010,217),(1010,195),(1030,195)],GRAY,True)
    parts.append('<path d="M465 263V217" fill="none" stroke="#75828e" stroke-width="2" stroke-dasharray="6 5"/>')
    text(40,385,'TEST SAMPLE',19,anchor='start',bold=True)
    photo(60,448,False,110)
    for yy,cnn in [(421,False),(534,True)]:
        c=ORANGE if cnn else BLUE
        arrow([(170,503),(200,503),(200,yy+32),(230,yy+32)],c)
        box(230,yy,210,65,'Frozen ResNet-18' if cnn else 'Frozen ViT',c,'#fff4e6' if cnn else '#edf4fa')
        arrow([(440,yy+32),(500,yy+32)],c)
        box(500,yy,225,65,'1-NN anomaly score',c,'#fff4e6' if cnn else '#edf4fa',20)
    arrow([(980,180),(995,180),(995,330),(612,330),(612,421)],BLUE,True)
    arrow([(885,296),(885,356),(750,356),(750,510),(612,510),(612,534)],ORANGE,True)
    box(795,405,560,267,color=INK,fill='#fafbfc')
    text(1075,438,'NORMAL-REFERENCE-CALIBRATED FUSION',19,bold=True)
    arrow([(725,453),(825,453),(825,480),(845,480)],BLUE)
    box(845,456,190,50,'Global base V(x)',BLUE,'#edf4fa',20)
    arrow([(725,566),(845,566)],ORANGE)
    box(845,530,280,79,'High local rank\nStronger local evidence',ORANGE,'#fff4e6',19)
    text(1192,542,'+',25)
    arrow([(1035,480),(1192,480),(1192,516)],BLUE)
    arrow([(1125,566),(1192,566),(1192,550)],ORANGE)
    text(1075,647,'s(x) = global base + gated local correction',22)
    arrow([(1192,296),(1192,405)],GRAY,True)
    text(40,725,'The global score supplies the base; calibrated local evidence controls the correction.',19,GRAY,anchor='start')
    save('B_fusion_focused')

def version_c():
    begin('C  /  Compact stage layout','A left-to-right view of feature extraction, memory retrieval, and calibrated fusion.',650)
    for x,w,title in [(40,195,'INPUTS'),(270,240,'FROZEN ENCODERS'),(565,360,'DUAL-SCALE MEMORY'),(980,380,'FUSION')]:
        text(x+w/2,120,title,19,bold=True)
        parts.append(f'<path d="M{x} 137h{w}" stroke="#cdd6df" stroke-width="1.4"/>')
    photo(75,175,True,100);text(125,302,'Normal references',19)
    photo(75,365,False,100);text(125,492,'Test sample',19)
    box(275,220,235,80,'Frozen ViT',BLUE,'#edf4fa')
    box(275,390,235,80,'Frozen ResNet-18',ORANGE,'#fff4e6')
    arrow([(175,224),(215,224),(215,250),(275,250)],BLUE)
    arrow([(175,235),(200,235),(200,420),(275,420)],ORANGE)
    arrow([(175,415),(230,415),(230,280),(275,280)],BLUE)
    arrow([(175,430),(275,450)],ORANGE)
    for yy,cnn in [(190,False),(370,True)]:
        c=ORANGE if cnn else BLUE;fill='#fff4e6' if cnn else '#edf4fa'
        box(565,yy,360,145,color=c,fill=fill)
        text(745,yy+30,'LOCAL MEMORY' if cnn else 'GLOBAL MEMORY',19,c,bold=True)
        text(745,yy+63,'Farthest-point coreset',20)
        parts.append(f'<path d="M585 {yy+79}h320" stroke="{c}" stroke-opacity="0.3"/>')
        text(745,yy+116,'Test features → 1-NN score',21)
        arrow([(510,yy+70),(565,yy+70)],c)
    box(990,167,360,95,'Normal-score calibration\nMedian · IQR · Local rank',GRAY,'#f7f8fa',20)
    arrow([(925,235),(957,235),(957,200),(990,200)],GRAY,True)
    arrow([(925,413),(965,413),(965,240),(990,240)],GRAY,True)
    fusion_panel(990,310,360,225)
    arrow([(1170,262),(1170,310)],GRAY,True)
    arrow([(925,295),(978,295),(978,390),(990,390)],BLUE)
    arrow([(925,475),(990,475)],ORANGE)
    text(40,605,'Normal features build the memories and calibration statistics; test features query the fixed memories.',19,GRAY,anchor='start')
    save('C_compact_stages',650)

if __name__ == '__main__':
    version_a()
    version_b()
    version_c()
    print(OUT)
