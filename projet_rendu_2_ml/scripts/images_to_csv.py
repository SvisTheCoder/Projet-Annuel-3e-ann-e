import csv, sys
from pathlib import Path
import numpy as np
from PIL import Image

def main(folder, output, size=32):
    root=Path(folder); classes=sorted(p.name for p in root.iterdir() if p.is_dir())
    with open(output,'w',newline='',encoding='utf-8') as f:
        out=csv.writer(f)
        for label,name in enumerate(classes):
            for p in sorted((root/name).iterdir()):
                if p.suffix.lower() not in {'.png','.jpg','.jpeg','.bmp','.webp'}: continue
                img=Image.open(p).convert('L').resize((size,size))
                x=(np.asarray(img,dtype=np.float32)/255.0).reshape(-1)
                out.writerow([*x,label])
    print('Classes:',classes)
if __name__=='__main__':
    if len(sys.argv)<3: print('Usage: python images_to_csv.py dossier_images sortie.csv [taille]'); raise SystemExit(1)
    main(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 32)
