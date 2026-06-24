import sys, subprocess

def auto_install():
    required = {'google-cloud-storage', 'numpy', 'Pillow'}
    try:
        import google.cloud.storage, numpy, PIL
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *required])

auto_install()

import csv, sys, io
import numpy as np
from PIL import Image
from google.cloud import storage

def main(output, size=32):
    classes = ["Art déco", "Art nouveau", "Gothique"]
    bucket = storage.Client().bucket("photos_classification_architecture")
    with open(output,'w',newline='',encoding='utf-8') as f:
        out=csv.writer(f)
        for label,name in enumerate(classes):
            for blob in bucket.list_blobs(prefix=f"images/{name}/"):
                if not any(blob.name.lower().endswith(ext) for ext in {'.png','.jpg','.jpeg','.bmp','.webp'}): continue
                img=Image.open(io.BytesIO(blob.download_as_bytes())).convert('L').resize((size,size))
                x=(np.asarray(img,dtype=np.float32)/255.0).reshape(-1)
                out.writerow([*x,label])
    print('Classes:',classes)

if __name__=='__main__':
    if len(sys.argv)<2: raise SystemExit(1)
    main(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 32)

