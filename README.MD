# Locate RGB values in a picture! 40 x faster than PIL, 5 x faster than numpy

```python
pip install locate-pixelcolor
```

```python
from locate_pixelcolor import search_colors

# Let's use a 4525 x 6623 x 3 picture https://www.pexels.com/pt-br/foto/foto-da-raposa-sentada-no-chao-2295744/
import cv2
path = r"C:\Users\Gamer\Documents\Downloads\pexels-alex-andrews-2295744.jpg"
img = cv2.imread(path)
exa1 = search_colors(pic=img, colors=[(255, 255, 255)])

%timeit search_colors(pic=img, colors=[(255, 255, 255)])
96.8 ms ± 534 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# You can search for up to 9 different colors at the same time:
search_colors(pic=img, colors=[(255, 255, 255), (0, 0, 0)])

%timeit search_colors(pic=img, colors=[(255, 255, 255),(0, 0, 0)])
132 ms ± 382 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Let's compare it with PIL

from PIL import Image
img = Image.open(path)
img = img.convert("RGB")
datas = img.getdata()

def get_coords_with_pil(col):
    newData = []
    for item in datas:
        if item[0] == col[0] and item[1] == col[1] and item[2] == col[2]:
            newData.append(item)
    return newData

# %timeit get_coords_with_pil(col=(255, 255, 255))
# 3.34 s ± 51.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

```
