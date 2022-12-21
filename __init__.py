import numexpr
import numpy as np


def search_colors(pic: np.ndarray, colors: list[tuple]) -> np.ndarray:
    colorstosearch = colors
    red = pic[..., 0]
    green = pic[..., 1]
    blue = pic[..., 2]
    wholedict = {"blue": blue, "green": green, "red": red}
    wholecommand = ""
    for ini, co in enumerate(colorstosearch):
        for ini2, col in enumerate(co):
            wholedict[f"varall{ini}_{ini2}"] = np.array([col]).astype(np.uint8)
        wholecommand += f"((red == varall{ini}_0) & (green == varall{ini}_1) & (blue == varall{ini}_2))|"
    wholecommand = wholecommand.strip("|")
    expre = numexpr.evaluate(wholecommand, local_dict=wholedict)
    exa = np.array(np.where(expre)).T[::-1]
    return np.vstack([exa[..., 1], exa[..., 0]]).T
