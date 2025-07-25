import glob
import json
import os
import sys

import cv2
from tqdm import tqdm

if __name__ == "__main__":
    args = sys.argv
    mode = ""
    if len(args) >= 2:
        if args[1] == "v1":
            mode = "v1"
        elif args[1] == "v2":
            mode = "v2"
        else:
            print("The parameter must be 'v1' or 'v2'.", file=sys.stderr)
            sys.exit(1)
    else:
        print("The parameter must be 'v1' or 'v2'.", file=sys.stderr)
        sys.exit(1)

if mode == "v1":
    outputdir = "honkoku_oneline_v1"
    os.makedirs(outputdir, exist_ok=True)
    for bookdir in glob.glob(os.path.join("img_v1", "*")):
        os.makedirs(os.path.join(outputdir, os.path.basename(bookdir)), exist_ok=True)
        for imgfilepath in glob.glob(os.path.join(bookdir, "*.jpg")):
            jsonfilepath = imgfilepath.replace(".jpg", ".json").replace("img_v1", "v1")
            img = cv2.imread(imgfilepath)
            with open(jsonfilepath, encoding="utf-8") as jsonfile_open:
                jsonfile_load = json.load(jsonfile_open)
                for index, word in enumerate(jsonfile_load):
                    x0, y0 = word["boundingBox"][0]
                    x1, y1 = word["boundingBox"][1]
                    x2, y2 = word["boundingBox"][2]
                    x3, y3 = word["boundingBox"][3]
                    xmin = min([x0, x1, x2, x3])
                    xmax = max([x0, x1, x2, x3])
                    ymin = min([y0, y1, y2, y3])
                    ymax = max([y0, y1, y2, y3])
                    outtext = word["text"]
                    # 括弧を全角から半角に変換
                    outtext = outtext.replace("（", "(")
                    outtext = outtext.replace("）", ")")
                    outputimgpath = os.path.join(
                        outputdir,
                        os.path.basename(bookdir),
                        "{}_{}.jpg".format(
                            os.path.basename(bookdir) + "-" + os.path.basename(imgfilepath).split(".")[0], index
                        ),
                    )
                    outputtxtpath = outputimgpath.replace(".jpg", ".txt")
                    cv2.imwrite(outputimgpath, img[ymin:ymax, xmin:xmax])
                    with open(outputtxtpath, "w", encoding="utf-8") as wf:
                        wf.write(outtext)
if mode == "v2":
    outputdir = "honkoku_oneline_v2"
    os.makedirs(outputdir, exist_ok=True)
    for projectdir in tqdm(glob.glob(os.path.join("img_v2", "*"))):
        for bookdir in tqdm(glob.glob(os.path.join(projectdir, "*"))):
            os.makedirs(os.path.join(outputdir, os.path.basename(projectdir), os.path.basename(bookdir)), exist_ok=True)
            for imgfilepath in tqdm(glob.glob(os.path.join(bookdir, "*.jpg"))):
                jsonfilepath = imgfilepath.replace(".jpg", ".json").replace("img_v2", "v2")
                # print(imgfilepath)
                img = cv2.imread(imgfilepath)
                with open(jsonfilepath, encoding="utf-8") as jsonfile_open:
                    jsonfile_load = json.load(jsonfile_open)
                    for index, word in enumerate(jsonfile_load["words"]):
                        x0, y0 = word["boundingBox"][0]
                        x1, y1 = word["boundingBox"][1]
                        x2, y2 = word["boundingBox"][2]
                        x3, y3 = word["boundingBox"][3]
                        xmin = min([x0, x1, x2, x3])
                        xmax = max([x0, x1, x2, x3])
                        ymin = min([y0, y1, y2, y3])
                        ymax = max([y0, y1, y2, y3])
                        outtext = word["text"]
                        # 括弧を全角から半角に変換
                        outtext = outtext.replace("（", "(")
                        outtext = outtext.replace("）", ")")
                        outputimgpath = os.path.join(
                            outputdir,
                            os.path.basename(projectdir),
                            os.path.basename(bookdir),
                            "{}_{}.jpg".format(
                                os.path.basename(projectdir)
                                + "-"
                                + os.path.basename(bookdir)
                                + "-"
                                + os.path.basename(imgfilepath).split(".")[0],
                                index,
                            ),
                        )
                        outputtxtpath = outputimgpath.replace(".jpg", ".txt")
                        cv2.imwrite(outputimgpath, img[ymin:ymax, xmin:xmax])
                        with open(outputtxtpath, "w", encoding="utf-8") as wf:
                            wf.write(outtext)
