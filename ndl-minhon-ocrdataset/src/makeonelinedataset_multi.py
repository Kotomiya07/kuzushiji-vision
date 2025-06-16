import concurrent.futures
import glob
import json
import os
import sys

import cv2
from tqdm import tqdm


# 各画像ファイルとその関連データを処理するワーカー関数
# この関数は、並列実行される個々のタスクを定義します。
def process_single_image_data(task_args):
    imgfilepath, mode, base_outputdir, current_book_or_project_basename, current_book_basename_v2 = task_args

    try:
        img = cv2.imread(imgfilepath)
        if img is None:
            print(f"Warning: 画像 {imgfilepath} を読み取れませんでした。スキップします。", file=sys.stderr)
            return

        if mode == "v1":
            jsonfilepath = imgfilepath.replace(".jpg", ".json").replace("img_v1", "v1")
            target_output_dir = os.path.join(base_outputdir, current_book_or_project_basename)

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
                    outtext = outtext.replace("（", "(").replace("）", ")")

                    if "□" in outtext or "？" in outtext or "?" in outtext or \
                        "𠁅" in outtext or \
                        "𢔗" in outtext or \
                        "𢳣" in outtext or \
                        "𣪃" in outtext or \
                        "𤧀" in outtext or \
                        "𤴼" in outtext or \
                        "𤺺" in outtext or \
                        "𥙊" in outtext or \
                        "𥝐" in outtext or \
                        "𦑄" in outtext or \
                        "𦒳" in outtext or \
                        "𧇾" in outtext or \
                        "𧹛" in outtext or \
                        "𧹬" in outtext or \
                        "𨻳" in outtext or \
                        "𩞀" in outtext or \
                        "𩷚" in outtext or \
                        "𪅂" in outtext or \
                        "𫚄" in outtext:
                        #print(f"Warning: 画像 {imgfilepath} に含まれるテキスト {outtext} に '□' が含まれています。スキップします。", file=sys.stderr)
                        continue
                    else:
                        outputimgpath = os.path.join(
                            target_output_dir,
                            "{}_{}.jpg".format(
                                current_book_or_project_basename + "-" + os.path.basename(imgfilepath).split(".")[0], index
                            ),
                        )
                        outputtxtpath = outputimgpath.replace(".jpg", ".txt")

                        cv2.imwrite(outputimgpath, img[ymin:ymax, xmin:xmax])
                        with open(outputtxtpath, "w", encoding="utf-8") as wf:
                            wf.write(outtext)

        elif mode == "v2":
            jsonfilepath = imgfilepath.replace(".jpg", ".json").replace("img_v2", "v2")
            target_output_dir = os.path.join(base_outputdir, current_book_or_project_basename, current_book_basename_v2)

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
                    outtext = outtext.replace("（", "(").replace("）", ")")

                    if "□" in outtext or "？" in outtext or "?" in outtext or \
                        "𠁅" in outtext or \
                        "𢔗" in outtext or \
                        "𢳣" in outtext or \
                        "𣪃" in outtext or \
                        "𤧀" in outtext or \
                        "𤴼" in outtext or \
                        "𤺺" in outtext or \
                        "𥙊" in outtext or \
                        "𥝐" in outtext or \
                        "𦑄" in outtext or \
                        "𦒳" in outtext or \
                        "𧇾" in outtext or \
                        "𧹛" in outtext or \
                        "𧹬" in outtext or \
                        "𨻳" in outtext or \
                        "𩞀" in outtext or \
                        "𩷚" in outtext or \
                        "𪅂" in outtext or \
                        "𫚄" in outtext:
                        #print(f"Warning: 画像 {imgfilepath} に含まれるテキスト {outtext} に '□' が含まれています。スキップします。", file=sys.stderr)
                        continue
                    else:
                        outputimgpath = os.path.join(
                            target_output_dir,
                            "{}_{}.jpg".format(
                                current_book_or_project_basename
                                + "-"
                                + current_book_basename_v2
                                + "-"
                                + os.path.basename(imgfilepath).split(".")[0],
                                index,
                            ),
                        )
                        outputtxtpath = outputimgpath.replace(".jpg", ".txt")

                        cv2.imwrite(outputimgpath, img[ymin:ymax, xmin:xmax])
                        with open(outputtxtpath, "w", encoding="utf-8") as wf:
                            wf.write(outtext)
    except Exception as e:
        print(f"Error processing {imgfilepath}: {e}", file=sys.stderr)

if __name__ == "__main__":
    args = sys.argv
    mode = ""
    if len(args) >= 2:
        if args[1] == "v1":
            mode = "v1"
        elif args[1] == "v2":
            mode = "v2"
        else:
            print("パラメータは 'v1' または 'v2' でなければなりません。", file=sys.stderr)
            sys.exit(1)
    else:
        print("パラメータは 'v1' または 'v2' でなければなりません。", file=sys.stderr)
        sys.exit(1)

    all_tasks = [] # 並列処理するすべてのタスク（画像のパスとその関連情報）を格納するリスト

    if mode == "v1":
        outputdir = "honkoku_oneline_v1"
        os.makedirs(outputdir, exist_ok=True) # ベース出力ディレクトリを作成
        for bookdir in tqdm(glob.glob(os.path.join("img_v1", "*")), desc="V1タスクを収集中"):
            current_book_basename = os.path.basename(bookdir)
            # 各bookdirに対応する出力ディレクトリを事前に作成
            os.makedirs(os.path.join(outputdir, current_book_basename), exist_ok=True)
            for imgfilepath in glob.glob(os.path.join(bookdir, "*.jpg")):
                # タスクとして、画像のパス、モード、ベース出力ディレクトリ、現在のbookのベース名、v2用のNoneを追加
                all_tasks.append((imgfilepath, mode, outputdir, current_book_basename, None))
    elif mode == "v2":
        outputdir = "honkoku_oneline_v2"
        os.makedirs(outputdir, exist_ok=True) # ベース出力ディレクトリを作成
        for projectdir in tqdm(glob.glob(os.path.join("img_v2", "*")), desc="V2プロジェクトタスクを収集中"):
            current_project_basename = os.path.basename(projectdir)
            # 各projectdirに対応する出力ディレクトリを事前に作成
            os.makedirs(os.path.join(outputdir, current_project_basename), exist_ok=True)
            for bookdir in tqdm(glob.glob(os.path.join(projectdir, "*")), desc=f"V2 bookタスクを {current_project_basename} 内で収集中"):
                current_book_basename = os.path.basename(bookdir)
                # 各projectdir/bookdirに対応する出力ディレクトリを事前に作成
                os.makedirs(os.path.join(outputdir, current_project_basename, current_book_basename), exist_ok=True)
                for imgfilepath in glob.glob(os.path.join(bookdir, "*.jpg")):
                    # タスクとして、画像のパス、モード、ベース出力ディレクトリ、現在のprojectのベース名、現在のbookのベース名を追加
                    all_tasks.append((imgfilepath, mode, outputdir, current_project_basename, current_book_basename))

    print(f"処理するタスクが {len(all_tasks)} 件見つかりました。")

    # ProcessPoolExecutorを使用して並列処理を実行
    # max_workersはデフォルトでCPUのコア数になります。必要に応じて調整可能です。
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # tqdmをexecutor.mapに適用することで、並列処理の進捗状況を表示
        list(tqdm(executor.map(process_single_image_data, all_tasks), total=len(all_tasks), desc="画像を処理中"))

    print("処理が完了しました。")
