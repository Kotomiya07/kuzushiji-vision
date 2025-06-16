#!/usr/bin/env python3
"""
異体字・外字を常用漢字に変換するスクリプト
（史料編纂所データベースから作成された変換辞書を使用）

使用方法:
    python convert_kanji.py

機能:
    - data/honkoku/honkoku.txt を1文ごとに処理
    - 史料編纂所データベースの異体字変換辞書を使用
    - 変換結果をprocessed_honkoku.txtに保存
    - 変換統計をログファイルに出力
"""

import json
import logging
import re
import shutil
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("kanji_conversion.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class KanjiConverter:
    """異体字・外字を常用漢字に変換するクラス（史料編纂所データベース版）"""

    def __init__(self, conversion_dict_path: str = "conversion_dict.json"):
        """
        初期化：史料編纂所データベースから作成された変換辞書を読み込み

        Args:
            conversion_dict_path: 変換辞書ファイルのパス
        """
        self.variant_to_standard = {}
        self.conversion_stats = Counter()
        self.detected_variants = Counter()
        self.unprocessed_chars = Counter()

        # 史料編纂所データベースの変換辞書を読み込み
        self._load_conversion_dictionary(conversion_dict_path)

        # Unicode正規化を使用した変換も準備
        self.normalization_forms = ["NFC", "NFKC"]

        logger.info("KanjiConverter初期化完了")

    def _load_conversion_dictionary(self, dict_path: str):
        """史料編纂所データベースから作成された変換辞書を読み込み"""
        if not Path(dict_path).exists():
            logger.warning(f"変換辞書ファイルが見つかりません: {dict_path}")
            logger.info("史料編纂所データベースから変換辞書を作成中...")

            # conversion_dict.jsonが存在しない場合は作成
            try:
                import subprocess

                result = subprocess.run(
                    ["python", "scripts/create_itaiji_to_hyojun_map.py"], capture_output=True, text=True, encoding="utf-8"
                )
                if result.returncode != 0:
                    logger.error(f"変換辞書作成に失敗: {result.stderr}")
                    raise RuntimeError("変換辞書の作成に失敗しました")
                logger.info("変換辞書の作成が完了しました")
            except Exception as e:
                logger.error(f"変換辞書作成エラー: {e}")
                logger.info("基本的な変換辞書にフォールバックします")
                self._build_fallback_dictionary()
                return

        try:
            with open(dict_path, encoding="utf-8") as f:
                self.variant_to_standard = json.load(f)
            logger.info(f"史料編纂所変換辞書読み込み完了: {len(self.variant_to_standard)}件")
        except Exception as e:
            logger.error(f"変換辞書読み込みエラー: {e}")
            logger.info("基本的な変換辞書にフォールバックします")
            self._build_fallback_dictionary()

    def _build_fallback_dictionary(self):
        """変換辞書の読み込みに失敗した場合のフォールバック辞書"""
        basic_variants = {
            # 基本的な旧字体から新字体への変換
            "亞": "亜",
            "惡": "悪",
            "壓": "圧",
            "圍": "囲",
            "爲": "為",
            "醫": "医",
            "壹": "一",
            "應": "応",
            "櫻": "桜",
            "奧": "奥",
            "假": "仮",
            "價": "価",
            "畫": "画",
            "擴": "拡",
            "覺": "覚",
            "學": "学",
            "樂": "楽",
            "觀": "観",
            "歸": "帰",
            "氣": "気",
            "舊": "旧",
            "恆": "恒",
            "廣": "広",
            "國": "国",
            "黑": "黒",
            "濟": "済",
            "雜": "雑",
            "參": "参",
            "產": "産",
            "絲": "糸",
            "實": "実",
            "淨": "浄",
            "眞": "真",
            "圖": "図",
            "粹": "粋",
            "聲": "声",
            "專": "専",
            "戰": "戦",
            "禪": "禅",
            "總": "総",
            "臺": "台",
            "體": "体",
            "對": "対",
            "瀧": "滝",
            "單": "単",
            "團": "団",
            "斷": "断",
            "癡": "痴",
            "蟲": "虫",
            "廳": "庁",
            "點": "点",
            "傳": "伝",
            "黨": "党",
            "難": "難",
            "貳": "二",
            "拜": "拝",
            "發": "発",
            "變": "変",
            "邊": "辺",
            "瓣": "弁",
            "寶": "宝",
            "豐": "豊",
            "萬": "万",
            "滿": "満",
            "譽": "誉",
            "餘": "余",
            "龍": "竜",
            "綠": "緑",
            "壘": "塁",
            "類": "類",
            "勵": "励",
            "禮": "礼",
            "靈": "霊",
            "爐": "炉",
            "灣": "湾",
            # 特殊文字
            "〇": "○",
            "◯": "○",
            "〻": "々",
        }
        self.variant_to_standard.update(basic_variants)
        logger.info(f"フォールバック変換辞書構築完了: {len(self.variant_to_standard)}件")

    def _detect_variants_and_gaiji(self, text: str) -> set[str]:
        """異体字・外字を検出"""
        variants = set()

        for char in text:
            # 基本的なASCII文字やひらがな・カタカナをスキップ
            if (
                ord(char) < 128
                or "\u3040" <= char <= "\u309f"  # ひらがな
                or "\u30a0" <= char <= "\u30ff"  # カタカナ
            ):
                continue

            # 変換辞書に存在する文字を検出
            if char in self.variant_to_standard:
                variants.add(char)
                self.detected_variants[char] += 1
                continue

            # CJK統合漢字の範囲外の文字を検出（補完的チェック）
            if (
                ord(char) > 0x9FFF  # 基本的なCJK統合漢字の範囲外
                or "\u3400" <= char <= "\u4dbf"  # CJK拡張A
                or "\u20000" <= char <= "\u2a6dF"  # CJK拡張B
                or "\u2a700" <= char <= "\u2b73F"  # CJK拡張C
            ):
                variants.add(char)
                self.detected_variants[char] += 1

        return variants

    def _convert_character(self, char: str) -> str:
        """単一文字を変換"""
        # 1. 史料編纂所データベースの変換辞書から変換
        if char in self.variant_to_standard:
            converted = self.variant_to_standard[char]
            self.conversion_stats[f"{char} -> {converted}"] += 1
            return converted

        # 2. Unicode正規化を試行
        for form in self.normalization_forms:
            normalized = unicodedata.normalize(form, char)
            if normalized != char and len(normalized) == 1:
                self.conversion_stats[f"{char} -> {normalized} (正規化)"] += 1
                return normalized

        # 3. 変換できない場合はそのまま返し、統計に記録
        self.unprocessed_chars[char] += 1
        return char

    def convert_text(self, text: str) -> str:
        """テキスト全体を変換"""
        result = []
        variants_found = self._detect_variants_and_gaiji(text)

        for char in text:
            if char in variants_found:
                converted_char = self._convert_character(char)
                result.append(converted_char)
            else:
                result.append(char)

        return "".join(result)

    def split_into_sentences(self, text: str) -> list[str]:
        """テキストを文に分割"""
        # 日本語の文区切り文字で分割
        sentences = re.split(r"[。．？！\n]+", text.strip())
        # 空の文を除去し、前後の空白を削除
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def process_file(self, input_file: str, output_file: str) -> dict:
        """ファイルを処理"""
        logger.info(f"処理開始: {input_file} -> {output_file}")

        # バックアップファイル作成
        backup_file = f"{input_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(input_file, backup_file)
        logger.info(f"バックアップ作成: {backup_file}")

        # ファイル読み込み
        try:
            with open(input_file, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning("UTF-8での読み込みに失敗、代替エンコーディングを試行")
            with open(input_file, encoding="cp932", errors="ignore") as f:
                content = f.read()

        logger.info(f"読み込み完了: {len(content)}文字")

        # 文に分割して処理
        sentences = self.split_into_sentences(content)
        logger.info(f"文分割完了: {len(sentences)}文")

        converted_sentences = []
        for i, sentence in enumerate(sentences):
            if i % 1000 == 0:
                logger.info(f"処理中: {i}/{len(sentences)}文")

            converted_sentence = self.convert_text(sentence)
            converted_sentences.append(converted_sentence)

        # 結果を保存
        with open(output_file, "w", encoding="utf-8") as f:
            for sentence in converted_sentences:
                f.write(sentence + "\n")

        logger.info(f"変換完了: {output_file}")

        # 統計情報を返す
        return {
            "total_sentences": len(sentences),
            "total_conversions": sum(self.conversion_stats.values()),
            "unique_conversions": len(self.conversion_stats),
            "unprocessed_chars": len(self.unprocessed_chars),
            "detected_variants": len(self.detected_variants),
        }

    def save_statistics(self, stats_file: str):
        """統計情報をファイルに保存"""
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write("=== 漢字変換統計情報（史料編纂所データベース版）===\n\n")
            f.write(f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"使用変換辞書: 史料編纂所データベース（{len(self.variant_to_standard)}件）\n\n")

            # 変換統計
            f.write("=== 変換統計 ===\n")
            for conversion, count in self.conversion_stats.most_common(50):
                f.write(f"{conversion}: {count}回\n")

            f.write(f"\n総変換回数: {sum(self.conversion_stats.values())}\n")
            f.write(f"ユニーク変換数: {len(self.conversion_stats)}\n\n")

            # 検出された異体字・外字
            f.write("=== 検出された異体字・外字 ===\n")
            for char, count in self.detected_variants.most_common(50):
                f.write(f"{char} (U+{ord(char):04X}): {count}回\n")

            # 未処理文字
            f.write("\n=== 未処理文字 ===\n")
            for char, count in self.unprocessed_chars.most_common(50):
                f.write(f"{char} (U+{ord(char):04X}): {count}回\n")

        logger.info(f"統計情報保存: {stats_file}")


def main():
    """メイン処理"""
    # ファイルパス設定
    input_file = "data/honkoku/honkoku.txt"
    output_file = "data/honkoku/processed_honkoku_shiryohensan.txt"
    stats_file = "kanji_conversion_stats_shiryohensan.txt"

    # ファイル存在チェック
    if not Path(input_file).exists():
        logger.error(f"入力ファイルが見つかりません: {input_file}")
        return

    # 変換器初期化（史料編纂所データベース版）
    converter = KanjiConverter("conversion_dict.json")

    try:
        # ファイル処理
        stats = converter.process_file(input_file, output_file)

        # 統計情報保存
        converter.save_statistics(stats_file)

        # 結果表示
        logger.info("=== 処理結果 ===")
        logger.info(f"総文数: {stats['total_sentences']:,}")
        logger.info(f"総変換回数: {stats['total_conversions']:,}")
        logger.info(f"ユニーク変換数: {stats['unique_conversions']:,}")
        logger.info(f"未処理文字数: {stats['unprocessed_chars']:,}")
        logger.info(f"検出異体字数: {stats['detected_variants']:,}")

        print("\n史料編纂所データベース版変換完了！")
        print(f"入力ファイル: {input_file}")
        print(f"出力ファイル: {output_file}")
        print(f"統計ファイル: {stats_file}")
        print("ログファイル: kanji_conversion.log")
        print(f"使用変換辞書: 史料編纂所データベース（{len(converter.variant_to_standard)}件）")

    except Exception as e:
        logger.error(f"処理エラー: {e}")
        raise


if __name__ == "__main__":
    main()
