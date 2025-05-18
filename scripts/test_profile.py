# %%
import pstats
from pstats import SortKey

# プロファイルデータをロード
p = pstats.Stats("padding_data_fix1.prof")

# 不要なパス情報を削除して表示を短縮
p.strip_dirs()

# 累積時間 (cumtime) でソートして上位10件を表示
print("Sort by cumulative time:")
p.sort_stats(SortKey.CUMULATIVE).print_stats(10)

# 関数ごとの実行時間 (tottime) でソートして上位10件を表示
print("\nSort by total time (per function):")
p.sort_stats(SortKey.TIME).print_stats(10)

# 呼び出し回数 (ncalls) でソートして上位10件を表示
print("\nSort by number of calls:")
p.sort_stats(SortKey.CALLS).print_stats(10)
# %%
