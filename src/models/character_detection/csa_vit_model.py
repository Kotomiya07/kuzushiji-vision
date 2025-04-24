# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig
import heapq # Beam Search用

# ViT の内部コンポーネントをインポート
from transformers.models.vit.modeling_vit import (
    ViTAttention,
    ViTEmbeddings,
    ViTIntermediate,
    ViTOutput,
)

# 必要に応じて他のモジュールをインポート
try:
    from torch_geometric.nn import GATConv
    PYG_AVAILABLE = True
except ImportError:
    GATConv = None # PyG がない場合はプレースホルダー
    PYG_AVAILABLE = False
    print("Warning: PyTorch Geometric (PyG) not found. GAT structure module will not be available.")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
    # torchaudio.functional.ctc_decode が存在するか確認 (バージョン依存)
    if not hasattr(torchaudio.functional, 'ctc_decode'):
        print("Warning: torchaudio.functional.ctc_decode not found. CTC Beam Search decoding will not be available. Please update torchaudio.")
        TORCHAUDIO_AVAILABLE = False
except ImportError:
    torchaudio = None
    TORCHAUDIO_AVAILABLE = False
    print("Warning: torchaudio not found. CTC Beam Search decoding will not be available.")


def _build_patch_graph(num_patches_h: int, num_patches_w: int, device: torch.device) -> torch.Tensor:
    """パッチグリッドの隣接関係に基づいて edge_index を作成する (4-connectivity)"""
    num_patches = num_patches_h * num_patches_w
    edge_list = []
    for i in range(num_patches):
        r, c = divmod(i, num_patches_w)
        # 上下左右の隣接ノードを追加
        neighbors = []
        if r > 0: neighbors.append(i - num_patches_w) # 上
        if r < num_patches_h - 1: neighbors.append(i + num_patches_w) # 下
        if c > 0: neighbors.append(i - 1) # 左
        if c < num_patches_w - 1: neighbors.append(i + 1) # 右

        for neighbor in neighbors:
            edge_list.append([i, neighbor])
            # 無向グラフとして扱うため逆方向も追加 (GATConvは有向グラフベースだが、多くの実装で無向を想定)
            edge_list.append([neighbor, i])

    if not edge_list: # パッチが1つの場合など
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # 重複を除去 (上下左右で2回追加されるため)
    edge_set = set(tuple(sorted(edge)) for edge in edge_list)
    unique_edge_list = [list(edge) for edge in edge_set]


    # torch_geometric の形式 [2, num_edges] に変換
    edge_index = torch.tensor(unique_edge_list, dtype=torch.long, device=device).t().contiguous()
    return edge_index


# --- CSA-ViT Layer ---
class CSAViTLayer(nn.Module):
    """CSA-ViTのカスタムTransformerブロック"""
    def __init__(self, config: ViTConfig, use_structure_module: bool = False, use_context_module: bool = False):
        super().__init__()
        self.config = config
        self.use_structure_module = use_structure_module
        self.use_context_module = use_context_module

        # 1. 構造適応モジュール
        if self.use_structure_module:
            # model_config.structure_module_type: 'cnn' or 'gat' or None
            # model_config は CSAViTEncoder から渡される想定
            structure_module_type = self.model_config.get("structure_module_type")

            if structure_module_type == 'cnn':
                # Depthwise Separable Convolution
                # model_config.structure_cnn_kernel_size: 例 3
                kernel_size = self.model_config.get("structure_cnn_kernel_size", 3)
                padding = kernel_size // 2
                hidden_size = config.hidden_size
                # BatchNormを使うかどうかも設定可能に
                use_batch_norm = self.model_config.get("structure_cnn_use_bn", True)

                layers = [
                    # Depthwise Conv
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding, groups=hidden_size, bias=not use_batch_norm),
                ]
                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(hidden_size))
                layers.extend([
                    nn.ReLU(),
                    # Pointwise Conv
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=not use_batch_norm),
                ])
                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(hidden_size))
                layers.append(nn.ReLU())

                self.structure_module = nn.Sequential(*layers)
                # CNNブランチであることを示す属性を追加 (forwardでの判定用)
                self.is_cnn_structure = True
                print(f"Initialized Structure-Aware Module (CNN Branch, kernel={kernel_size}, BN={use_batch_norm})")

            elif structure_module_type == 'gat':
                if not PYG_AVAILABLE:
                     raise ImportError("PyTorch Geometric (PyG) is required for GAT structure module, but it's not installed.")

                # GATConv の設定 (設定ファイルから読み込む想定)
                hidden_size = config.hidden_size
                # model_config.gat_heads: 例 4
                gat_heads = self.model_config.get("gat_heads", 4)
                # model_config.gat_dropout: 例 0.1
                gat_dropout = self.model_config.get("gat_dropout", 0.1)
                # model_config.gat_add_self_loops: 例 True
                add_self_loops = self.model_config.get("gat_add_self_loops", True)

                # GATConv レイヤーを初期化
                # 入力/出力次元は hidden_size
                # concat=False にすると、マルチヘッドの出力が平均化される (次元が変わらない)
                # concat=True の場合、出力次元は hidden_size * gat_heads になるため、後続の処理で調整が必要
                # ここでは concat=False を仮定
                self.structure_module = GATConv(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    heads=gat_heads,
                    dropout=gat_dropout,
                    add_self_loops=add_self_loops,
                    concat=False # 出力次元を hidden_size に保つ
                )
                self.is_cnn_structure = False
                print(f"Initialized Structure-Aware Module (GAT Branch, Heads: {gat_heads}, Dropout: {gat_dropout}, Concat: False)")

            else:
                print(f"Warning: Unknown or no structure_module_type specified ('{structure_module_type}'). Using Identity for structure module.")
                self.structure_module = nn.Identity()
                self.is_cnn_structure = False
        else:
            self.structure_module = nn.Identity()
            self.is_cnn_structure = False

        # LayerNorm before MHSA/Structure Module
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 2. Multi-Head Self-Attention (MHSA)
        self.attention = ViTAttention(config)

        # LayerNorm after MHSA, before Context Module
        self.layernorm_after_mhsa = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # 追加

        # 3. 文脈統合モジュール (Cross-Attention)
        if self.use_context_module:
            # Cross-Attention用の設定
            # model_config.context_embedding_dim: 例 256
            # model_config.layout_embedding_dim: 例 128
            # model_config.cross_attention_heads: 例 config.num_attention_heads と同じか、別の値
            # model_config.cross_attention_dropout: 例 config.attention_probs_dropout_prob
            context_dim = self.model_config.get("context_embedding_dim", 0)
            layout_dim = self.model_config.get("layout_embedding_dim", 0)
            # 文脈とレイアウトの両方を使う場合、次元を合わせるか、別々に処理するか検討
            # ここでは単純に合計するが、実際には projection layer などが必要になる可能性が高い
            kv_dim = context_dim + layout_dim
            if kv_dim > 0:
                hidden_size = config.hidden_size
                num_heads = self.model_config.get("cross_attention_heads", config.num_attention_heads)
                dropout = self.model_config.get("cross_attention_dropout", config.attention_probs_dropout_prob)

                # 補助入力を hidden_size に射影し、活性化・正規化する層
                self.kv_projection = nn.Sequential(
                    nn.Linear(kv_dim, hidden_size),
                    nn.ReLU(), # 活性化関数
                    nn.LayerNorm(hidden_size, eps=config.layer_norm_eps) # LayerNormを追加
                )
                print(f"Initialized KV projection layer with ReLU and LayerNorm for Cross-Attention (Input: {kv_dim}, Output: {hidden_size})")

                # 補助入力用のLayerNorm (次元が0より大きい場合のみ初期化)
                if context_dim > 0:
                    self.layernorm_context = nn.LayerNorm(context_dim, eps=config.layer_norm_eps)
                if layout_dim > 0:
                    self.layernorm_layout = nn.LayerNorm(layout_dim, eps=config.layer_norm_eps)

                # Cross-Attention モジュール (nn.MultiheadAttention を使用)
                self.context_module = nn.MultiheadAttention(
                    embed_dim=hidden_size,      # Query dimension
                    kdim=hidden_size,           # Key dimension (after projection)
                    vdim=hidden_size,           # Value dimension (after projection)
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True            # 入力形式を [B, N, D] に
                )
                self.has_context_module = True
                print(f"Initialized Cross-Attention Module (nn.MultiheadAttention, Heads: {num_heads})")
            else:
                print("Warning: Context/Layout embedding dimensions are zero or not specified. Skipping Cross-Attention module initialization.")
                self.context_module = nn.Identity()
                self.has_context_module = False
        else:
            self.context_module = nn.Identity()
            self.has_context_module = False

        # LayerNorm after Context Module, before FFN
        self.layernorm_after_context = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # 追加

        # 4. FeedForward Network (FFN)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config) # FFN + Residual + Dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_embeddings: torch.Tensor = None,
        layout_embeddings: torch.Tensor = None,
        head_mask: torch.Tensor | None = None, # ViTAttention用
        output_attentions: bool = False, # ViTAttention用
    ) -> tuple[torch.Tensor, ...]:

        # --- 構造適応 & MHSA ---
        # LayerNorm -> Structure Module (if applicable) -> MHSA -> Residual
        normed_hidden_states = self.layernorm_before(hidden_states)

        # 構造適応モジュール (CNNの場合、形状変換が必要)
        # CNNブランチかどうかを __init__ で設定した属性で判定
        if self.use_structure_module and self.is_cnn_structure:
            # CLSトークンを除外して形状変換: [B, N_patches+1, D] -> [B, N_patches, D]
            # ViTEmbeddingsはCLSトークンを先頭に追加するので、それを分離
            cls_token = normed_hidden_states[:, 0:1, :]
            patch_tokens = normed_hidden_states[:, 1:, :]
            B, N, D = patch_tokens.shape
            # パッチ数が期待値と一致するか確認 (入力画像サイズが変わった場合など)
            if N != self.num_patches:
                 # 実行時エラーの方が親切かもしれない
                 print(f"Warning: Number of patches ({N}) does not match expected ({self.num_patches}). Reshaping might fail.")
                 # 必要ならここでエラーを発生させるか、処理をスキップする
                 input_to_attention = normed_hidden_states # 構造適応をスキップ
            else:
                # [B, N, D] -> [B, D, H', W']
                patch_tokens_reshaped = patch_tokens.permute(0, 2, 1).reshape(B, D, self.num_patches_h, self.num_patches_w)
                # CNN適用
                cnn_features = self.structure_module(patch_tokens_reshaped)
                # [B, D, H', W'] -> [B, N, D]
                cnn_features_reshaped = cnn_features.flatten(2).permute(0, 2, 1)
                # 元のパッチトークンに加算 (CLSトークンはそのまま)
                # 加算ではなく連結(concat)やゲート機構も検討可能
                structured_patch_tokens = patch_tokens + cnn_features_reshaped
                # CLSトークンを再結合: [B, N_patches+1, D]
                input_to_attention = torch.cat((cls_token, structured_patch_tokens), dim=1)

        elif self.use_structure_module: # GATの場合
            if not PYG_AVAILABLE:
                 # __init__でチェック済みだが念のため
                 print("Warning: PyG not available, skipping GAT.")
                 input_to_attention = normed_hidden_states
            else:
                # CLSトークンを除外
                cls_token = normed_hidden_states[:, 0:1, :]
                patch_tokens = normed_hidden_states[:, 1:, :]
                B, N, D = patch_tokens.shape

                if N != self.num_patches:
                    print(f"Warning: Number of patches ({N}) does not match expected ({self.num_patches}). Skipping GAT.")
                    input_to_attention = normed_hidden_states
                else:
                    try:
                        # グラフ構造を構築 (バッチ内で共通)
                        # Note: 毎回構築するのは非効率な場合がある。__init__で計算して保持する方が良いかも。
                        single_edge_index = _build_patch_graph(self.num_patches_h, self.num_patches_w, patch_tokens.device)

                        # PyGのバッチ処理形式に変換
                        x = patch_tokens.reshape(B * N, D)
                        num_nodes = N
                        # バッチ内の各グラフのedge_indexをオフセットして結合
                        edge_indices = [single_edge_index + i * num_nodes for i in range(B)]
                        batch_edge_index = torch.cat(edge_indices, dim=1)

                        # GATConv適用
                        gat_output_flat = self.structure_module(x, batch_edge_index) # [B*N, D]

                        # 元の形状に戻す
                        gat_output = gat_output_flat.reshape(B, N, D)

                        # 残差接続
                        structured_patch_tokens = patch_tokens + gat_output

                        # CLSトークンを再結合
                        input_to_attention = torch.cat((cls_token, structured_patch_tokens), dim=1)

                    except Exception as e:
                        print(f"Error during GAT processing: {e}. Skipping GAT.")
                        input_to_attention = normed_hidden_states

        else: # 構造適応モジュールなし
            input_to_attention = normed_hidden_states


        # MHSA + Residual (ViTAttention内部で残差接続は行われないので注意)
        # attention_outputs はタプル (attention_output, attention_probs[optional])
        attention_outputs = self.attention(
            input_to_attention, # 構造適応後の特徴を入力
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]
        self_attn_weights = attention_outputs[1] if output_attentions else None # Self-Attentionの重みを取得

        # 手動で残差接続
        hidden_states = hidden_states + attention_output

        # --- 文脈統合 ---
        # LayerNorm -> Context Module -> Residual
        normed_hidden_states_for_context = self.layernorm_after_mhsa(hidden_states) # MHSA後にLayerNorm

        cross_attention_weights = None # 初期化
        # Cross-Attentionの実行
        if self.use_context_module and self.has_context_module and (context_embeddings is not None or layout_embeddings is not None):
            # 補助入力を結合・処理
            processed_kv = []
            if context_embeddings is not None and hasattr(self, 'layernorm_context'):
                # context_embeddings の前処理 (LayerNorm)
                normed_context = self.layernorm_context(context_embeddings)
                processed_kv.append(normed_context)
            elif context_embeddings is not None: # LayerNormがない場合 (次元が0など)
                processed_kv.append(context_embeddings)

            if layout_embeddings is not None and hasattr(self, 'layernorm_layout'):
                # layout_embeddings の前処理 (LayerNorm)
                normed_layout = self.layernorm_layout(layout_embeddings)
                processed_kv.append(normed_layout)
            elif layout_embeddings is not None: # LayerNormがない場合
                processed_kv.append(layout_embeddings)


            if processed_kv:
                # 結合 (次元が異なる場合は注意)
                combined_kv = torch.cat(processed_kv, dim=1) # [B, N_context + N_layout, D_kv]

                # 補助入力を射影
                projected_kv = self.kv_projection(combined_kv)

                # Cross-Attention実行 (nn.MultiheadAttention)
                # Q: normed_hidden_states_for_context [B, N_img+1, D_hid]
                # K: projected_kv [B, N_ctx+N_lay, D_hid]
                # V: projected_kv [B, N_ctx+N_lay, D_hid]
                cross_attention_output, cross_attention_weights = self.context_module(
                    query=normed_hidden_states_for_context,
                    key=projected_kv,
                    value=projected_kv,
                    need_weights=output_attentions # output_attentionsフラグに応じて重みを計算
                )
                # cross_attention_output: [B, N_img+1, D_hid]

                # 手動で残差接続 (Cross-Attentionの結果を加算) + Dropout
                # nn.MultiheadAttentionのdropoutとは別に適用
                context_output_with_residual = hidden_states + cross_attention_output
                # config.hidden_dropout_prob を流用
                dropout_prob = self.config.hidden_dropout_prob
                hidden_states = nn.Dropout(dropout_prob)(context_output_with_residual)
            else:
                 # 補助入力がない場合はスキップ
                 pass
        else:
            # 文脈統合を使用しない、または補助入力がない場合はスキップ
            pass

        # --- FFN ---
        # LayerNorm -> FFN -> Residual (ViTOutput内部で実行)
        layer_output = self.layernorm_after_context(hidden_states) # 文脈統合後にLayerNorm
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states) # ViTOutput内で残差接続

        outputs = (layer_output,)
        if output_attentions:
            # Self-Attentionの重みを追加
            outputs = outputs + (self_attn_weights,)
            # Cross-Attentionの重みを追加 (計算されていれば)
            if self.use_context_module and self.has_context_module:
                 # cross_attention_weights が計算された場合はそれを、そうでなければ None を追加
                 outputs = outputs + (cross_attention_weights,)
            # 文脈モジュールを使わない場合でも、返り値の要素数を合わせるために None を追加するかどうかは設計次第
            # ここでは、文脈モジュールを使わない場合は Cross-Attention の重み自体が存在しないので追加しない
        return outputs


# --- CSA-ViT Encoder ---
class CSAViTEncoder(nn.Module):
    # model_config (辞書) を引数に追加
    def __init__(self, config: ViTConfig, model_config: dict, num_layers: int, use_structure_module_indices: list[int] | None = None, use_context_module_indices: list[int] | None = None):
        super().__init__()
        self.config = config # ViTConfig
        self.model_config = model_config # 元の辞書config
        self.layer = nn.ModuleList()
        for i in range(num_layers):
            use_structure = use_structure_module_indices is not None and i in use_structure_module_indices
            use_context = use_context_module_indices is not None and i in use_context_module_indices
            # CSAViTLayerに ViTConfig と model_config を渡す
            layer = CSAViTLayer(config, use_structure_module=use_structure, use_context_module=use_context)
            self.layer.append(layer)
        self.gradient_checkpointing = False # 必要ならTrueにする

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_embeddings: torch.Tensor = None,
        layout_embeddings: torch.Tensor = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> tuple | dict:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        # Cross-Attentionの重みを格納するタプルを初期化 (output_attentionsがTrueの場合のみ)
        all_cross_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                # Gradient Checkpointing を適用
                # checkpoint 関数はタプルを返すため、非タプルの返り値の場合は調整が必要
                # CSAViTLayer.forward はタプルを返すのでそのまま使える
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer_module,
                    hidden_states,
                    context_embeddings,
                    layout_embeddings,
                    layer_head_mask,
                    output_attentions,
                    # use_reentrant=False を推奨 (PyTorch 1.11以降)
                    # ただし、互換性のためにデフォルト(True)のままにするか、バージョンを確認して設定
                    use_reentrant=True # or False depending on PyTorch version and preference
                )
            else:
                layer_outputs = layer_module(hidden_states, context_embeddings, layout_embeddings, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                # layer_outputs は CSAViTLayer.forward の返り値タプル
                # (layer_output, self_attn_weights[opt], cross_attn_weights[opt])

                # Self-Attention weights
                # all_self_attentions が None でないことを確認 (output_attentions=True)
                if all_self_attentions is not None and len(layer_outputs) > 1:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],) # layer_outputs[1] は None の可能性あり

                # Cross-Attention weights
                # all_cross_attentions が None でないことを確認 (output_attentions=True)
                if all_cross_attentions is not None:
                    # layer_outputs[2] が存在する場合のみ追加 (文脈モジュール未使用層では存在しない)
                    cross_attn_weight = layer_outputs[2] if len(layer_outputs) > 2 else None
                    all_cross_attentions = all_cross_attentions + (cross_attn_weight,) # None も含めて追加


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            # Cross-Attentionの重みも返すように変更
            outputs = [hidden_states]
            if output_hidden_states:
                outputs.append(all_hidden_states)
            if output_attentions:
                outputs.append(all_self_attentions)
                outputs.append(all_cross_attentions) # all_cross_attentions は None の可能性あり
            return tuple(v for v in outputs if v is not None) # None のタプルは返さない

        return { # BaseModelOutputを模倣
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions, # Self-Attention weights (Noneを含む可能性あり)
            "cross_attentions": all_cross_attentions, # Cross-Attention weights (Noneを含む可能性あり)
        }


class CSAViTModel(nn.Module):
    """
    文脈・構造適応型 Vision Transformer (CSA-ViT) モデル。
    設計ドキュメント: docs/character_detection_model_architecture_plan.md
    実装計画: docs/csa_vit_implementation_plan.md
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): モデル設定 (config/model/character_detection.yaml から読み込まれる想定)
        """
        super().__init__()
        self.config = config
        self.current_epoch = 0 # 損失計算などで使用する場合

        # --- ここから各モジュールを初期化 ---
        # 1. 入力モジュール (パッチ埋め込み、位置エンコーディング)
        # ViT設定 (configから読み込む想定)
        # config.model.input_size: 例 [224, 224]
        # config.model.patch_size: 例 16
        # config.model.num_channels: 例 3
        # config.model.hidden_size: 例 768
        # config.model.dropout: 例 0.1
        vit_config = ViTConfig(
            image_size=config.model.input_size[0],
            patch_size=config.model.patch_size,
            num_channels=config.model.num_channels,
            hidden_size=config.model.hidden_size,
            hidden_dropout_prob=config.model.dropout,
            # ViTEmbeddingsが必要とする他の設定があれば追加
            layer_norm_eps=config.model.layer_norm_eps, # LayerNormのepsilonもViTConfigに渡す
        )
        self.embeddings = ViTEmbeddings(vit_config)

        # 2. CSA-ViT エンコーダ
        # config.model.num_layers: 例 12
        # config.model.use_structure_module_indices: 例 [0, 1, 2] # どの層で構造適応を使うか
        # config.model.use_context_module_indices: 例 [9, 10, 11] # どの層で文脈統合を使うか
        self.encoder = CSAViTEncoder(
            vit_config, # ViTConfigオブジェクト
            model_config=config.model, # 元の辞書configを渡す
            num_layers=config.model.num_layers,
            # use_structure/context_module_indices を正しく渡す
            use_structure_module_indices=config.model.get("use_structure_module_indices"),
            use_context_module_indices=config.model.get("use_context_module_indices"),
        )
        # self.encoder.config = config.model # CSAViTEncoderの__init__で設定されるので不要

        # 3. デコーダ (Transformer or CTC or Hybrid)
        # config.model.decoder_type: 例 'ctc' or 'transformer'
        decoder_type = config.model.get("decoder_type", "ctc") # デフォルトはCTCとする
        if decoder_type == "ctc":
            self.ctc_head = nn.Linear(config.model.hidden_size, config.model.num_classes + 1) # +1 for blank token
        elif decoder_type == "transformer":
            # Transformerデコーダの初期化
            hidden_size = config.model.hidden_size
            num_classes = config.model.num_classes
            decoder_layers = config.model.decoder_layers
            nhead = config.model.num_heads # エンコーダと同じヘッド数を使用
            dim_feedforward = config.model.decoder_ffn_dim
            dropout = config.model.dropout
            max_target_length = config.model.max_target_length

            # BOS (Begin of Sequence) と EOS/PAD (End of Sequence / Padding) の ID
            # num_classes は 0 から num_classes-1 までのクラスID
            self.bos_token_id = num_classes
            self.eos_token_id = num_classes + 1 # EOS と PAD を兼ねる
            vocab_size = num_classes + 2 # クラス数 + BOS + EOS/PAD

            # ターゲット埋め込み (語彙サイズ: クラス数 + BOS + EOS/PAD)
            self.decoder_embedding = nn.Embedding(vocab_size, hidden_size)

            # 位置エンコーディング (学習可能)
            self.decoder_pos_encoding = nn.Embedding(max_target_length, hidden_size)

            # Transformer Decoder Layer
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True # 入力形式を [B, SeqLen, Dim] に
            )

            # Transformer Decoder
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=decoder_layers,
                norm=nn.LayerNorm(hidden_size, eps=config.model.layer_norm_eps) # デコーダ出力の正規化
            )

            # 出力層 (デコーダ出力を語彙確率に変換)
            # 予測対象は元のクラス + EOS/PAD (BOSは予測しない)
            self.decoder_output_layer = nn.Linear(hidden_size, num_classes + 1) # 出力はクラス数 + EOS/PAD

            print(f"Initialized Transformer Decoder (Layers: {decoder_layers}, Heads: {nhead}, Vocab: {vocab_size})")

        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")
        self.decoder_type = decoder_type


        # LayerNorm for the final encoder output
        self.layernorm = nn.LayerNorm(config.model.hidden_size, eps=config.model.layer_norm_eps)


        # --- モジュールの初期化ここまで ---

        print(f"Initializing CSAViTModel (Decoder: {self.decoder_type})...") # 初期化確認用

    def forward(self,
                images: torch.Tensor,
                context_embeddings: torch.Tensor = None, # 補助入力: 文脈
                layout_embeddings: torch.Tensor = None,  # 補助入力: レイアウト
                targets: dict[str, list[torch.Tensor]] = None # 学習時のターゲット
                ) -> dict[str, torch.Tensor]:
        """
        順伝播処理

        Args:
            images (torch.Tensor): 主入力画像 [B, C, H, W]
            context_embeddings (torch.Tensor, optional): 周辺文脈埋め込み [B, N_context, D_context]
            layout_embeddings (torch.Tensor, optional): レイアウト埋め込み [B, N_layout, D_layout]
            targets (dict[str, list[torch.Tensor]], optional): 学習時のターゲット情報
                - boxes: バウンディングボックス [B, M, 4]
                - labels: ラベル [B, M]
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]:
                学習時: 損失の辞書 (例: {"loss": ..., "detection_loss": ..., "classification_loss": ...})
                推論時: 検出結果の辞書 (例: {"boxes": ..., "scores": ..., "labels": ...})
        """

        # --- ここから順伝播処理を実装 ---
        # 1. 入力画像をパッチ埋め込みに変換
        embedding_output = self.embeddings(images) # [B, N_patches + 1, D] (CLSトークン含む)

        # 2. CSA-ViT エンコーダで特徴抽出
        # head_mask は現状使用しないため None を渡す
        encoder_outputs = self.encoder(
            embedding_output,
            context_embeddings=context_embeddings,
            layout_embeddings=layout_embeddings,
            head_mask=None, # head_mask は使用しない
            output_attentions=False, # 必要ならTrue
            output_hidden_states=False, # 必要ならTrue
            return_dict=True,
        )
        sequence_output = encoder_outputs["last_hidden_state"] # [B, N_patches + 1, D]
        sequence_output = self.layernorm(sequence_output) # 最終出力にLayerNorm

        # CLSトークンを除外 (デコーダの種類によって扱いが変わる可能性あり)
        # CTCの場合はパッチ特徴のみ使うことが多い
        patch_features = sequence_output[:, 1:, :] # [B, N_patches, D]

        # 3. デコーダで文字シーケンス or 確率分布を出力
        if self.decoder_type == "ctc":
            logits = self.ctc_head(patch_features) # [B, N_patches, NumClasses+1]
            # CTC損失計算のために対数ソフトマックスを適用
            log_probs = F.log_softmax(logits, dim=-1)

            # 4. 損失計算 or 後処理 (CTC)
            if self.training and targets is not None:
                # CTC損失計算
                loss = self._compute_ctc_loss(log_probs, targets)
                return {"loss": loss}
            else:
                # 推論時の後処理 (Greedy decoding)
                # TODO: Beam search の実装も検討
                decoded_ids, decoded_scores = self._decode_ctc(log_probs)
                # デコード結果を整形して返す (DETR形式に合わせるか、シーケンス形式にするか要検討)
                # ここでは仮にDETR形式に合わせるためのダミー値を返す
                batch_size = images.size(0)
                dummy_boxes = [torch.zeros((0, 4), device=images.device)] * batch_size
                # decoded_scores は各シーケンスのスコアのリスト
                # decoded_ids は各シーケンスのIDテンソルのリスト
                scores_list = decoded_scores
                labels_list = decoded_ids

                # TODO: デコード結果 (labels_list, scores_list) から実際のボックス、スコア、ラベルを生成する処理を追加
                # 例: パッチ位置とIDシーケンスをマッピングしてボックスを推定するなど
                return {"boxes": dummy_boxes, "scores": scores_list, "labels": labels_list}

        elif self.decoder_type == "transformer":
            # エンコーダ出力 (CLSトークン含むか、除くかは設計次第。ここでは含む)
            # memory: [B, N_patches + 1, D]
            memory = sequence_output

            if self.training and targets is not None:
                # --- 学習時の処理 ---
                # ターゲットラベルを取得 [B, MaxLabelLength]
                # TODO: targets辞書のキー名を確認・調整
                target_labels = targets.get("labels")
                # TODO: ターゲットのパディングマスクを取得 (例: 0がパディング)
                # target_padding_mask = targets.get("padding_mask") # [B, MaxLabelLength] bool (Trueがパディング)
                # 仮に、EOSトークンIDでパディングされていると仮定
                target_padding_mask = (target_labels == self.eos_token_id)

                if target_labels is None:
                     print("Warning: Missing 'labels' in targets for Transformer decoder training. Returning zero loss.")
                     return {"loss": torch.tensor(0.0, device=images.device, requires_grad=True)}

                batch_size, max_len = target_labels.shape
                device = target_labels.device

                # 1. ターゲット入力の準備
                # BOSトークンを追加: [B, 1+MaxLabelLength]
                bos_tokens = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
                decoder_input_ids = torch.cat([bos_tokens, target_labels[:, :-1]], dim=1) # Teacher Forcing

                # ターゲット埋め込み + 位置エンコーディング
                tgt_emb = self.decoder_embedding(decoder_input_ids) # [B, 1+MaxLabelLength, D]
                pos = torch.arange(0, 1 + max_len -1, device=device).unsqueeze(0) # [1, 1+MaxLabelLength]
                pos_emb = self.decoder_pos_encoding(pos) # [1, 1+MaxLabelLength, D]
                tgt = tgt_emb + pos_emb # [B, 1+MaxLabelLength, D]
                tgt = nn.Dropout(self.config.model.dropout)(tgt) # Dropout適用

                # 2. マスクの作成
                # デコーダ自己注意マスク (未来の情報を見ないように)
                tgt_len = tgt.size(1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device) # [tgt_len, tgt_len]

                # パディングマスク (BOSトークン分を追加)
                # [B, MaxLabelLength] -> [B, 1+MaxLabelLength]
                # BOSトークンはパディングしないので False を追加
                bos_padding = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
                # target_padding_mask は target_labels[:, :-1] に対応するマスクが必要
                # target_labels の EOS/PAD を使う場合、[:-1] のパディングマスクは元のパディングマスクの[:-1]
                tgt_key_padding_mask = torch.cat([bos_padding, target_padding_mask[:, :-1]], dim=1) # [B, 1+MaxLabelLength]

                # エンコーダ出力のパディングマスク (もしあれば)
                # memory_key_padding_mask = ... # [B, N_patches+1] (Trueがパディング)
                # ViTの場合、通常パディングはないので None とする
                memory_key_padding_mask = None

                # 3. Transformerデコーダ実行
                decoder_output = self.transformer_decoder(
                    tgt=tgt,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_mask=None, # 通常エンコーダ出力全体にアテンション
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                ) # [B, 1+MaxLabelLength, D]

                # 4. 出力層適用
                logits = self.decoder_output_layer(decoder_output) # [B, 1+MaxLabelLength, VocabSize-1] (BOS除く)

                # 5. 損失計算 (CrossEntropy)
                # 損失計算のターゲットは元のラベルシーケンス [B, MaxLabelLength]
                # logits は [B, 1+MaxLabelLength, VocabSize-1] なので、形状を合わせる
                # 最初のBOSトークンに対応するlogitは不要
                loss_logits = logits.permute(0, 2, 1) # [B, VocabSize-1, 1+MaxLabelLength]
                loss_targets = target_labels # [B, MaxLabelLength]

                # CrossEntropyLoss は内部で Softmax を計算
                # ignore_index でパディング (EOS/PAD ID) を無視
                loss = F.cross_entropy(
                    loss_logits[:, :, 1:], # BOSに対応するlogitを除外 [B, VocabSize-1, MaxLabelLength]
                    loss_targets,          # [B, MaxLabelLength]
                    ignore_index=self.eos_token_id # EOS/PAD トークンを無視
                )

                return {"loss": loss}

            else:
                # --- 推論時の処理 (自己回帰生成) ---
                inference_strategy = self.config.model.get("inference_strategy", "greedy")
                if inference_strategy == "beam":
                    # Beam Search を実行
                    beam_size = self.config.model.get("beam_size", 5)
                    alpha = self.config.model.get("length_penalty_alpha", 0.0) # デフォルト0 (無効)
                    generated_ids, generated_scores = self._generate_sequence_beam(memory, beam_size, alpha) # list[torch.Tensor], list[float]
                elif inference_strategy == "greedy":
                    # Greedy Search を実行
                    generated_ids = self._generate_sequence_greedy(memory) # list[torch.Tensor] (可変長)
                    # Greedy Search ではシーケンススコアは単純には得られないため、ダミースコア (0.0)
                    generated_scores = [0.0] * len(generated_ids) # list[float]
                else:
                    raise ValueError(f"Unsupported inference_strategy: {inference_strategy}")


                # 生成されたIDシーケンスを返す (DETR形式に合わせるためのダミー値も含む)
                batch_size = images.size(0)
                dummy_boxes = [torch.zeros((0, 4), device=images.device)] * batch_size
                # scores_list はシーケンスごとのスコア (floatのリスト)
                # labels_list はIDテンソルのリスト
                scores_list = generated_scores # list[float]
                labels_list = generated_ids    # list[torch.Tensor]

                # TODO: 生成結果 (labels_list, scores_list) から実際のボックス、スコア、ラベルを生成する処理を追加
                # DETR形式に合わせる場合、scores_list をテンソルのリストに変換する必要があるかもしれない
                # 例: 各シーケンスのスコアを、そのシーケンスの各トークンに割り当てるなど
                # ここではダミーの boxes と、シーケンススコアをそのまま返す (後処理で解釈が必要)
                # (Transformerデコーダの場合、通常はシーケンスのみが重要)
                return {"boxes": dummy_boxes, "scores": scores_list, "labels": labels_list}
        else:
             raise ValueError(f"Unsupported decoder_type in forward pass: {self.decoder_type}")
        # --- 順伝播処理ここまで ---

    def set_epoch(self, epoch: int):
        """現在のエポック数を設定 (動的IoU閾値などで使用)"""
        self.current_epoch = epoch

    # --- 必要に応じてヘルパーメソッドを追加 ---
    def _compute_ctc_loss(self, log_probs: torch.Tensor, targets: dict[str, list[torch.Tensor]]) -> torch.Tensor:
        """
        CTC損失を計算する。

        Args:
            log_probs (torch.Tensor): モデル出力の対数確率 [B, N_patches, NumClasses+1]
            targets (dict[str, list[torch.Tensor]]): ターゲット情報
                - labels (torch.Tensor): ターゲットラベルシーケンス [B, MaxLabelLength]
                - label_lengths (torch.Tensor): 各ターゲットラベルの長さ [B]
                - input_lengths (torch.Tensor): モデル入力シーケンスの長さ (パッチ数) [B]

        Returns:
            torch.Tensor: 計算されたCTC損失 (スカラー)
        """
        # log_probs を CTC損失関数が期待する形式 [N_patches, B, NumClasses+1] に変換
        log_probs_for_loss = log_probs.permute(1, 0, 2)

        # ターゲットラベルと長さを取得
        # TODO: targets辞書のキー名を確認・調整
        target_labels = targets.get("labels")
        target_lengths = targets.get("label_lengths")
        input_lengths = targets.get("input_lengths") # 各バッチ要素の有効なパッチ数を指定

        if target_labels is None or target_lengths is None or input_lengths is None:
            # 訓練時でもターゲットがない場合 (エラーハンドリングまたは警告)
            print("Warning: Missing 'labels', 'label_lengths', or 'input_lengths' in targets for CTC loss calculation. Returning zero loss.")
            return torch.tensor(0.0, device=log_probs.device, requires_grad=True)

        # TODO: input_lengths が None の場合、デフォルト値 (N_patches) を設定
        # if input_lengths is None:
        #     batch_size, num_patches, _ = log_probs.shape
        #     input_lengths = torch.full(size=(batch_size,), fill_value=num_patches, dtype=torch.long, device=log_probs.device)


        # CTC損失を計算
        # zero_infinity=True は勾配がinf/nanになるのを防ぐため
        loss = F.ctc_loss(
            log_probs=log_probs_for_loss,
            targets=target_labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=self.config.model.num_classes, # blank token はクラス数のインデックス
            reduction='mean', # バッチ全体で平均を取る
            zero_infinity=True
        )
        return loss

    def _decode_ctc(self, log_probs: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        CTC出力 (log_probs) をデコードする (Greedy Decoding)。

        Args:
            log_probs (torch.Tensor): モデル出力の対数確率 [B, N_patches, NumClasses+1]

        Returns:
            tuple[torch.Tensor | None, torch.Tensor | None]:
                - デコードされたIDシーケンスのリスト (パディングされている可能性あり)
                - 各シーケンスのスコア (対数確率の合計など、簡易的なもの)
                # TODO: より洗練されたスコアリングや、バッチ処理に対応した形式を検討
        """
        batch_size = log_probs.shape[0]
        blank_id = self.config.model.num_classes # blank token ID

        decoded_ids_list = []
        decoded_scores_list = []

        # バッチ内の各サンプルに対してGreedy Decodingを実行
        for i in range(batch_size):
            sample_log_probs = log_probs[i] # [N_patches, NumClasses+1]
            # 各タイムステップで最も確率の高いクラスIDを取得
            best_path = torch.argmax(sample_log_probs, dim=-1) # [N_patches]

            # 連続する重複IDとblank IDを削除
            hyp = []
            scores = [] # 各非blank文字の対数確率を保持 (簡易スコア用)
            prev_id = blank_id
            for t in range(best_path.size(0)):
                current_id = best_path[t].item()
                if current_id != blank_id and current_id != prev_id:
                    hyp.append(current_id)
                    scores.append(sample_log_probs[t, current_id].item()) # 対数確率
                prev_id = current_id

            # デコード結果をテンソルに変換
            decoded_ids = torch.tensor(hyp, dtype=torch.long, device=log_probs.device)
            # 簡易的なスコアとして対数確率の平均を計算 (必要に応じて変更)
            # スコアが空の場合は 0 とする
            sequence_score = torch.tensor(scores, device=log_probs.device).mean() if scores else torch.tensor(0.0, device=log_probs.device)

            decoded_ids_list.append(decoded_ids)
            decoded_scores_list.append(sequence_score) # 各シーケンスに1つのスコア

        # TODO: バッチ処理に適した形式で返す (例: パディングしてテンソル化)
        # 現状はリストのまま返す (後続処理でパディングなどを想定)
        # スコアもリストで返す
        return decoded_ids_list, decoded_scores_list


    def _generate_sequence_greedy(self, memory: torch.Tensor) -> list[torch.Tensor]:
        """
        Transformerデコーダを用いてGreedy Searchでシーケンスを生成する。

        Args:
            memory (torch.Tensor): エンコーダ出力 [B, N_mem, D]

        Returns:
            list[torch.Tensor]: 生成されたIDシーケンスのリスト (各要素は可変長、BOS/EOS含まず)
        """
        batch_size = memory.size(0)
        device = memory.device
        max_len = self.config.model.max_target_length

        # 各バッチ要素の生成済みシーケンスを保持 (BOSトークンで初期化)
        generated_sequences = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        # 各バッチ要素がEOSに到達したかを示すフラグ
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1): # 最大長-1ステップ生成 (BOS含むため)
            current_len = generated_sequences.size(1)

            # 1. デコーダ入力準備
            tgt_emb = self.decoder_embedding(generated_sequences) # [B, current_len, D]
            pos = torch.arange(0, current_len, device=device).unsqueeze(0) # [1, current_len]
            pos_emb = self.decoder_pos_encoding(pos) # [1, current_len, D]
            tgt = tgt_emb + pos_emb # [B, current_len, D]
            # Note: 推論時はDropoutを適用しないのが一般的

            # 2. マスク作成
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_len, device=device) # [current_len, current_len]
            # 推論時はパディングはない想定 (tgt_key_padding_mask=None)
            # memory_key_padding_mask も None と仮定

            # 3. デコーダ実行
            decoder_output = self.transformer_decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None
            ) # [B, current_len, D]

            # 4. 次のトークン予測
            # 最後のタイムステップの出力のみ使用
            last_output = decoder_output[:, -1, :] # [B, D]
            logits = self.decoder_output_layer(last_output) # [B, VocabSize-1]
            next_token_ids = torch.argmax(logits, dim=-1) # [B]

            # 5. シーケンス更新 & 終了判定
            # EOSに到達したサンプルはEOSを、そうでなければ予測トークンを追加
            # 予測IDは 0 ~ num_classes なので、そのまま使える
            next_token_ids_with_eos = torch.where(finished, self.eos_token_id, next_token_ids)
            generated_sequences = torch.cat([generated_sequences, next_token_ids_with_eos.unsqueeze(1)], dim=1)

            # 新たにEOSに到達したかチェック
            just_finished = (next_token_ids_with_eos == self.eos_token_id)
            finished = finished | just_finished

            # 全てのバッチ要素が終了したらループを抜ける
            if finished.all():
                break

        # BOSを除き、各シーケンスのEOS以降を除去してリストで返す
        output_list = []
        for i in range(batch_size):
            seq = generated_sequences[i, 1:] # BOS除去
            eos_indices = (seq == self.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                first_eos_idx = eos_indices[0]
                output_list.append(seq[:first_eos_idx])
            else: # EOSが出なかった場合
                output_list.append(seq)

        return output_list


    def _generate_sequence_beam(self, memory: torch.Tensor, beam_size: int, alpha: float = 0.0) -> tuple[list[torch.Tensor], list[float]]:
        """
        Transformerデコーダを用いてBeam Searchでシーケンスを生成する。
        Length Normalization を適用可能。

        Args:
            memory (torch.Tensor): エンコーダ出力 [B, N_mem, D]
            beam_size (int): ビームサイズ
            alpha (float): Length Normalization の指数 (0で無効)

        Returns:
            tuple[list[torch.Tensor], list[float]]:
                - 生成されたIDシーケンスのリスト (各要素は可変長、BOS/EOS含まず)
                - 各シーケンスに対応する正規化されたスコア (対数確率) のリスト
        """
        batch_size = memory.size(0)
        device = memory.device
        max_len = self.config.model.max_target_length

        final_sequences = [] # 各バッチ要素の最終的な生成シーケンス
        final_scores = []    # 各バッチ要素の最終的なスコア

        # --- バッチ内の各サンプルに対してBeam Searchを実行 ---
        for i in range(batch_size):
            # このサンプルのエンコーダ出力を取得
            # memory_i: [N_mem, D]
            memory_i = memory[i]

            # 各ビームの状態: (累積対数確率スコア, 現在のシーケンス Tensor[1, current_len], 正規化用スコア)
            # 初期状態: BOSトークンのみ、スコア0
            # 正規化用スコアは heapq での比較に使用
            initial_beam = (0.0, torch.full((1, 1), self.bos_token_id, dtype=torch.long, device=device))
            active_beams = [initial_beam] # list[tuple(score, sequence)]
            finished_beams = [] # list[tuple(normalized_score, score, sequence)]

            # --- デコードステップ ---
            for step in range(max_len - 1):
                if not active_beams: # アクティブなビームがなくなったら終了
                    break

                # 現在のステップで生成される候補: (新しいスコア, 新しいシーケンス)
                candidates = []
                # 各アクティブビームを展開
                for current_score, current_seq in active_beams:
                    current_len = current_seq.size(1)

                    # 1. デコーダ入力準備 (単一ビームに対して)
                    tgt_emb = self.decoder_embedding(current_seq) # [1, current_len, D]
                    pos = torch.arange(0, current_len, device=device).unsqueeze(0) # [1, current_len]
                    pos_emb = self.decoder_pos_encoding(pos) # [1, current_len, D]
                    tgt = tgt_emb + pos_emb # [1, current_len, D]

                    # 2. マスク作成
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_len, device=device)

                    # 3. デコーダ実行 (単一ビーム、単一メモリ)
                    # memory_i を [1, N_mem, D] に拡張
                    # torch.no_grad() で推論時の勾配計算を無効化
                    with torch.no_grad():
                        decoder_output = self.transformer_decoder(
                            tgt=tgt,
                            memory=memory_i.unsqueeze(0),
                            tgt_mask=tgt_mask,
                            memory_mask=None,
                            tgt_key_padding_mask=None,
                            memory_key_padding_mask=None
                        ) # [1, current_len, D]

                    # 4. 次のトークンの対数確率を取得
                    last_output = decoder_output[:, -1, :] # [1, D]
                    logits = self.decoder_output_layer(last_output) # [1, VocabSize-1]
                    log_probs = F.log_softmax(logits, dim=-1) # [1, VocabSize-1]

                    # 5. 上位k個の候補トークンを選択 (k=beam_size)
                    k = beam_size # シンプルに常にbeam_size個生成
                    topk_log_probs, topk_indices = torch.topk(log_probs, k, dim=-1) # [1, k]

                    # 6. 候補ビームを生成
                    for j in range(topk_indices.size(1)):
                        next_token_id = topk_indices[0, j].unsqueeze(0).unsqueeze(0) # [1, 1]
                        token_log_prob = topk_log_probs[0, j].item()

                        new_seq = torch.cat([current_seq, next_token_id], dim=1) # [1, current_len+1]
                        new_score = current_score + token_log_prob

                        # 候補リストに追加
                        candidates.append((new_score, new_seq))

                # --- 候補の中から次のビームを選択 ---
                # スコアでソートし、上位 beam_size 個を選択 (heapq を使用)
                # candidates: list[tuple(score, sequence)]
                # heapq.nlargest はスコア (タプルの最初の要素) で比較する
                next_beams_unfiltered = heapq.nlargest(beam_size, candidates, key=lambda x: x[0])

                # 完了したビームとアクティブなビームに振り分け
                active_beams = []
                newly_finished = []
                for score, seq in next_beams_unfiltered:
                    seq_len = seq.size(1) # BOSを含む長さ
                    normalized_score = score
                    if alpha > 0:
                        # Length Normalization (Google NMT style)
                        # 長さはBOSを含まない実際の生成長 (seq_len - 1) で計算することも考えられる
                        # ここでは seq_len を使う (実装による)
                        length_penalty = ((5.0 + seq_len) / 6.0) ** alpha
                        # length_penalty = seq_len ** alpha # よりシンプルな形式
                        if length_penalty > 0: # ゼロ除算回避
                            normalized_score = score / length_penalty
                        else:
                            normalized_score = -float('inf') # ペナルティが0の場合はスコアを最低に

                    if seq[0, -1].item() == self.eos_token_id:
                        # EOSで終了したビーム
                        newly_finished.append((normalized_score, score, seq))
                    elif step == max_len - 2: # 最大長に達した場合も完了扱い
                         newly_finished.append((normalized_score, score, seq))
                    else:
                        # まだアクティブなビーム (スコアとシーケンスのみ保持)
                        active_beams.append((score, seq))

                # 新しく完了したビームを finished_beams に追加し、正規化スコアで上位 beam_size 個を保持
                finished_beams.extend(newly_finished)
                # heapq.nlargest はタプルの最初の要素 (normalized_score) で比較する
                finished_beams = heapq.nlargest(beam_size, finished_beams, key=lambda x: x[0])

                # 完了ビームが beam_size 個に達したら、アクティブビームの探索を打ち切るか検討
                # ここでは、アクティブビームが残っている限り探索を続ける

            # --- サンプルi の最終結果を選択 ---
            # 完了したビームがなければ、アクティブなビームから最良のものを選択
            if not finished_beams:
                if active_beams: # EOSに到達せず最大長に達した場合
                    # アクティブビームにも Length Normalization を適用して比較
                    temp_finished = []
                    for score, seq in active_beams:
                        seq_len = seq.size(1)
                        normalized_score = score
                        if alpha > 0:
                            length_penalty = ((5.0 + seq_len) / 6.0) ** alpha
                            if length_penalty > 0:
                                normalized_score = score / length_penalty
                            else:
                                normalized_score = -float('inf')
                        temp_finished.append((normalized_score, score, seq))
                    finished_beams = heapq.nlargest(beam_size, temp_finished, key=lambda x: x[0])
                else: # レアケース: 初期状態から進めなかった場合
                     # ダミーの完了ビームを追加
                     finished_beams.append((-float('inf'), -float('inf'), torch.tensor([[self.bos_token_id]], dtype=torch.long, device=device)))


            # 完了ビームの中から最も正規化スコアの高いものを選択
            # finished_beams は正規化スコアでソートされている
            best_normalized_score, best_raw_score, best_seq = finished_beams[0]

            # BOSを除き、EOSがあればそれも除く
            final_seq = best_seq[0, 1:] # BOS除去
            eos_indices = (final_seq == self.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                first_eos_idx = eos_indices[0]
                final_seq = final_seq[:first_eos_idx]

            final_sequences.append(final_seq)
            final_scores.append(best_normalized_score) # 正規化スコアを返す

        return final_sequences, final_scores


    # TODO: Transformerデコーダ用のヘルパーメソッド (Beam Searchなど)
    # --- ヘルパーメソッドここまで ---

# %%
if __name__ == "__main__":
    # テスト用の簡易的な設定
    import yaml
    from utils.util import EasyDict
    import pathlib

    project_root = pathlib.Path(__file__).resolve().parent.parent.parent

    with open(f"{project_root}/config/model/character_detection.yaml", "r") as f:
        config = EasyDict(yaml.safe_load(f))
    model = CSAViTModel(config)
    model.to(config.device)
    print(model)
    # テスト用のダミー入力
    images = torch.randn(2, 3, 224, 224, device=config.device) # [B, C, H, W]
    targets = {
        "labels": torch.randint(0, 100, (2, 10)), # [B, MaxLabelLength]
        "label_lengths": torch.tensor([10, 8]), # [B]
        "input_lengths": torch.tensor([20, 20]) # [B]
    }
    outputs = model(images, targets)
    print(outputs)
    # 出力の確認

# %%
