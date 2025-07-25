import lightning as L


class EMACallback(L.Callback):
    """
    Exponential Moving Average (EMA) callback for model parameters

    Usage:
    ema_callback = EMACallback(decay=0.9999, target_module_names=["generator"])
    """

    def __init__(self, decay: float = 0.9999, target_module_names: str | list[str] | None = None):
        """
        Args:
            decay: EMAの減衰率
            target_module_names: EMAを適用するモジュール名（複数可）。Noneの場合はpl_module全体。
        """
        super().__init__()
        if not (0.0 <= decay <= 1.0):
            raise ValueError("Decay must be between 0 and 1.")
        self.decay = decay
        self.target_module_names = [target_module_names] if isinstance(target_module_names, str) else target_module_names
        self.shadow = {}
        self.backup = {}

    def _get_target_parameters(self, pl_module: L.LightningModule):
        """EMAを適用するパラメータを取得"""
        if self.target_module_names is None:
            # モジュール指定がない場合はpl_module全体のパラメータを対象
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    yield name, param
        else:
            # 指定されたモジュールのパラメータを対象
            for module_name in self.target_module_names:
                try:
                    module = getattr(pl_module, module_name)
                    for name, param in module.named_parameters():
                        if param.requires_grad:
                            # パラメータ名にプレフィックスを追加して一意にする
                            yield f"{module_name}.{name}", param
                except AttributeError:
                    print(f"Warning: Module '{module_name}' not found in LightningModule. Skipping EMA for this module.")

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """訓練開始時にEMAパラメータを初期化"""
        self.shadow.clear()
        for name, param in self._get_target_parameters(pl_module):
            self.shadow[name] = param.data.clone()

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx):
        """各バッチ後にEMAを更新"""
        for name, param in self._get_target_parameters(pl_module):
            if name in self.shadow:
                # self.shadow[name] が on_load_checkpoint または on_fit_start で
                # pl_module のデバイス上に移動済みであることを想定しています。
                # よりメモリ効率の良いインプレース操作を使用。
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """検証開始時にモデルのパラメータをEMAに置き換え"""
        self.backup.clear()
        for name, param in self._get_target_parameters(pl_module):
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """検証終了時にモデルのパラメータを元に戻す"""
        for name, param in self._get_target_parameters(pl_module):
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def on_save_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict) -> dict:
        """チェックポイント保存時にEMAの状態も保存"""
        # DDP環境では、通常グローバルランク0のプロセスのみがチェックポイントを保存します。
        # チェックポイントのポータビリティのため、保存前にEMAパラメータをCPUに移動します。
        if trainer.is_global_zero:
            cpu_shadow = {name: s.cpu() for name, s in self.shadow.items()}
            checkpoint["ema_shadow"] = cpu_shadow
        return checkpoint

    def on_load_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict):
        """チェックポイント読み込み時にEMAの状態を復元"""
        if "ema_shadow" in checkpoint:
            loaded_shadow = checkpoint["ema_shadow"]
            self.shadow.clear() # 既存のシャドウをクリア
            # ここが最も重要な修正点です。
            # ロードされたテンソルを、現在の pl_module が存在するデバイスに明示的に移動させます。
            # pl_module.device は現在のDDPプロセスが使用しているデバイス (例: cuda:0, cuda:1) を返します。
            for name, shadow_tensor in loaded_shadow.items():
                self.shadow[name] = shadow_tensor.to(pl_module.device)
        else:
            print("Warning: EMA shadow parameters not found in checkpoint. Initializing EMA from current model parameters.")
            # shadow がチェックポイントにない場合は、現在のモデルパラメータで初期化し直します。
            # この場合、on_fit_start が呼び出され、テンソルは正しいデバイス上に初期化されます。
            self.on_fit_start(trainer, pl_module)
