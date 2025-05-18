import requests
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

# 事前学習済みモデルの指定
encoder_checkpoint = "baseplate/vit-gpt2-image-captioning"  # エンコーダとデコーダがセットになったモデルの例
decoder_checkpoint = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250511_192051/final_model"  # 通常はエンコーダとデコーダで適切なものを指定
image_processor_checkpoint = "baseplate/vit-gpt2-image-captioning"
tokenizer_checkpoint = "experiments/pretrain_language_model/roberta-small-japanese-aozora-char/20250511_192051/final_model"


# モデル、プロセッサ、トークナイザのロード
model = VisionEncoderDecoderModel.from_pretrained(encoder_checkpoint)  # この例では統合モデルを使用
image_processor = ViTImageProcessor.from_pretrained(image_processor_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

# 画像の準備 (例としてURLから取得)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

# キャプション生成
generated_ids = model.generate(pixel_values)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
