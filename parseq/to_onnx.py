from strhub.models.utils import load_from_checkpoint
import torch

# To ONNX
device = "cpu"
ckpt_path = "outputs/parseq/2024-10-22_15-25-44/checkpoints/best.ckpt"
onnx_path = "tib-parseq-norbu+gbooks.onnx"
fname = 'demo_images/ic13_word_256.png'


parseq = load_from_checkpoint(ckpt_path)
parseq.refine_iters = 0
parseq.decode_ar = False
parseq = parseq.to(device).eval()



dummy_input = torch.rand(1, 3, *parseq.hparams.img_size)  # (1, 3, 32, 128) by default

# To ONNX
parseq.to_onnx(onnx_path, dummy_input, opset_version=14)  # opset v14 or newer is required

