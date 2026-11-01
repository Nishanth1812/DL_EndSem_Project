import zipfile
from PIL import Image
import io

z = zipfile.ZipFile('results/run_40gb_CNN_LSTM/results.zip')
pred_bytes = z.read('results/out/prediction.png')
im = Image.open(io.BytesIO(pred_bytes))
print('Friend prediction.png format:', im.format, 'size:', im.size)

loss_bytes = z.read('results/out/loss_curve.png')
im_loss = Image.open(io.BytesIO(loss_bytes))
print('Friend loss_curve.png format:', im_loss.format, 'size:', im_loss.size)
