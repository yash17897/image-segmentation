
import torch
from model import UNet
from PIL import Image
import torchvision.transforms as transforms

def inference(image_path, model_path, output_path='output.png'):
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).cuda()

    model = UNet(num_classes=20).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    out_img = Image.fromarray(prediction.astype('uint8'))
    out_img.save(output_path)

if __name__ == "__main__":
    inference("test_image.png", "unet_epoch_10.pth")
