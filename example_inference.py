import torch
from enhanced_tsf_net_model import TSFNet, load_model, predict_video
import argparse

parser = argparse.ArgumentParser(description='TSF-Net Deepfake Video Inference')
parser.add_argument('--video_path', type=str, required=True, help='Path to the input video')
parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(args.model_path, device=device)

print(f"Processing video: {args.video_path}")
result = predict_video(model, args.video_path, device=device)
print("Prediction Result:")
print(result)
