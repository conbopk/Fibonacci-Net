import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import argparse
import os


class Avg2MaxPooling(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        """Novel Avg-2Max Pooling layer (as per paper)"""
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        # Emphasize edges by subtracting twice the max pooled value from avg pooled value
        return self.avg_pool(x) - (self.max_pool(x) + self.max_pool(x))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution with ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=kernel_size // 2,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class FibonacciNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1):
        super().__init__()

        # Block 1 (21 filters)
        self.block1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 21, kernel_size=3, padding=1),
            nn.BatchNorm2d(21),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # -> 21 x 112 x 112
        )

        # Block 2 (34 filters)
        self.block2 = nn.Sequential(
            nn.Conv2d(21, 34, kernel_size=3, padding=1),
            nn.BatchNorm2d(34)
        )
        self.block2_relu = nn.ReLU()  # Separate for skip connection
        self.block2_pool = nn.MaxPool2d(kernel_size=2)  # -> 34 x 56 x 56

        # Block 3 (55 filters)
        self.block3 = nn.Sequential(
            nn.Conv2d(34, 55, kernel_size=3, padding=1),
            nn.BatchNorm2d(55)
        )
        self.block3_relu = nn.ReLU()  # Separate for skip connection
        self.block3_pool = nn.MaxPool2d(kernel_size=2)  # -> 55 x 28 x 28

        # PCB1: Block 2 -> Block 4
        # We need to go from 56x56 to 14x14, so we need one more pooling operation
        self.pcb1 = nn.Sequential(
            nn.Conv2d(34, 24, kernel_size=3, padding=1),  # 24 x 112 x 112
            Avg2MaxPooling(),  # 24 x 56 x 56
            nn.Conv2d(24, 24, kernel_size=3, padding=1),  # 24 x 56 x 56
            Avg2MaxPooling(),  # 24 x 28 x 28
            nn.Conv2d(24, 24, kernel_size=3, padding=1),  # 24 x 28 x 28
            Avg2MaxPooling()  # 24 x 14 x 14
        )

        # Block 4 (89 filters)
        self.block4 = nn.Sequential(
            nn.Conv2d(55, 89, kernel_size=3, padding=1),
            nn.BatchNorm2d(89),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # -> 89 x 14 x 14
        )

        # PCB2: Block 3 -> Block 5
        # We need to go from 28x28 to 7x7, so ensure proper downsampling
        self.pcb2 = nn.Sequential(
            nn.Conv2d(55, 24, kernel_size=3, padding=1),  # 24 x 56 x 56
            Avg2MaxPooling(),  # 24 x 28 x 28
            nn.Conv2d(24, 24, kernel_size=3, padding=1),  # 24 x 28 x 28
            Avg2MaxPooling(),  # 24 x 14 x 14
            nn.Conv2d(24, 24, kernel_size=3, padding=1),  # 24 x 14 x 14
            Avg2MaxPooling()  # 24 x 7 x 7
        )

        # Block 5 (144 filters)
        self.block5 = nn.Sequential(
            nn.Conv2d(89 + 24, 144, kernel_size=3, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # -> 144 x 7 x 7
        )

        # Block 6 (233 filters, DWSC)
        self.block6 = DepthwiseSeparableConv(144 + 24, 233)  # 233 x 7 x 7

        # Block 7 (377 filters, DWSC)
        self.block7 = DepthwiseSeparableConv(233, 377)  # 377 x 7 x 7

        # Output
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(377, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Block 1
        x = self.block1(x)  # 21 x 112 x 112

        # Block 2
        x = self.block2(x)
        x2 = self.block2_relu(x)  # Save for pcb1     34 x 112 x 112
        x = self.block2_pool(x2)  # 34 x 56 x 56

        # Block 3
        x = self.block3(x)
        x3 = self.block3_relu(x)  # Save for pcb1     55 x 56 x 56
        x = self.block3_pool(x)  # 55 x 28 x 28

        # Block 4
        x = self.block4(x)  # 89 x 14 x 14

        # PCB1 processing
        pcb1 = self.pcb1(x2)  # Using x2 from block 2
        # 24 x 14 x 14

        # Concat block4 and pcb1
        x = torch.cat([x, pcb1], dim=1)  # (89 + 24) x 14 x 14

        # Block 5
        x = self.block5(x)  # 144 x 7 x 7

        # PCB2 processing
        pcb2 = self.pcb2(x3)  # Using x3 from block 3
        # 24 x 7 x 7

        # Concat Block 5 and pcb2
        x = torch.cat([x, pcb2], dim=1)  # (144 + 24) x 7 x 7

        # Block 6 & 7
        x = self.block6(x)  # 233 x 7 x 7
        x = self.block7(x)  # 377 x 7 x 7

        # Output
        x = self.global_pool(x)  # (377,1,1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


class BrainTumorClassifier:
    def __init__(self, model_path='fibonacci_net.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = FibonacciNet(input_shape=(3, 224, 224), num_classes=1)

        # Load the trained model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """
        Predict if the input image contains a brain tumor

        Args:
            image_path (str): Path to the input MRI image

        Returns:
            tuple: (prediction class, probability)
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and preprocess the image
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
            img_tensor = img_tensor.to(self.device)
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

        # Make prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probability = output.item()
            prediction = 1 if probability >= 0.5 else 0

        result = {
            "prediction": "Tumor" if prediction == 1 else "Healthy",
            "probability": probability,
            "confidence": probability if prediction == 1 else 1 - probability
        }

        return result


def main():
    parser = argparse.ArgumentParser(description='Brain Tumor MRI Classifier')
    parser.add_argument('--image', type=str, required=True, help='Path to the input MRI image')
    parser.add_argument('--model', type=str, default='fibonacci_net.pth', help='Path to the trained model')
    args = parser.parse_args()

    try:
        # Create classifier instance
        classifier = BrainTumorClassifier(model_path=args.model)

        # Predict
        result = classifier.predict(args.image)

        # Print results
        print("\n===== Brain Tumor Classification Result =====")
        print(f"Image: {args.image}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence'] * 100:.2f}%)")
        print("===========================================\n")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()