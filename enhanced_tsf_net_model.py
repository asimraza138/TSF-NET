import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import math
import numpy as np
import cv2
from facenet_pytorch import MTCNN
from typing import List, Tuple, Dict, Optional, Union

class FacePreprocessor:
    """
    Preprocessing module for face detection, alignment, and augmentation
    for deepfake detection tasks.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (299, 299),
                 use_mtcnn: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the face preprocessing module.
        
        Args:
            target_size: Target size for face images (height, width)
            use_mtcnn: Whether to use MTCNN for face detection
            device: Device to use for processing
        """
        self.target_size = target_size
        self.device = device
        self.use_mtcnn = use_mtcnn
        
        # Initialize MTCNN face detector
        if use_mtcnn:
            self.face_detector = MTCNN(
                image_size=target_size[0],
                margin=20,
                keep_all=False,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.9],
                factor=0.709,
                post_process=True,
                device=device
            )
        
        # Define transformations for training
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define transformations for validation/testing
        self.test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_face(self, image: np.ndarray, is_training: bool = False) -> Optional[torch.Tensor]:
        """
        Extract face from an image using MTCNN or a fallback method.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            is_training: Whether the extraction is for training (applies augmentation)
            
        Returns:
            Preprocessed face tensor or None if no face detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_mtcnn:
            try:
                # Detect face using MTCNN
                face = self.face_detector(image_rgb)
                
                if face is None:
                    # Fallback to traditional face detection if MTCNN fails
                    return self._extract_face_cv2(image_rgb, is_training)
                
                # Apply transformations
                if is_training:
                    face = self.train_transforms(face.cpu().numpy().transpose(1, 2, 0))
                else:
                    face = self.test_transforms(face.cpu().numpy().transpose(1, 2, 0))
                
                return face
                
            except Exception as e:
                print(f"MTCNN face detection failed: {e}")
                # Fallback to traditional face detection
                return self._extract_face_cv2(image_rgb, is_training)
        else:
            # Use traditional face detection
            return self._extract_face_cv2(image_rgb, is_training)
    
    def _extract_face_cv2(self, image_rgb: np.ndarray, is_training: bool) -> Optional[torch.Tensor]:
        """
        Extract face using OpenCV's Haar cascade classifier as a fallback method.
        
        Args:
            image_rgb: Input RGB image
            is_training: Whether the extraction is for training
            
        Returns:
            Preprocessed face tensor or None if no face detected
        """
        # Load OpenCV's face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # If no face detected, use the whole image
            face_img = cv2.resize(image_rgb, self.target_size)
        else:
            # Extract the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Add margin
            margin = int(0.2 * max(w, h))
            x_min = max(0, x - margin)
            y_min = max(0, y - margin)
            x_max = min(image_rgb.shape[1], x + w + margin)
            y_max = min(image_rgb.shape[0], y + h + margin)
            
            # Extract face region
            face_img = image_rgb[y_min:y_max, x_min:x_max]
            face_img = cv2.resize(face_img, self.target_size)
        
        # Apply transformations
        if is_training:
            face_tensor = self.train_transforms(face_img)
        else:
            face_tensor = self.test_transforms(face_img)
        
        return face_tensor
    
    def process_video(self, video_path: str, max_frames: int = 32, stride: int = 1, is_training: bool = False) -> List[torch.Tensor]:
        """
        Process video by extracting faces from frames.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            stride: Frame sampling stride
            is_training: Whether the extraction is for training
            
        Returns:
            List of preprocessed face tensors
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        frame_count = 0
        
        while len(frames) < max_frames:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process every 'stride' frames
            if frame_count % stride == 0:
                # Extract face
                face = self.extract_face(frame, is_training)
                
                if face is not None:
                    frames.append(face)
            
            frame_count += 1
        
        cap.release()
        
        # If no faces were detected, return empty list
        if len(frames) == 0:
            return []
        
        # If fewer frames than max_frames, duplicate the last frame
        while len(frames) < max_frames:
            frames.append(frames[-1])
        
        # Convert list of tensors to a single tensor
        return frames[:max_frames]


class FeatureExtractor:
    """
    Feature extraction module that combines EfficientNetV2L and XceptionNet
    for extracting complementary spatial features.
    """
    
    def __init__(self, 
                 pretrained: bool = True, 
                 freeze_initial: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the feature extraction module.
        
        Args:
            pretrained: Whether to use pretrained weights
            freeze_initial: Whether to freeze weights for initial training
            device: Device to use for processing
        """
        self.device = device
        
        # Initialize EfficientNetV2L feature extractor
        self.efficient_extractor = EfficientNetV2LFeatureExtractor(
            pretrained=pretrained,
            freeze_initial=freeze_initial
        ).to(device)
        
        # Initialize XceptionNet feature extractor
        self.xception_extractor = ModifiedXceptionNet(
            pretrained=pretrained,
            freeze_initial=freeze_initial
        ).to(device)
    
    def extract_features(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from input frames using both networks.
        
        Args:
            frames: Input frames tensor of shape [batch_size, seq_len, channels, height, width]
            
        Returns:
            Tuple of (EfficientNetV2L features, XceptionNet features)
        """
        batch_size, seq_len, channels, height, width = frames.shape
        
        # Reshape for batch processing
        frames_reshaped = frames.view(-1, channels, height, width)
        
        # Extract features
        efficient_features = self.efficient_extractor(frames_reshaped)
        xception_features = self.xception_extractor(frames_reshaped)
        
        # Reshape back to sequence format
        efficient_features = efficient_features.view(batch_size, seq_len, -1)
        xception_features = xception_features.view(batch_size, seq_len, -1)
        
        return efficient_features, xception_features


"""
TempoSpatialFusion Network (TSF-Net) for Deepfake Detection

This implementation includes:
1. Dual-CNN feature extraction (EfficientNetV2L and XceptionNet)
2. Cross-Modal Attention Fusion (CMAF)
3. Temporal Inconsistency Attention Module (TIAM)
4. Artifact-Aware Loss Function (AALF)
5. Adaptive Computational Scaling (ACS)
"""

class EfficientNetV2LFeatureExtractor(nn.Module):
    """EfficientNetV2L feature extractor with custom modifications for deepfake detection."""
    
    def __init__(self, pretrained: bool = True, freeze_initial: bool = True):
        """
        Initialize the EfficientNetV2L feature extractor.
        
        Args:
            pretrained: Whether to use pretrained weights
            freeze_initial: Whether to freeze weights for initial training
        """
        super(EfficientNetV2LFeatureExtractor, self).__init__()
        
        # Load pretrained EfficientNetV2L
        self.model = models.efficientnet_v2_l(pretrained=pretrained)
        
        # Remove the classification head
        self.model.classifier = nn.Identity()
        
        # Feature dimension
        self.feature_dim = 1280
        
        # Freeze initial layers if specified
        if freeze_initial:
            for param in list(self.model.parameters())[:-20]:  # Freeze all but last few layers
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input frames.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Feature tensor of shape [batch_size, feature_dim]
        """
        features = self.model(x)
        return features


class ModifiedXceptionNet(nn.Module):
    """Modified XceptionNet with custom adaptations for deepfake detection."""
    
    def __init__(self, pretrained: bool = True, freeze_initial: bool = True):
        """
        Initialize the modified XceptionNet.
        
        Args:
            pretrained: Whether to use pretrained weights
            freeze_initial: Whether to freeze weights for initial training
        """
        super(ModifiedXceptionNet, self).__init__()
        
        # Load pretrained XceptionNet
        self.model = models.xception(pretrained=pretrained)
        
        # Remove the classification head
        self.model.fc = nn.Identity()
        
        # Feature dimension
        self.feature_dim = 2048
        
        # Modify middle flow to repeat 12 times instead of 8
        middle_flow = self.model.blocks[4:8]
        extended_middle_flow = middle_flow * 3  # 12 blocks total
        
        # Reconstruct the model with extended middle flow
        self.model.blocks = nn.ModuleList([
            *self.model.blocks[:4],  # Entry flow
            *extended_middle_flow,   # Extended middle flow
            *self.model.blocks[8:]   # Exit flow
        ])
        
        # Add residual connections between blocks
        self.residual_connections = nn.ModuleList([
            nn.Conv2d(728, 728, kernel_size=1, stride=1, bias=False)
            for _ in range(len(extended_middle_flow))
        ])
        
        # Freeze initial layers if specified
        if freeze_initial:
            for param in list(self.model.parameters())[:-30]:  # Freeze all but last few layers
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input frames with added residual connections.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Feature tensor of shape [batch_size, feature_dim]
        """
        # Entry flow
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        
        # Process through blocks with residual connections for middle flow
        for i, block in enumerate(self.model.blocks):
            # Apply residual connections for middle flow blocks
            if 4 <= i < 16:  # Middle flow blocks
                residual = self.residual_connections[i-4](x)
                x = block(x) + residual
            else:
                x = block(x)
        
        # Final processing
        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        
        x = self.model.conv4(x)
        x = self.model.bn4(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        return x


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Fusion (CMAF) mechanism that dynamically integrates 
    complementary features from EfficientNetV2L and XceptionNet.
    """
    
    def __init__(self, 
                 efficient_dim: int = 1280, 
                 xception_dim: int = 2048, 
                 output_dim: int = 1024, 
                 num_heads: int = 8):
        """
        Initialize the Cross-Modal Attention Fusion module.
        
        Args:
            efficient_dim: Dimension of EfficientNetV2L features
            xception_dim: Dimension of XceptionNet features
            output_dim: Dimension of output features
            num_heads: Number of attention heads
        """
        super(CrossModalAttentionFusion, self).__init__()
        
        self.efficient_dim = efficient_dim
        self.xception_dim = xception_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        # Projection matrices for each head
        self.query_projections = nn.ModuleList([
            nn.Linear(efficient_dim, self.head_dim) for _ in range(num_heads)
        ])
        
        self.key_projections = nn.ModuleList([
            nn.Linear(xception_dim, self.head_dim) for _ in range(num_heads)
        ])
        
        self.value_projections = nn.ModuleList([
            nn.Linear(xception_dim, self.head_dim) for _ in range(num_heads)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(num_heads * self.head_dim, output_dim)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(efficient_dim)
        self.layer_norm2 = nn.LayerNorm(xception_dim)
        self.layer_norm3 = nn.LayerNorm(output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, efficient_features: torch.Tensor, xception_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-modal attention fusion to integrate features.
        
        Args:
            efficient_features: Features from EfficientNetV2L [batch_size, efficient_dim]
            xception_features: Features from XceptionNet [batch_size, xception_dim]
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        batch_size = efficient_features.shape[0]
        
        # Apply layer normalization
        efficient_features = self.layer_norm1(efficient_features)
        xception_features = self.layer_norm2(xception_features)
        
        # Process each attention head
        head_outputs = []
        for i in range(self.num_heads):
            # Project queries from EfficientNetV2L features
            queries = self.query_projections[i](efficient_features)  # [batch_size, head_dim]
            
            # Project keys and values from XceptionNet features
            keys = self.key_projections[i](xception_features)  # [batch_size, head_dim]
            values = self.value_projections[i](xception_features)  # [batch_size, head_dim]
            
            # Reshape for attention computation
            queries = queries.unsqueeze(1)  # [batch_size, 1, head_dim]
            keys = keys.unsqueeze(1)  # [batch_size, 1, head_dim]
            values = values.unsqueeze(1)  # [batch_size, 1, head_dim]
            
            # Compute attention scores
            attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # [batch_size, 1, 1]
            attention_scores = attention_scores / math.sqrt(self.head_dim)
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1, 1]
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention weights to values
            head_output = torch.matmul(attention_weights, values)  # [batch_size, 1, head_dim]
            head_output = head_output.squeeze(1)  # [batch_size, head_dim]
            
            head_outputs.append(head_output)
        
        # Concatenate outputs from all heads
        multi_head_output = torch.cat(head_outputs, dim=1)  # [batch_size, num_heads*head_dim]
        
        # Apply output projection
        fused_features = self.output_projection(multi_head_output)  # [batch_size, output_dim]
        
        # Apply layer normalization and residual connection
        fused_features = self.layer_norm3(fused_features)
        
        return fused_features


class TemporalInconsistencyAttentionModule(nn.Module):
    """
    Temporal Inconsistency Attention Module (TIAM) that explicitly targets 
    frame-to-frame discontinuities in deepfake videos.
    """
    
    def __init__(self, 
                 input_dim: int = 1024, 
                 hidden_dim: int = 256, 
                 num_layers: int = 2, 
                 dropout: float = 0.5):
        """
        Initialize the Temporal Inconsistency Attention Module.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(TemporalInconsistencyAttentionModule, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.bi_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 128),  # +1 for frame difference
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Process sequence of features with temporal inconsistency attention.
        
        Args:
            features: Sequence of features [batch_size, seq_len, input_dim]
            
        Returns:
            Weighted features with temporal attention [batch_size, seq_len, hidden_dim*2]
        """
        batch_size, seq_len, _ = features.shape
        
        # Compute frame-to-frame differences
        frame_diffs = torch.zeros(batch_size, seq_len, 1, device=features.device)
        for t in range(1, seq_len):
            frame_diffs[:, t, 0] = torch.norm(
                features[:, t] - features[:, t-1], dim=1
            )
        
        # Process with Bi-LSTM
        self.bi_lstm.flatten_parameters()  # For efficiency
        lstm_out, _ = self.bi_lstm(features)  # [batch_size, seq_len, hidden_dim*2]
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Compute attention weights based on hidden states and frame differences
        combined_features = torch.cat([lstm_out, frame_diffs], dim=2)
        attention_scores = self.attention_layer(combined_features)  # [batch_size, seq_len, 1]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights to LSTM outputs
        weighted_output = lstm_out * attention_weights
        
        # Apply dropout
        weighted_output = self.dropout(weighted_output)
        
        return weighted_output


class ArtifactDetector(nn.Module):
    """
    Artifact detector module that estimates the likelihood of manipulation
    based on the strength of detected artifacts in feature maps.
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256):
        """
        Initialize the Artifact Detector module.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
        """
        super(ArtifactDetector, self).__init__()
        
        self.detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate artifact likelihood from features.
        
        Args:
            features: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Artifact score [batch_size, 1]
        """
        # Global average pooling over sequence dimension
        pooled_features = torch.mean(features, dim=1)  # [batch_size, input_dim]
        
        # Compute artifact score
        artifact_score = self.detector(pooled_features)  # [batch_size, 1]
        
        return artifact_score


class ArtifactAwareLossFunction(nn.Module):
    """
    Artifact-Aware Loss Function (AALF) that combines weighted binary cross-entropy
    with a regularization term that penalizes predictions inconsistent with detected artifacts.
    """
    
    def __init__(self, w_real: float = 0.4, w_fake: float = 0.6, lambda_reg: float = 0.25):
        """
        Initialize the Artifact-Aware Loss Function.
        
        Args:
            w_real: Weight for real class
            w_fake: Weight for fake class
            lambda_reg: Regularization coefficient
        """
        super(ArtifactAwareLossFunction, self).__init__()
        
        self.w_real = w_real
        self.w_fake = w_fake
        self.lambda_reg = lambda_reg
        
        # Artifact detector
        self.artifact_detector = ArtifactDetector(input_dim=512)
    
    def forward(self, 
                y_pred: torch.Tensor, 
                y_true: torch.Tensor, 
                features: torch.Tensor) -> torch.Tensor:
        """
        Compute the artifact-aware loss.
        
        Args:
            y_pred: Predicted probabilities [batch_size, 1]
            y_true: Ground truth labels [batch_size, 1]
            features: Combined features [batch_size, seq_len, feature_dim]
            
        Returns:
            Loss value
        """
        # Compute weighted binary cross-entropy
        bce_loss = -self.w_real * y_true * torch.log(y_pred + 1e-7) - \
                   self.w_fake * (1 - y_true) * torch.log(1 - y_pred + 1e-7)
        bce_loss = bce_loss.mean()
        
        # Compute artifact score
        artifact_score = self.artifact_detector(features)  # [batch_size, 1]
        
        # Compute regularization term
        reg_term = torch.mean((y_pred - artifact_score) ** 2)
        
        # Combine losses
        total_loss = bce_loss + self.lambda_reg * reg_term
        
        return total_loss


class ComplexityEstimator(nn.Module):
    """
    Complexity Estimator for Adaptive Computational Scaling (ACS).
    Analyzes the first frame of a video to estimate detection difficulty.
    """
    
    def __init__(self, input_channels: int = 3):
        """
        Initialize the Complexity Estimator.
        
        Args:
            input_channels: Number of input channels
        """
        super(ComplexityEstimator, self).__init__()
        
        self.estimator = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate complexity score from first frame.
        
        Args:
            x: First frame tensor [batch_size, channels, height, width]
            
        Returns:
            Complexity score [batch_size, 1]
        """
        return self.estimator(x)


class TSFNet(nn.Module):
    """
    Tempo-Spatial-Fusion Network (TSF-Net) for deepfake detection.
    """
    
    def __init__(self, 
                 efficient_dim: int = 1280,
                 xception_dim: int = 2048,
                 fused_dim: int = 1024,
                 hidden_dim: int = 256,
                 num_classes: int = 1,
                 dropout: float = 0.5,
                 use_acs: bool = True):
        """
        Initialize the TSF-Net model.
        
        Args:
            efficient_dim: Dimension of EfficientNetV2L features
            xception_dim: Dimension of XceptionNet features
            fused_dim: Dimension of fused features
            hidden_dim: Dimension of hidden layers
            num_classes: Number of output classes (1 for binary classification)
            dropout: Dropout rate
            use_acs: Whether to use Adaptive Computational Scaling
        """
        super(TSFNet, self).__init__()
        
        self.use_acs = use_acs
        
        # Feature extractors
        self.efficient_extractor = EfficientNetV2LFeatureExtractor(pretrained=True)
        self.xception_extractor = ModifiedXceptionNet(pretrained=True)
        
        # Cross-Modal Attention Fusion
        self.cmaf = CrossModalAttentionFusion(
            efficient_dim=efficient_dim,
            xception_dim=xception_dim,
            output_dim=fused_dim
        )
        
        # Temporal Inconsistency Attention Module
        self.tiam = TemporalInconsistencyAttentionModule(
            input_dim=fused_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )
        
        # Adaptive Computational Scaling components
        if use_acs:
            self.complexity_estimator = ComplexityEstimator()
            
            # Lightweight path
            self.lightweight_model = nn.Sequential(
                models.mobilenet_v3_small(pretrained=True),
                nn.Linear(1000, 1),
                nn.Sigmoid()
            )
            
            # Standard path
            self.standard_model = nn.Sequential(
                models.efficientnet_v2_s(pretrained=True),
                nn.Linear(1000, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor, use_acs: bool = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the TSF-Net model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, channels, height, width]
            use_acs: Whether to use Adaptive Computational Scaling (overrides init setting)
            
        Returns:
            Output tensor of shape [batch_size, num_classes] or dict with outputs and attention weights
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Determine whether to use ACS
        use_acs_now = self.use_acs if use_acs is None else use_acs
        
        if use_acs_now:
            # Estimate complexity from first frame
            first_frame = x[:, 0]
            complexity_score = self.complexity_estimator(first_frame)
            
            # Select processing path based on complexity
            if complexity_score.mean() < 0.3:
                # Lightweight path for simple videos
                return self.lightweight_model(first_frame)
            elif complexity_score.mean() < 0.7:
                # Standard path for moderate videos
                return self.standard_model(first_frame)
            # Otherwise, continue with full model
        
        # Reshape for batch processing through CNNs
        x_reshaped = x.view(-1, channels, height, width)
        
        # Extract features from both networks
        efficient_features = self.efficient_extractor(x_reshaped)
        xception_features = self.xception_extractor(x_reshaped)
        
        # Reshape back to sequence format
        efficient_features = efficient_features.view(batch_size, seq_len, -1)
        xception_features = xception_features.view(batch_size, seq_len, -1)
        
        # Process each frame with CMAF
        fused_features = []
        for t in range(seq_len):
            fused_t = self.cmaf(efficient_features[:, t], xception_features[:, t])
            fused_features.append(fused_t)
        
        # Stack fused features
        fused_features = torch.stack(fused_features, dim=1)  # [batch_size, seq_len, fused_dim]
        
        # Process with TIAM
        temporal_features = self.tiam(fused_features)  # [batch_size, seq_len, hidden_dim*2]
        
        # Global average pooling over sequence dimension
        pooled_features = torch.mean(temporal_features, dim=1)  # [batch_size, hidden_dim*2]
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output
    
    def extract_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for visualization and analysis.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, channels, height, width]
            
        Returns:
            Dictionary containing attention weights and outputs
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for batch processing through CNNs
        x_reshaped = x.view(-1, channels, height, width)
        
        # Extract features from both networks
        efficient_features = self.efficient_extractor(x_reshaped)
        xception_features = self.xception_extractor(x_reshaped)
        
        # Reshape back to sequence format
        efficient_features = efficient_features.view(batch_size, seq_len, -1)
        xception_features = xception_features.view(batch_size, seq_len, -1)
        
        # Process each frame with CMAF and collect attention weights
        fused_features = []
        cmaf_weights = []
        
        for t in range(seq_len):
            # Store original forward method
            original_forward = self.cmaf.forward
            
            # Define a new forward method to capture attention weights
            def forward_with_attention(self, efficient_features, xception_features):
                batch_size = efficient_features.shape[0]
                
                # Apply layer normalization
                efficient_features = self.layer_norm1(efficient_features)
                xception_features = self.layer_norm2(xception_features)
                
                # Process each attention head
                head_outputs = []
                all_attention_weights = []
                
                for i in range(self.num_heads):
                    # Project queries from EfficientNetV2L features
                    queries = self.query_projections[i](efficient_features)
                    
                    # Project keys and values from XceptionNet features
                    keys = self.key_projections[i](xception_features)
                    values = self.value_projections[i](xception_features)
                    
                    # Reshape for attention computation
                    queries = queries.unsqueeze(1)
                    keys = keys.unsqueeze(1)
                    values = values.unsqueeze(1)
                    
                    # Compute attention scores
                    attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
                    attention_scores = attention_scores / math.sqrt(self.head_dim)
                    
                    # Apply softmax to get attention weights
                    attention_weights = F.softmax(attention_scores, dim=-1)
                    all_attention_weights.append(attention_weights)
                    
                    attention_weights = self.dropout(attention_weights)
                    
                    # Apply attention weights to values
                    head_output = torch.matmul(attention_weights, values)
                    head_output = head_output.squeeze(1)
                    
                    head_outputs.append(head_output)
                
                # Concatenate outputs from all heads
                multi_head_output = torch.cat(head_outputs, dim=1)
                
                # Apply output projection
                fused_features = self.output_projection(multi_head_output)
                
                # Apply layer normalization
                fused_features = self.layer_norm3(fused_features)
                
                return fused_features, torch.cat(all_attention_weights, dim=1)
            
            # Replace the forward method temporarily
            self.cmaf.forward = lambda ef, xf: forward_with_attention(self.cmaf, ef, xf)
            
            # Call the modified forward method
            fused_t, weights_t = self.cmaf(efficient_features[:, t], xception_features[:, t])
            
            # Restore the original forward method
            self.cmaf.forward = original_forward
            
            fused_features.append(fused_t)
            cmaf_weights.append(weights_t)
        
        # Stack fused features and attention weights
        fused_features = torch.stack(fused_features, dim=1)
        cmaf_weights = torch.stack(cmaf_weights, dim=1)
        
        # Process with TIAM and collect temporal attention weights
        batch_size, seq_len, _ = fused_features.shape
        
        # Compute frame-to-frame differences
        frame_diffs = torch.zeros(batch_size, seq_len, 1, device=fused_features.device)
        for t in range(1, seq_len):
            frame_diffs[:, t, 0] = torch.norm(
                fused_features[:, t] - fused_features[:, t-1], dim=1
            )
        
        # Process with Bi-LSTM
        self.tiam.bi_lstm.flatten_parameters()
        lstm_out, _ = self.tiam.bi_lstm(fused_features)
        
        # Apply layer normalization
        lstm_out = self.tiam.layer_norm(lstm_out)
        
        # Compute attention weights based on hidden states and frame differences
        combined_features = torch.cat([lstm_out, frame_diffs], dim=2)
        attention_scores = self.tiam.attention_layer(combined_features)
        
        # Apply softmax to get attention weights
        temporal_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights to LSTM outputs
        weighted_output = lstm_out * temporal_weights
        
        # Global average pooling over sequence dimension
        pooled_features = torch.mean(weighted_output, dim=1)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return {
            'output': output,
            'cmaf_weights': cmaf_weights,
            'temporal_weights': temporal_weights,
            'frame_diffs': frame_diffs
        }


def load_model(model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> TSFNet:
    """
    Load a pretrained TSF-Net model.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded TSF-Net model
    """
    model = TSFNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_video(model: TSFNet, video_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
    """
    Predict whether a video is a deepfake.
    
    Args:
        model: TSF-Net model
        video_path: Path to the video file
        device: Device to run inference on
        
    Returns:
        Dictionary with prediction results
    """
    # Initialize face preprocessor
    preprocessor = FacePreprocessor(target_size=(299, 299), device=device)
    
    # Process video
    frames = preprocessor.process_video(video_path, max_frames=32, stride=2)
    
    if len(frames) == 0:
        return {'error': 'No faces detected in video'}
    
    # Stack frames into a batch
    frames_tensor = torch.stack(frames).unsqueeze(0).to(device)  # [1, seq_len, channels, height, width]
    
    # Run inference
    with torch.no_grad():
        # Get prediction and attention weights
        results = model.extract_attention_weights(frames_tensor)
        
        output = results['output'].item()
        temporal_weights = results['temporal_weights'].squeeze().cpu().numpy()
        
        # Find frame with highest temporal attention
        max_attention_frame = int(np.argmax(temporal_weights))
        
        return {
            'probability': float(output),
            'prediction': 'FAKE' if output > 0.5 else 'REAL',
            'confidence': float(abs(output - 0.5) * 2),  # Scale to [0, 1]
            'max_attention_frame': max_attention_frame,
            'temporal_weights': temporal_weights.tolist()
        }
