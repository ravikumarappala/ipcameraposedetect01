"""
HMR2 Direct SMPL Regressor - Single Image → SMPL Mesh
Replaces stereo triangulation + optimization
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path

# HMR2 model (download pretrained weights)
HMR2_MODEL_PATH = "hmr2_wild.pth"  # Download from: https://github.com/HongsukNoh/HMR2

class HMR2Regressor:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Simplified HMR2 (backbone + regressor head)
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.backbone.fc = torch.nn.Identity()  # Remove classifier
        
        # SMPL regressor head (72 pose + 10 shape + 3 cam)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 85)  # 72 pose + 10 shape + 3 cam
        )
        
        # Load pretrained (you'll need to download)
        checkpoint = torch.load(model_path, map_location=device)
        self.regressor.load_state_dict(checkpoint['regressor'])
        self.regressor.to(device)
        self.regressor.eval()
        
        # SMPL model (reuse your existing one)
        self.smpl = SMPLModel(SMPL_MODEL_PATH, device)  # Your existing class
        
    def predict(self, image):
        """
        image: (H,W,3) BGR uint8
        Returns: betas(10), pose(72), trans(3), verts, joints
        """
        # Preprocess
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.0
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0), (224, 224)
        ).to(self.device)
        
        # Forward pass
        features = self.backbone(img_tensor)
        params = self.regressor(features)  # [72+10+3=85]
        
        pose = params[:, :72]
        betas = params[:, 72:82] * 2.0  # Scale to SMPL range
        trans = params[:, 82:] * 1000   # Scale to mm
        
        # SMPL forward
        verts, joints = self.smpl.forward(betas[0], pose[0], trans[0])
        verts_mm = verts.cpu().numpy() * 1000
        joints_mm = joints.cpu().numpy() * 1000
        
        return betas[0].cpu().numpy(), pose[0].cpu().numpy(), trans[0].cpu().numpy(), verts_mm, joints_mm

def process_single_image(image_path, model_path=HMR2_MODEL_PATH, output_dir="hmr2_measurements"):
    """Direct HMR2 pipeline - single image only!"""
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"[1/4] Processing {image_path}")
    print(f"  Image size: {image.shape}")
    
    # HMR2 inference
    print(f"\n[2/4] HMR2 inference...")
    hmr2 = HMR2Regressor(model_path, device)
    betas, pose, trans, fitted_verts, fitted_joints = hmr2.predict(image)
    
    print(f"  Betas: {betas[:5].round(3)}...")
    print(f"  Fitted {len(fitted_verts)} verts, {len(fitted_joints)} joints")
    
    # Measurements (reuse your existing class)
    print(f"\n[3/4] Computing measurements...")
    measurer = BodyMeasurements(fitted_joints, fitted_verts)
    measurements = measurer.compute_all()
    
    # Render mesh (reuse your existing renderers)
    print(f"\n[4/4] Rendering...")
    verts_ready = rotate_for_upright_view(fitted_verts)
    try:
        mesh_img = render_smpl_mesh(verts_ready, hmr2.smpl.faces)
        mesh_img_bgr = cv2.cvtColor(mesh_img, cv2.COLOR_RGBA2BGR) if mesh_img.shape[2]==4 else cv2.cvtColor(mesh_img, cv2.COLOR_RGB2BGR)
    except:
        mesh_img = render_smpl_mesh_matplotlib(verts_ready, hmr2.smpl.faces)
        mesh_img_bgr = cv2.cvtColor(mesh_img, cv2.COLOR_RGB2BGR)
    
    # Save everything
    csv_path = f"{output_dir}/{base_name}_hmr2_measurements_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Measurement', 'cm'])
        for name, vals in measurements.items():
            writer.writerow([name, f"{vals['cm']:.1f}"])
    
    mesh_path = f"{output_dir}/{base_name}_hmr2_mesh_{timestamp}.png"
    cv2.imwrite(mesh_path, mesh_img_bgr)
    
    print(f"\n✅ Saved:")
    print(f"  {csv_path}")
    print(f"  {mesh_path}")
    print(f"\nKey measurements:")
    print(f"  Height: {measurements.get('Total Height (chain)', {}).get('cm', '?'):.1f} cm")
    print(f"  Shoulder width: {measurements.get('Shoulder Width', {}).get('cm', '?'):.1f} cm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Single input image")
    parser.add_argument("--model", default=HMR2_MODEL_PATH, help="HMR2 model path")
    args = parser.parse_args()
    
    process_single_image(args.image, args.model)
