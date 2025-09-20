# fix.py
import torch

def fix_device_mismatch(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint type: {type(checkpoint)}")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, tuple):
        print("Checkpoint is a tuple - processing each element")
        fixed_checkpoint = []
        for i, item in enumerate(checkpoint):
            if isinstance(item, torch.Tensor):
                print(f"  Item {i}: Tensor on device {item.device}")
                fixed_checkpoint.append(item.cpu())
            else:
                print(f"  Item {i}: {type(item)}")
                fixed_checkpoint.append(item)
        fixed_checkpoint = tuple(fixed_checkpoint)
        
    elif isinstance(checkpoint, dict):
        print("Checkpoint is a dictionary")
        fixed_checkpoint = {}
        for key in checkpoint:
            if isinstance(checkpoint[key], torch.Tensor):
                print(f"  {key}: Tensor on device {checkpoint[key].device}")
                fixed_checkpoint[key] = checkpoint[key].cpu()
            else:
                print(f"  {key}: {type(checkpoint[key])}")
                fixed_checkpoint[key] = checkpoint[key]
                
    else:
        print(f"Unknown checkpoint type: {type(checkpoint)}")
        fixed_checkpoint = checkpoint
    
    # Save fixed checkpoint
    fixed_path = checkpoint_path.replace('.pth', '_fixed.pth')
    torch.save(fixed_checkpoint, fixed_path)
    print(f"Fixed checkpoint saved to: {fixed_path}")
    return fixed_path

# Usage
if __name__ == "__main__":
    fixed_checkpoint = fix_device_mismatch("output/people_snapshot/chkpnt_best.pth")