from torchvision.datasets import ImageFolder
from torchvision import transforms
from train import train_linear_layer
from test_clip import test_with_clip
from model import MiniGPT4Wrapper
from config import CONFIG

transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
])

def main():
    # Charger les ensembles pour une seule classe (classe 0)
    class_idx = 0
    dataset = ImageFolder(root=f"{CONFIG['dataset_path']}/class_{class_idx}/dummy_folder", transform=transform)

    # Entraînement
    print(f"Training Task {class_idx}")
    train_linear_layer(dataset)

    # Génération des descriptions textuelles
    minigpt4 = MiniGPT4Wrapper(CONFIG["minigpt4_path"], CONFIG["vicuna7b_path"], CONFIG["device"])
    test_images = [f"{CONFIG['dataset_path']}/class_{class_idx}/dummy_folder/image_{i}.png" for i in range(len(dataset))]
    generated_texts = [minigpt4.generate_text(None) for _ in test_images]  # Placeholder

    # Testing avec CLIP
    print("Testing with CLIP...")
    test_with_clip(test_images, generated_texts)

if __name__ == "__main__":
    main()