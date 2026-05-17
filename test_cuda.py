import torch

cuda_dispo = torch.cuda.is_available()
print(f"CUDA actif : {cuda_dispo}")

if cuda_dispo:
    print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
    t = torch.tensor([1.0]).to("cuda")
    print(f"Localisation d'un tenseur de test : {t.device}")
else:
    print("Échec critique : PyTorch est cantonné au CPU.")

    