import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from cirtorch.networks.imageretrievalnet import init_network

# 1. 基础配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "data/networks/gl18-tl-resnet101-gem-w-a4d43db.pth"
IMAGE_DIR = "data/test/paris6k/jpg"  # 你的底库图片文件夹
SAVE_DIR = "app/data/retrieval_db"   # 保存密钥和密文库的地方

os.makedirs(SAVE_DIR, exist_ok=True)

# 2. SkNN 密钥生成函数 (回归论文原版：双矩阵)
def generate_sknn_keys(d):
    # 生成两个 d x d 的可逆矩阵 (即 2048 x 2048)
    M1 = torch.randn(d, d)
    M2 = torch.randn(d, d)
    # 生成 d 维的二值向量 S
    S = torch.randint(0, 2, (d,)).float()
    return M1, M2, S

def encrypt_db_feature(p, M1, M2, S):
    d = len(p)
    p1 = torch.zeros(d)
    p2 = torch.zeros(d)
    
    for i in range(d):
        if S[i] == 0:
            # 论文规则(1): S=0 时，直接复制
            p1[i] = p[i]
            p2[i] = p[i]
        else:
            # 论文规则(2): S=1 时，随机拆分
            r = torch.randn(1).item()
            p1[i] = r
            p2[i] = p[i] - r
            
    # 加密变换: 分别乘以 M1.T 和 M2.T
    enc_part1 = torch.matmul(M1.T, p1)
    enc_part2 = torch.matmul(M2.T, p2)
    
    # 将两个 d 维向量拼接为 2d 维的密文向量
    p_enc = torch.cat((enc_part1, enc_part2))
    return p_enc
# 4. 图像预处理
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    print(f"正在加载 GL18 模型...")
    state = torch.load(MODEL_PATH)
    model = init_network({'architecture': state['meta']['architecture'], 'pooling': state['meta']['pooling'], 'whitening': state['meta']['whitening']})
    model.load_state_dict(state['state_dict'])
    model.to(DEVICE)
    model.eval()

    print("正在生成 SkNN 密钥...")
    M1, M2, S = generate_sknn_keys(2048)
    # 保存密钥供后端使用
    torch.save({'M1': M1, 'M2': M2, 'S': S}, f"{SAVE_DIR}/sknn_keys.pth")

    db_features = []
    db_images = []

    print(f"正在扫描图库: {IMAGE_DIR}")
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    

    with torch.no_grad():
        for idx, img_name in enumerate(image_files):
            img_path = os.path.join(IMAGE_DIR, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                # 提取 2048 维明文特征
                feature = model(tensor).squeeze().cpu()
                
                # 加密为 4096 维密文特征
                enc_feature = encrypt_db_feature(feature, M1, M2, S)
                
                db_features.append(enc_feature)
                db_images.append(img_name)
                
                if (idx + 1) % 50 == 0:
                    print(f"已处理 {idx + 1}/{len(image_files)} 张图片")
            except Exception as e:
                print(f"处理 {img_name} 失败: {e}")

    # 将列表转换为巨大的张量矩阵并保存
    db_features_tensor = torch.stack(db_features)
    torch.save(db_features_tensor, f"{SAVE_DIR}/encrypted_features.pth")
    torch.save(db_images, f"{SAVE_DIR}/image_names.pth")
    
    print(f"\n🎉 建库大功告成！")
    print(f"密文维度: {db_features_tensor.shape}")
    print(f"所有文件已保存至: {SAVE_DIR}")

if __name__ == "__main__":
    main()