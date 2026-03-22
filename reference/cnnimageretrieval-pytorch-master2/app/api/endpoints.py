import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from torchvision import transforms
from cirtorch.networks.imageretrievalnet import init_network
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RETRIEVAL_DB_DIR = "app/data/retrieval_db"
MODEL_PATH = "data/networks/gl18-tl-resnet101-gem-w-a4d43db.pth"

retrieval_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

retrieval_model = None
sknn_keys = None
db_features = None
db_image_names = None

def load_retrieval_system():
    """懒加载引擎"""
    global retrieval_model, sknn_keys, db_features, db_image_names
    if retrieval_model is not None:
        return
    
    logger.info("正在将 GL18 模型和密文库装载进显存...")
    try:
        # 加载 GL18
        state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        retrieval_model = init_network({'architecture': state['meta']['architecture'], 'pooling': state['meta']['pooling'], 'whitening': state['meta']['whitening']})
        retrieval_model.load_state_dict(state['state_dict'])
        retrieval_model.to(device)
        retrieval_model.eval()

        # 加载 SkNN 密文库和密钥
        sknn_keys = torch.load(f"{RETRIEVAL_DB_DIR}/sknn_keys.pth", map_location=device, weights_only=False)
        db_features = torch.load(f"{RETRIEVAL_DB_DIR}/encrypted_features.pth", map_location=device, weights_only=False)
        db_image_names = torch.load(f"{RETRIEVAL_DB_DIR}/image_names.pth", weights_only=False)
        logger.info("🎉 隐私检索引擎加载完毕！")
    except Exception as e:
        logger.error(f"加载失败: {e}")
        raise e

def encrypt_query_feature(p, M1, M2, S):
    d = len(p)
    q1 = torch.zeros(d, device=device)
    q2 = torch.zeros(d, device=device)
    
    for i in range(d):
        if S[i] == 0:
            # 论文规则(1): 查询端 S=0 时，与建库端相反，进行拆分
            # 为了消除搜索时的浮点数随机跳动，将此处的随机噪音 r 设为 0
            r = 0.0
            q1[i] = r
            q2[i] = p[i] - r
        else:
            # 论文规则(2): 查询端 S=1 时，直接复制
            q1[i] = p[i]
            q2[i] = p[i]
            
    # 【高精度优化】将矩阵提升至 Float64 计算逆矩阵，消除浮点误差
    M1_double = M1.to(torch.float64).to(device)
    M2_double = M2.to(torch.float64).to(device)
    q1_double = q1.to(torch.float64)
    q2_double = q2.to(torch.float64)
    
    # 加密变换: 分别乘以 M1 和 M2 的逆矩阵
    trap_part1 = torch.matmul(torch.inverse(M1_double), q1_double)
    trap_part2 = torch.matmul(torch.inverse(M2_double), q2_double)
    
    # 拼接并转回 Float32 供后续点乘
    q_enc = torch.cat((trap_part1, trap_part2)).to(torch.float32)

    # === 密文验证输出 ===
    print("\n" + "="*40)
    print("隐私加密验证进行中...")
    print(f"[明文特征 p] 维度: {p.shape}")
    print(f"   前 4 个数值: {p[:4].cpu().numpy()}")
    print("-" * 40)
    print(f"[密文特征 q_enc] 维度: {q_enc.shape}")
    print(f"   前 4 个数值: {q_enc[:4].cpu().numpy()}")
    print("="*40 + "\n")
    
    return q_enc

@router.post("/privacy_search")
async def privacy_search(image: UploadFile = File(...)):
    """加密图像检索核心接口"""
    load_retrieval_system()
    
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "无效的图像格式")
        
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = retrieval_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_feature = retrieval_model(tensor).squeeze()
            M1 = sknn_keys['M1'].to(device)
            M2 = sknn_keys['M2'].to(device)
            S = sknn_keys['S'].to(device)
            enc_query = encrypt_query_feature(query_feature, M1, M2, S)
            
        # 密文内积计算
        db_feats_device = db_features.to(device)
        scores = torch.matmul(db_feats_device, enc_query)
        
        # 取 Top-10
        topk_scores, topk_indices = torch.topk(scores, k=10)
        
        results = []
        for score, idx in zip(topk_scores, topk_indices):
            img_name = db_image_names[idx.item()]
            results.append({
                "name": img_name,
                "url": f"/data/test/paris6k/jpg/{img_name}",  # 拼接静态访问路径
                "score": round(score.item(), 4)
            })
            
        return {"status": "success", "results": results}
        
    except Exception as e:
        logger.error(f"检索失败: {str(e)}")
        raise HTTPException(500, f"检索过程中出错: {str(e)}")