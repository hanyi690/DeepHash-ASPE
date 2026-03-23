import argparse
import os
import time
import pickle
import pdb

import numpy as np

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

from ..networks.imageretrievalnet import init_network, extract_vectors
from ..datasets.datahelpers import cid2filename
from ..datasets.testdataset import configdataset
from ..utils.download import download_train, download_test
from ..utils.whiten import whitenlearn, whitenapply
from ..utils.evaluate import compute_map_and_print
from ..utils.general import get_data_root, htime

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet50-gem-w'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}

datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')

# network
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                    help="pretrained network or network path (destination where network is saved)")
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
                    help="off-the-shelf network, in the format 'ARCHITECTURE-POOLING' or 'ARCHITECTURE-POOLING-{reg-lwhiten-whiten}'," + 
                        " examples: 'resnet101-gem' | 'resnet101-gem-reg' | 'resnet101-gem-whiten' | 'resnet101-gem-lwhiten' | 'resnet101-gem-reg-whiten'")

# test options
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='oxford5k,paris6k',
                    help="comma separated list of test datasets: " + 
                        " | ".join(datasets_names) + 
                        " (default: 'oxford5k,paris6k')")
parser.add_argument('--image-size', '-imsize', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1]', 
                    help="use multiscale vectors for testing, " + 
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--whitening', '-w', metavar='WHITENING', default=None, choices=whitening_names,
                    help="dataset used to learn whitening for testing: " + 
                        " | ".join(whitening_names) + 
                        " (default: None)")

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

# ==========================================
# 加密模块开始
# ==========================================
def GenKey(dim):
    """
    Init: 生成密钥 (M1, M2, S) - 修复溢出版
    """
    print(f">> Encryption: Generating SkNN Keys (M1, M2, S)...")
    
    # 辅助函数：生成可逆矩阵
    def get_invertible_matrix(d):
        while True:
            M = np.random.randn(d, d)
            try:
                # 直接尝试求逆，如果成功说明矩阵可逆，不需要算行列式
                np.linalg.inv(M)
                return M
            except np.linalg.LinAlgError:
                continue # 极其罕见的情况，重试

    M1 = get_invertible_matrix(dim)
    M2 = get_invertible_matrix(dim)
        
    # 生成二进制向量 S
    S = np.random.randint(0, 2, size=(dim, 1))
    
    return M1, M2, S

def GenEnc(key, v):
    """
    GenEnc: 加密数据库向量 (Database Vector Encryption)
    v shape: (d, N)
    规则:
    (1) S[i]=0 -> v1[i] = v2[i] = v[i]
    (2) S[i]=1 -> v1[i] + v2[i] = v[i] (随机拆分)
    最后加密: v_hat = [M1.T * v1, M2.T * v2]
    """
    print(">> Encryption: Encrypting Database (GenEnc)...")
    M1, M2, S = key
    d, N = v.shape
    
    # 初始化 v1, v2
    v1 = np.zeros_like(v)
    v2 = np.zeros_like(v)
    
    # 为了加速，使用掩码 (Mask) 操作而不是 for 循环
    mask_0 = (S == 0).flatten() # S=0 的索引
    mask_1 = (S == 1).flatten() # S=1 的索引
    
    # 规则 (1): S=0 时，直接复制
    v1[mask_0, :] = v[mask_0, :]
    v2[mask_0, :] = v[mask_0, :]
    
    # 规则 (2): S=1 时，随机拆分 (v1=rand, v2=v-rand)
    # 生成与 v[mask_1] 形状相同的随机数
    rand_vals = np.random.randn(np.sum(mask_1), N) 
    v1[mask_1, :] = rand_vals
    v2[mask_1, :] = v[mask_1, :] - rand_vals
    
    # 加密变换: M1.T * v1 和 M2.T * v2
    # 注意：公式中是 M1^T * v1
    enc_part1 = np.dot(M1.T, v1)
    enc_part2 = np.dot(M2.T, v2)
    
    # 拼接成 2d 维度的长向量
    return np.vstack((enc_part1, enc_part2))

def GenTrap(key, w):
    """
    GenTrap: 加密查询向量 (Trapdoor Generation)
    w shape: (d, Nq)
    规则 (与 GenEnc 相反):
    (1) S[i]=0 -> w1[i] + w2[i] = w[i] (随机拆分)
    (2) S[i]=1 -> w1[i] = w2[i] = w[i]
    最后加密: w_hat = [M1^-1 * w1, M2^-1 * w2]
    """
    print(">> Encryption: Encrypting Query (GenTrap)...")
    M1, M2, S = key
    d, Nq = w.shape
    
    w1 = np.zeros_like(w)
    w2 = np.zeros_like(w)
    
    mask_0 = (S == 0).flatten()
    mask_1 = (S == 1).flatten()
    
    # 规则 (1): S=0 时，随机拆分 (注意这里和 GenEnc 是反的)
    rand_vals = np.random.randn(np.sum(mask_0), Nq)
    w1[mask_0, :] = rand_vals
    w2[mask_0, :] = w[mask_0, :] - rand_vals
    
    # 规则 (2): S=1 时，直接复制
    w1[mask_1, :] = w[mask_1, :]
    w2[mask_1, :] = w[mask_1, :]
    
    # 准备逆矩阵
    M1_inv = np.linalg.inv(M1)
    M2_inv = np.linalg.inv(M2)
    
    # 加密变换: M1^-1 * w1
    trap_part1 = np.dot(M1_inv, w1)
    trap_part2 = np.dot(M2_inv, w2)
    
    return np.vstack((trap_part1, trap_part2))
# ==========================================
# 加密模块结束
# ==========================================

def main():
    args = parser.parse_args()

    # check if there are unknown datasets
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    # download_train(get_data_root())
    download_test(get_data_root())

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path
    if args.network_path is not None:

        print(">> Loading network:\n>>>> '{}'".format(args.network_path))
        if args.network_path in PRETRAINED:
            # pretrained networks (downloaded automatically)
            state = load_url(PRETRAINED[args.network_path], model_dir=os.path.join(get_data_root(), 'networks'))
        else:
            # fine-tuned network from path
            state = torch.load(args.network_path)

        # parsing net params from meta
        # architecture, pooling, mean, std required
        # the rest has default values, in case that is doesnt exist
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get('local_whitening', False)
        net_params['regional'] = state['meta'].get('regional', False)
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False

        # load network
        net = init_network(net_params)
        net.load_state_dict(state['state_dict'])
        
        # if whitening is precomputed
        if 'Lw' in state['meta']:
            net.meta['Lw'] = state['meta']['Lw']
        
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # loading offtheshelf network
    elif args.network_offtheshelf is not None:
        
        # parse off-the-shelf parameters
        offtheshelf = args.network_offtheshelf.split('-')
        net_params = {}
        net_params['architecture'] = offtheshelf[0]
        net_params['pooling'] = offtheshelf[1]
        net_params['local_whitening'] = 'lwhiten' in offtheshelf[2:]
        net_params['regional'] = 'reg' in offtheshelf[2:]
        net_params['whitening'] = 'whiten' in offtheshelf[2:]
        net_params['pretrained'] = True

        # load off-the-shelf network
        print(">> Loading off-the-shelf network:\n>>>> '{}'".format(args.network_offtheshelf))
        net = init_network(net_params)
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))
    if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
        msp = net.pool.p.item()
        print(">> Set-up multiscale:")
        print(">>>> ms: {}".format(ms))            
        print(">>>> msp: {}".format(msp))
    else:
        msp = 1

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    if args.whitening is not None:
        start = time.time()

        if 'Lw' in net.meta and args.whitening in net.meta['Lw']:
            
            print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
            
            if len(ms)>1:
                Lw = net.meta['Lw'][args.whitening]['ms']
            else:
                Lw = net.meta['Lw'][args.whitening]['ss']

        else:

            # if we evaluate networks from path we should save/load whitening
            # not to compute it every time
            if args.network_path is not None:
                whiten_fn = args.network_path + '_{}_whiten'.format(args.whitening)
                if len(ms) > 1:
                    whiten_fn += '_ms'
                whiten_fn += '.pth'
            else:
                whiten_fn = None

            if whiten_fn is not None and os.path.isfile(whiten_fn):
                print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
                Lw = torch.load(whiten_fn)

            else:
                print('>> {}: Learning whitening...'.format(args.whitening))
                
                # loading db
                db_root = os.path.join(get_data_root(), 'train', args.whitening)
                ims_root = os.path.join(db_root, 'ims')
                db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.whitening))
                with open(db_fn, 'rb') as f:
                    db = pickle.load(f)
                images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

                # extract whitening vectors
                print('>> {}: Extracting...'.format(args.whitening))
                wvecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
                
                # learning whitening 
                print('>> {}: Learning...'.format(args.whitening))
                wvecs = wvecs.numpy()
                m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
                Lw = {'m': m, 'P': P}

                # saving whitening if whiten_fn exists
                if whiten_fn is not None:
                    print('>> {}: Saving to {}...'.format(args.whitening, whiten_fn))
                    torch.save(Lw, whiten_fn)

        print('>> {}: elapsed time: {}'.format(args.whitening, htime(time.time()-start)))

    else:
        Lw = None

    # evaluate on test datasets
    datasets = args.datasets.split(',')
    for dataset in datasets: 
        start = time.time()

        print('>> {}: Extracting...'.format(dataset))

        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
        images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
        qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
        try:
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        except:
            bbxs = None  # for holidaysmanrot and copydays
        
        # extract database and query vectors
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
        print('>> {}: query images...'.format(dataset))
        qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, msp=msp)
        
        print('>> {}: Evaluating...'.format(dataset))
        # ==========================================
        # 插入加密操作 (START)
        # ==========================================
        # [关键修复 1] 将 PyTorch Tensor 转换为 Numpy Array
        # 如果 vecs 在 GPU 上，需要先 .cpu()
        if hasattr(vecs, 'cpu'):
            vecs_np = vecs.cpu().numpy()
            qvecs_np = qvecs.cpu().numpy()
        else:
        # 如果已经是 numpy 或者 tensor 在 cpu
            vecs_np = vecs.numpy() if hasattr(vecs, 'numpy') else vecs
            qvecs_np = qvecs.numpy() if hasattr(qvecs, 'numpy') else qvecs

        # 1. 生成密钥 (包含 M1, M2, S)
        sknn_key = GenKey(vecs_np.shape[0])

        # 2. 备份明文以便验证 (使用 .copy() 是安全的，因为现在是 numpy 了)
        vecs_plain = vecs_np.copy()
    
        # 3. 执行加密
        # 注意：加密后 vecs_enc 维度会从 2048 变成 4096
        vecs_enc = GenEnc(sknn_key, vecs_np)     
        qvecs_trap = GenTrap(sknn_key, qvecs_np) 

        # 4. 验证加密效果
        print(f"   [Verify] Original Dim: {vecs_plain.shape[0]}")
        print(f"   [Verify] Encrypted Dim: {vecs_enc.shape[0]} (Dimension Expanded)")
        print(f"   [Verify] Plain value[0,0]: {vecs_plain[0,0]:.4f}")
        print(f"   [Verify] Encrypted value[0,0]: {vecs_enc[0,0]:.4f}")

        # 5. 计算内积 (使用加密后的 numpy 数组)
        # [关键修复 2] 这里直接使用加密后的 numpy 数组计算，不需要再转回 Tensor
        scores = np.dot(vecs_enc.T, qvecs_trap)
        #在之前的代码逻辑中，所有的特征向量都经过了 L2 归一化（长度变为了 1）。在数学上，当两个向量的长度均为 1 时，它们的点积（Dot Product）就等于它们之间夹角的余弦相似度（Cosine Similarity）。分数越接近1越相似
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset, ranks, cfg['gnd'])
    
        if Lw is not None:
            # whiten the vectors
            vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
            qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])

            # search, rank, and print
            scores = np.dot(vecs_lw.T, qvecs_lw)
            ranks = np.argsort(-scores, axis=0)
            compute_map_and_print(dataset + ' + whiten', ranks, cfg['gnd'])
        
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


if __name__ == '__main__':
    main()