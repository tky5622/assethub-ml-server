



from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
from _util.threedee_v0 import * ; import _util.threedee_v0 as u3d
from _util.video_v1 import * ; import _util.video_v1 as uvid

import _train.eg3dc.util.eg3dc_v0 as ueg3d
import _util.serving_v1 as userving
from _util import sketchers_v2 as usketch
from _util import eg3d_metrics3d as egm
import trimesh
from PIL import Image

device = torch.device('cuda')
import secrets
import string

import os
import random
import string
import trimesh
import pdb
import ipdb


def random_string(n):
    """ランダムな文字列を生成する関数"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def create_material(mc_colors):
    material = trimesh.visual.material.PBRMaterial()

    mc_colors = (mc_colors * 255).astype(np.uint8)
    n_pixels = mc_colors.shape[0]
    texture_size = int(np.ceil(np.sqrt(n_pixels)))
    padded_colors = np.zeros((texture_size * texture_size, 3), dtype=np.uint8)
    padded_colors[:n_pixels] = mc_colors
    texture_image = Image.fromarray(padded_colors.reshape(texture_size, texture_size, 3), mode='RGB')
    texture_image.save('temp_texture_image.png')  # 確認用

    # 以下の行を変更
    material.baseColorTexture = TextureVisuals(image=texture_image)
    return material



def convert_to_glb(mesh, output_dir):
    filename = "{}.glb".format(mesh['name'])
    filepath = os.path.join(output_dir, filename)

    # set vertex colors to white if not specified
    if 'vertex_colors' not in mesh:
        mesh['vertex_colors'] = trimesh.visual.ColorVisuals([[1.0, 1.0, 1.0, 1.0]]*len(mesh['vertices']))
    
    # create trimesh object and export as GLB
    trimesh_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'], vertex_colors=mesh['vertex_colors'])
    trimesh_mesh.export(filepath)


inferquery = 'ecrutileE_eclustrousC_n120-00000-000200'
edn = f'./temp/eval/{inferquery}'


# load dataset
from _databacks import lustrous_renders_v1 as dklustr
dk = dklustr.DatabackendMinna()
bns = [
    f'daredemoE/fandom_align/{bn}/front'
    for bn in uutil.read_bns('./_data/lustrous/subsets/daredemoE_test.csv')
]
aligndata = pload('./_data/lustrous/renders/daredemoE/fandom_align_alignment.pkl')

# load illustration-to-render module
from _train.img2img.util import rmline_wrapper
rmline_model = rmline_wrapper.RMLineWrapper(('rmlineE_rmlineganA_n04', 199)).eval().to(device)
def rmline(img, aligndata):
    with torch.no_grad():
        out = rmline_model(
            img,
            rmline_wrapper._apply_M_keypoints(
                aligndata['transformation'],
                aligndata['_alignment']['source']['keypoints'][
                    aligndata['_alignment']['source']['_detection_used']
                ][None,],
            )[0,:,:2],
        )
    return out

# load reconstruction module
ckpt = ueg3d.load_eg3dc_model(inferquery, force_sigmoid=True)
G = ckpt.G.eval().to(device)
inference_opts = {
    'triplane_crop': 0.1,
    'cull_clouds': 0.5,
    # 'binarize_clouds': 0.4,
    'paste_params': {
        'mode': 'default',
        'thresh_weight': 0.95,
        'thresh_edges': 0.02,
        'thresh_occ': 0.05, 'offset_occ': 0.01,
        'thresh_dxyz': 0.000005,
    },
}

# load reconstruction module (resnet extractor)
from _train.danbooru_tagger.helpers.katepca import ResnetFeatureExtractorPCA
resnet = ResnetFeatureExtractorPCA(
    './_data/lustrous/preprocessed/minna_resnet_feats_ortho/pca.pkl', 512,
).eval().to(device)


# eval over samples
bw = G.rendering_kwargs['box_warp']
rk = G.rendering_kwargs
r0,r1 = rk['ray_start'], rk['ray_end']
seed = 0
count=0
# ipdb.set_trace()

for bn in tqdm(bns):
    # preprocess
    x = dk[bn]
    print(x, 'this is X##############################################')
    print(x.image, "image of XXXXxxxxxxxxxxxxx")
    with torch.no_grad():
        x['resnet_features'] = resnet(x.image)
        x['image_rmline'] = rmline(x.image, aligndata[bn])

    # get geometry (marching cubes)
    with torch.no_grad():
        xin = {
            'cond': {
                'image_ortho_front': x['image_rmline'].bg('w').convert('RGB').t()[None].to(device),
                'resnet_chonk': x['resnet_features'][None,0],
            },
            'seeds': [seed,],
            **inference_opts,
        }
        vol = egm.get_eg3d_volume(G, xin)
        print(vol, 'vol')
        mc = egm.marching_cubes(
            vol['densities'].cpu().numpy()[0,0],
            vol['rgbs'].cpu().numpy()[0,:3],
            G.rendering_kwargs['box_warp'],
            level=0.5,
        )
        print(mc, 'mc')

    ipdb.set_trace()
    fn_march = f'{edn}/{bn.replace("fandom_align","marching_cubes")}.pkl'
    uutil.pdump(mc, mkfile(fn_march))
    import trimesh
    import pygltflib
    from trimesh.exchange.gltf import export_glb

        # Trimesh形式に変換
    mesh = trimesh.Trimesh(vertices=mc['verts'], faces=mc['faces'], vertex_colors=mc['colors'])

    # メッシュをシーンに追加
    # scene = trimesh.Scene(mesh)

    # # GLB形式のバイナリデータに変換
    # glb_data = export_glb(scene)
    import trimesh.smoothing as smoothing

# 既存のコードでTrimesh形式に変換されたmeshオブジェクトを使用
    smoothed_mesh = mesh.copy()
    vertex_neighbors = smoothed_mesh.vertex_neighbors
    iterations = 20

    for _ in range(iterations):
        new_vertices = np.zeros_like(smoothed_mesh.vertices)
        for vertex_index, neighbors in enumerate(vertex_neighbors):
            new_vertices[vertex_index] = smoothed_mesh.vertices[vertex_index] + np.mean(smoothed_mesh.vertices[neighbors] - smoothed_mesh.vertices[vertex_index], axis=0)
        smoothed_mesh.vertices = new_vertices
    
    smoothed_scene = trimesh.Scene(smoothed_mesh)
    glb_data = export_glb(smoothed_scene)
    with open("output3333.glb", "wb") as f:
        f.write(glb_data)
    count +=1



        # get images (various views)
    for cm,cam_view,elev,azim,fov in [
        ('camO', 'front', 0, 0, -1),
        ('camO', 'left', 0, 90, -1),
        ('camO', 'right', 0, -90, -1),
        ('camO', 'back', 0, 180, -1),
        *[
            ('camP', f'{v:04d}', *dklustr.cam60[v].to(device), 30)
            for v in dklustr.camsubs['spin12']
        ],
    ]:
        with torch.no_grad():
            xin = {
                'elevations': elev *torch.ones(1).to(device),
                'azimuths': azim *torch.ones(1).to(device),
                'fovs': fov *torch.ones(1).to(device),
                'cond': {
                    'image_ortho_front': x['image_rmline'].bg('w').convert('RGB').t()[None].to(device),
                    'resnet_chonk': x['resnet_features'][None,0],
                },
                'seeds': [seed,],
                **inference_opts,
            }
            out = G.f(xin, return_more=True)

        if cm=='camO':
            fn_pred_rgb = f'{edn}/{bn.replace("fandom_align","ortho")}.png'
            fn_pred_xyza = f'{edn}/{bn.replace("fandom_align","ortho_xyza")}.png'
        elif cm=='camP':
            fn_pred_rgb = f'{edn}/{bn.replace("fandom_align","rgb60")}.png'
            fn_pred_xyza = f'{edn}/{bn.replace("fandom_align","xyza60")}.png'
        else:
            assert 0
        fn_pred_rgb = fn_pred_rgb.replace('/front', f'/{cam_view}')
        fn_pred_xyza = fn_pred_xyza.replace('/front', f'/{cam_view}')

        xyza = torch.cat([
            (out['image_xyz']+bw/2)/bw,
            out['image_weights'],
        ], dim=1)
        I(out['image']).save(mkfile(fn_pred_rgb))
        I(xyza).save(mkfile(fn_pred_xyza))
        
        del xin, out, xyza
        # break



