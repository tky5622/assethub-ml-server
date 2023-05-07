
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
import trimesh.smoothing as smoothing

device = torch.device('cuda')
import secrets
import string

import os
import random
import string
import trimesh
import pdb
import ipdb
import trimesh
import pygltflib
from trimesh.exchange.gltf import export_glb


# load reconstruction module (resnet extractor)
from _train.danbooru_tagger.helpers.katepca import ResnetFeatureExtractorPCA

aligndata = pload('./_data/lustrous/renders/daredemoE/fandom_align_alignment.pkl')

PRE_DIFNED_ALIGN = 'daredemoE/fandom_align/genshin/aether/front'

# load illustration-to-render module
from _train.img2img.util import rmline_wrapper
# load reconstruction module

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

INFER_QUERY = 'ecrutileE_eclustrousC_n120-00000-000200'
ckpt = ueg3d.load_eg3dc_model(INFER_QUERY, force_sigmoid=True)
G = ckpt.G.eval().to(device)



# eval over samples
bw = G.rendering_kwargs['box_warp']
rk = G.rendering_kwargs
r0,r1 = rk['ray_start'], rk['ray_end']
seed = 0


def rmline(img, aligndata):
    rmline_model = rmline_wrapper.RMLineWrapper(('rmlineE_rmlineganA_n04', 199)).eval().to(device)
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


def generate_avatar(x, align):
    resnet = ResnetFeatureExtractorPCA(
    './_data/lustrous/preprocessed/minna_resnet_feats_ortho/pca.pkl', 512,
).eval().to(device)


    with torch.no_grad():
        # attribute error bg　正常な動作をする方で、imageの中身を検証
        x['resnet_features'] = resnet(x['image'])
        x['image_rmline'] = rmline(x['image'], aligndata[align])

    # get geometry (marching cubes)
    print(x['image_rmline'])
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
    return mc



def make_point_into_glb(mc):
    mesh = trimesh.Trimesh(vertices=mc['verts'], faces=mc['faces'], vertex_colors=mc['colors'])

    # メッシュをシーンに追加
    scene = trimesh.Scene(mesh)

    # GLB形式のバイナリデータに変換
    glb_data = export_glb(scene)

    # GLBデータをファイルに書き込む
    with open("output_done_by_flask.glb", "wb") as f:
        f.write(glb_data)

    # save 3d models to database
    # make it as async function

def make_point_with_smooth(mc):
    # Trimesh形式に変換
    mesh = trimesh.Trimesh(vertices=mc['verts'], faces=mc['faces'], vertex_colors=mc['colors'])

    # 既存のコードでTrimesh形式に変換されたmeshオブジェクトを使用
    mesh = trimesh.Trimesh(vertices=mc['verts'], faces=mc['faces'], vertex_colors=mc['colors'])
    smoothed_mesh = mesh.copy()
    vertex_neighbors = smoothed_mesh.vertex_neighbors
    iterations = 10
    print(mesh, 'mesh')

    # apply smooth
    for _ in range(iterations):
        new_vertices = np.zeros_like(smoothed_mesh.vertices)
        for vertex_index, neighbors in enumerate(vertex_neighbors):
            new_vertices[vertex_index] = smoothed_mesh.vertices[vertex_index] + np.mean(smoothed_mesh.vertices[neighbors] - smoothed_mesh.vertices[vertex_index], axis=0)
        smoothed_mesh.vertices = new_vertices


    print('loop finished')
    
    smoothed_scene = trimesh.Scene(smoothed_mesh)
    glb_data = export_glb(smoothed_scene)
    print(glb_data)
    
    with open("output3333erere.glb", "wb") as f:
        f.write(glb_data)
    print('saved')
    
def ml_api_method():
    x = {}
    image = Image.open('./front.png')
    x['image'] = u2d.I(image)
    merching_cube = generate_avatar(x, PRE_DIFNED_ALIGN)
    make_point_with_smooth(merching_cube)
    return 'done'

