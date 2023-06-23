import os
os.environ["PROJECT_DN"] = '/usr/src/api/common/libs/panic3d'


from common.libs.panic3d._util.util_v1 import * ; import common.libs.panic3d._util.util_v1 as uutil
from common.libs.panic3d._util.pytorch_v1 import * ; import common.libs.panic3d._util.pytorch_v1 as utorch
from common.libs.panic3d._util.twodee_v1 import * ; import common.libs.panic3d._util.twodee_v1 as u2d
from common.libs.panic3d._util.threedee_v0 import * ; import common.libs.panic3d._util.threedee_v0 as u3d
from common.libs.panic3d._util.video_v1 import * ; import common.libs.panic3d._util.video_v1 as uvid

import common.libs.panic3d._train.eg3dc.util.eg3dc_v0 as ueg3d
import common.libs.panic3d._util.serving_v1 as userving
from common.libs.panic3d._util import sketchers_v2 as usketch
from common.libs.panic3d._util import eg3d_metrics3d as egm
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
from common.libs.save_model import get_inference_images, save_avatar
from common.libs.alignment_image import create_keypoints_anime_face, face_alignment_transform
from io import BytesIO

# load reconstruction module (resnet extractor)
from common.libs.panic3d._train.danbooru_tagger.helpers.katepca import ResnetFeatureExtractorPCA
from common.libs.panic3d._util.twodee_v1 import * ; import common.libs.panic3d._util.twodee_v1 as u2d

# file path are changed from original one 
aligndata = pload('/usr/src/api/models/fandom_align_alignment.pkl')

PRE_DIFNED_ALIGN = 'daredemoE/fandom_align/genshin/aether/front'

# load illustration-to-render module
from common.libs.panic3d._train.img2img.util import rmline_wrapper
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


def rmline(img, aligndata, preds, M):
    rmline_model = rmline_wrapper.RMLineWrapper(('rmlineE_rmlineganA_n04', 199)).eval().to(device)
    # ipdb.set_trace()
    kpts = preds[0]['keypoints']
    # M = face_alignment_transform(kpts)
    print(M, 'MMMMMMMMMMMMMM')
    print(kpts, 'kptsssssssssssssssssssssssssssssss')
    print(kpts[None,], 'None kptsssssssssssssssssssssssssssssss')

#     print(                aligndata['transformation'],
#                 aligndata['_alignment']['source']['keypoints'][
#                     aligndata['_alignment']['source']['_detection_used']
#                 ][None,],
# 'testsetstststsststs')

    # ipdb.set_trace()

    with torch.no_grad():
        out = rmline_model(
            img,
            kpts[None,][0,:,:2],
            ),
    return out
    # with torch.no_grad():
    #     out = rmline_model(
    #         img,
    #         kpts,
    #     )[0,:,:2],
    
    # return out



    # with torch.no_grad():
    #     out = rmline_model(
    #         img,
    #         rmline_wrapper._apply_M_keypoints(
    #             aligndata['transformation'],
    #             aligndata['_alignment']['source']['keypoints'][
    #                 aligndata['_alignment']['source']['_detection_used']
    #             ][None,],
    #         )[0,:,:2],
    #     )
    # return out

## TODO align should be deleted
def generate_avatar(x, align, preds, M):
    resnet = ResnetFeatureExtractorPCA(
    #file path are changed from original one  
    '/usr/src/api/models/pca.pkl', 512,
).eval().to(device)


    with torch.no_grad():
        # attribute error bg　正常な動作をする方で、imageの中身を検証
        x['resnet_features'] = resnet(x['image'])
        # TODO: delete aligndata 
        x['image_rmline'] = rmline(x['image'], aligndata[align], preds, M)

    # get geometry (marching cubes)
    print(x['image_rmline'])
    ipdb.set_trace()
    with torch.no_grad():
        xin = {
            'cond': {
                'image_ortho_front': x['image_rmline'][0].bg('w').convert('RGB').t()[None].to(device),
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
    return glb_data

    # # GLBデータをファイルに書き込む
    # with open("output_done_by_flask.glb", "wb") as f:
    #     f.write(glb_data)

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
    return glb_data
    

# this code is copied from download_genshin_2d

def _apply_M(img, M, size=512):
    output = I(kornia.geometry.transform.warp_affine(
        img.convert('RGBA').bg('w').convert('RGB').t()[None],
        torch.tensor(M).float()[[1,0,2]].T[[1,0,2]].T[None,:2],
        (size,size),
        mode='bilinear',
        padding_mode='fill',
        align_corners=True,
        fill_value=torch.ones(3),
    )).alpha_set(I(kornia.geometry.transform.warp_affine(
        img['a'].t()[None],
        torch.tensor(M).float()[[1,0,2]].T[[1,0,2]].T[None,:2],
        (size,size),
        mode='bilinear',
        padding_mode='fill',
        align_corners=True,
        fill_value=torch.zeros(3),
    ))['r'])
     
    return output

def generate_image(image):
    # a = aligndata[bn]
    default_transformation = np.eye(3)
    a = {"transformation": default_transformation}
    fan,franch,idx,view = a['source'].split('/')
    img_src = I(f'{rdn}/{fan}/images/{franch}/{idx}/{view}.png')
    img_seg = I(f'./_data/lustrous/renders/{bn.replace("fandom_align","fandom_align_seg")}.png')
    out = _apply_M(img_src, a['transformation']).alpha_set(img_seg)
    if DEBUG:
        out.save(mkfile(f'/dev/shm/renders/{bn}.png'))
    else:
        out.save(mkfile(f'./_data/lustrous/renders/{bn}.png'))
   
def cv2pil(image_cv):
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cv)
    image_pil = image_pil.convert('RGB')

    return image_pil


# this function is run by flask API
# user_id, inference_resource_id are provided by API call
# as a test, fixed values are given at __init__.py
def ml_api_method(user_id, inference_resource_id):
    x = {}
    ## fetch image url from supabase
    image_url = get_inference_images(inference_resource_id)
    ## dowonload image data from url
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = image.convert('RGBA')
    ##
    
    ## make keypoints and preprocess image 
    results = create_keypoints_anime_face(image_url)

    ## TODO: delete unused images
    preds, transformed_image, image_cv, I_image, M = results
    
    # image_pil = cv2pil(image_cv)
    # image_pil = image_pil.convert('RGBA')

    x['image'] = u2d.I(image)


    # ipdb.set_trace()
    ## generate merching cube by edge3d
    merching_cube = generate_avatar(x, PRE_DIFNED_ALIGN, preds, M)
    glb_data = make_point_with_smooth(merching_cube)

    ## make array into glb file
    response = save_avatar(user_id, glb_data)

    return response


