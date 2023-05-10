import os
from supabase import create_client, Client
import uuid
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# file structure and code should be fixed as it is dirty

def get_inference_images(inference_resource_id):
    # print(inference_resource_id)
    response = supabase.table('inference_resources').select('*').eq('inference_resource_id', inference_resource_id).execute()
    data= response.data

    if data:
        first_row = data[0]  # リストの最初の要素（データの最初の行）を取得
        print(first_row)
        resource_url = first_row['resource_url']  # 'resource_url'列の値を取得
        # TODO: add return multiple images for reference (as next feature)
        return resource_url
    else:
        print("No data returned from database")
        return None

    return resource_url



def save_avatar(user_id, glb_data):
    avatar_id = str(uuid.uuid4())
    source = f'{user_id}/{avatar_id}.glb'
    from io import BytesIO
    # glb_dataがGLB形式のデータをバイナリ形式で保持しているとする
    import tempfile

    print(glb_data)

    # glb_dataがGLB形式のデータをバイナリ形式で保持しているとする
    # 一時ファイルを作成し、そのパスを取得
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(glb_data)
        print(temp_file)
        temp_path = temp_file.name

    # 一時ファイルをSupabaseにアップロード
    destination = source
    res = supabase.storage.from_('avatars').upload(destination, temp_path)

    # 一時ファイルを削除
    os.unlink(temp_path)


# from io import BytesIO
# import os
# import tempfile

# # glb_dataがGLB形式のデータをバイナリ形式で保持しているとする
# # 一時ファイルを作成し、そのパスを取得
# with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#     temp_file.write(glb_data)
#     temp_path = temp_file.name

# # 一時ファイルをSupabaseにアップロード
# destination = "folder/subfolder/6af772bf-7f04-44d6-9237-79bbc9550891.glb"
# res = supabase.storage().from_('avatars').upload(destination, temp_path)

# # 一時ファイルを削除
# os.unlink(temp_path)
