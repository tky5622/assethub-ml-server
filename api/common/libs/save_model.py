import os
from supabase import create_client, Client
import uuid
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# file structure and code should be fixed as it is dirty

def get_inference_images(inference_resource_id):
    response = supabase.table('inference_resource_id').select('*').eq('inference_resource_id', inference_resource_id).execute()
    print(response)
    # TODO: add return multiple images for reference (as next feature)
    return response['resource_url']



def save_avatar(user_id):
    avatar_id = str(uuid.uuid4())
    source = f'{avatar_id}.glb'
    with open(source, 'rb+') as f:
        res = supabase.storage().from_('avatars').upload(user_id, os.path.abspath(source))

    return res