from huggingface_hub import HfApi, HfFolder, CommitOperationAdd
from huggingface_hub import create_repo
import yaml
from pathlib import Path

batch_config =  yaml.safe_load(open('batch_config.yaml','r')) 

with open(HfFolder.path_token, 'r') as fin: hf_token = fin.read();


for batch in batch_config['batches']:
    base_outpath = Path(batch_config['base_outpath']) / batch['name']
    for c in base_outpath.iterdir():
        repo_id = f"sd-concepts-library/{c.stem}"
        text_file = open(c/"token_identifier.txt", "w")
        text_file.write(f"<{c.stem}>")
        text_file.close()
        operations = [
            CommitOperationAdd(path_in_repo="learned_embeds.bin", path_or_fileobj=str(c/"learned_embeds.bin")),
            CommitOperationAdd(path_in_repo="token_identifier.txt", path_or_fileobj=str(c/"token_identifier.txt")),
        ]
        create_repo(repo_id=repo_id,exist_ok=True)
        print(repo_id)
        api = HfApi()
        api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=f"Upload the concept <{c.stem}> embeds and token",
            token=hf_token
        )
            