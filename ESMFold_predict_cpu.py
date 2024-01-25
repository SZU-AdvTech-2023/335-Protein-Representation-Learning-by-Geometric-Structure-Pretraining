import torch
import esm
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model = esm.pretrained.esmfold_v1()
model = model.eval().cpu().float()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
#model.set_chunk_size(16)

# Multimer prediction can be done with chains separated by ':'
count=1
with open("/home2/hzeng/anaconda3/envs/esm/src/temp/undone_prot.txt",'r') as file:
    for line in file:
        if line.startswith('>'):
            prot_name=line[1:-1]
            continue
        else:
            print(f"[{count}] {prot_name}: structure predicting...")
            count+=1
            sequence=line[:-1]
            print(f"sequence length:{len(sequence)}")
            with torch.no_grad():
                try:
                    output = model.infer_pdb(sequence)
                except RuntimeError:
                    print("RuntimeError")
                    continue
                except Exception as e:
                    print(e)
                    continue
            pdb_file="/home2/hzeng/anaconda3/envs/esm/src/results/pdb/"+prot_name+".pdb"
            with open(pdb_file, "w") as f:
                f.write(output)
                
            print("pLDDT computing...")
            import biotite.structure.io as bsio
            struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
            with open("/home2/hzeng/anaconda3/envs/esm/src/results/pLDDT.txt","a") as f:
                f.write(prot_name+', ')
                f.write(str(struct.b_factor.mean()))  # this will be the pLDDT
                f.write('\n')