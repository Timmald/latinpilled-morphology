def swap_cols(path_to_tsv):
    with open(path_to_tsv,"r") as reader:
        lines = reader.readlines()
        newlines = [line.split("\t")[0]+"\t"+line.split("\t")[2][:-1]+"\t"+line.split("\t")[1]+"\n" for line in lines]
        reader.close()
    with open(path_to_tsv,"w") as writer:
        writer.writelines(newlines)
        writer.close()

swap_cols("Latin_stuff/lat.trn")