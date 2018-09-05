spk2utt = si_df.groupby('spk')['uttr_id'].apply(lambda x: x.tolist()).to_dict()
with open("spk2utt", "w") as f:
    for k, v in spk2utt.items():
        line = "{}\t{}\n".format(k, " ".join(v))
        f.write(line)



